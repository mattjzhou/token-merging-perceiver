from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer, Trainer, default_data_collator, EvalPrediction, HfArgumentParser, TrainingArguments
from datasets import load_dataset
import numpy as np
import tome
from dataclasses import dataclass, field
import evaluate
from typing import Optional
import torch
    
@dataclass
class ToMeArguments:
    model_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    task_name: str = field(
        metadata={"help": "The name of the task to train on: "},
    )
    r: int = field(
        metadata={"help": "The number of tokens merged using ToMe per layer"},
    )
    max_seq_length: int = field(
        default=2048,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )

if __name__ == '__main__':
    parser = HfArgumentParser((ToMeArguments, TrainingArguments))
    tome_args, training_args = parser.parse_args_into_dataclasses()
    print(f"Tome parameters {tome_args}\nTraining/eval parameters {training_args}")

    dataset = load_dataset("nyu-mll/glue", tome_args.task_name)
    if tome_args.task_name == "stsb":
        num_labels = 1
    else:
        num_labels = len(dataset["train"].features["label"].names)
    config = AutoConfig.from_pretrained(tome_args.model_path,
            num_labels=num_labels,
            finetuning_task=tome_args.task_name)
    tokenizer = AutoTokenizer.from_pretrained(tome_args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(tome_args.model_path,
            config=config)
    sentence1_key, sentence2_key = "sentence1", "sentence2"
    padding = "max_length" if tome_args.pad_to_max_length else False
    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=tome_args.max_seq_length, truncation=True)
        return result
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    print(model)
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    metric = evaluate.load("glue", tome_args.task_name)
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if tome_args.task_name == "stsb" else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    metrics["eval_samples"] = len(eval_dataset)
    print(metrics)
    baseline_throughput = tome.utils.benchmark(
        model,
        device="cuda:0",
        verbose=True,
        runs=50,
        batch_size=256,
        input_size=(len(dataset["train"][0]["input_ids"]),),
    )

    tome.patch.perceiver(model)
    model.r = tome_args.r
    print(model)
    tome_throughput = tome.utils.benchmark(
        model,
        device="cuda:0",
        verbose=True,
        runs=50,
        batch_size=256,
        input_size=(len(dataset["train"][0]["input_ids"]),),
    )
    print(f"Throughput improvement: {tome_throughput / baseline_throughput:.2f}x")
    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    metrics["eval_samples"] = len(eval_dataset)
    print(metrics)
    
    train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.save_model()
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    print(metrics)
    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    metrics["eval_samples"] = len(eval_dataset)
    trainer.save_metrics("eval", metrics)
    print(metrics)