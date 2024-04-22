from transformers import AutoModelForImageClassification, AutoConfig, AutoImageProcessor, Trainer, default_data_collator, EvalPrediction, HfArgumentParser, TrainingArguments
from datasets import load_dataset
import numpy as np
import tome
from dataclasses import dataclass, field
import evaluate
from typing import Optional
import torch
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
    
@dataclass
class ToMeArguments:
    model_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    dataset_name: str = field(
        metadata={"help": "The name of the dataset to train on: "},
    )
    r: int = field(
        metadata={"help": "The number of tokens merged using ToMe per layer"},
    )
    image_column_name: str = field(
        default="image",
        metadata={"help": "The name of the dataset column containing the image data. Defaults to 'image'."},
    )
    label_column_name: str = field(
        default="label",
        metadata={"help": "The name of the dataset column containing the labels. Defaults to 'label'."},
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )

if __name__ == '__main__':
    parser = HfArgumentParser((ToMeArguments, TrainingArguments))
    tome_args, training_args = parser.parse_args_into_dataclasses()
    print(f"Tome parameters {tome_args}\nTraining/eval parameters {training_args}")

    dataset = load_dataset(tome_args.dataset_name)
    dataset_column_names = dataset["train"].column_names
    split = dataset["train"].train_test_split(tome_args.train_val_split)
    dataset["train"] = split["train"]
    dataset["validation"] = split["test"]
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example[tome_args.label_column_name] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}
    labels = dataset["train"].features[tome_args.label_column_name].names
    label2id, id2label = {}, {}
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
    metric = evaluate.load("accuracy")
    def compute_metrics(p):
        """Computes accuracy on a batch of predictions"""
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

    config = AutoConfig.from_pretrained(tome_args.model_path,
            num_labels=len(labels),
            label2id=label2id,
            id2label=id2label,
            finetuning_task="image-classification")
    model = AutoModelForImageClassification.from_pretrained(tome_args.model_path,
            config=config)
    image_processor = AutoImageProcessor.from_pretrained(tome_args.model_path)
    
    if "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
    else:
        size = (image_processor.size["height"], image_processor.size["width"])
    normalize = (
        Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        if hasattr(image_processor, "image_mean") and hasattr(image_processor, "image_std")
        else Lambda(lambda x: x)
    )
    _train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )
    _val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(size),
            ToTensor(),
            normalize,
        ]
    )

    def train_transforms(example_batch):
        """Apply _train_transforms across a batch."""
        example_batch["pixel_values"] = [
            _train_transforms(pil_img.convert("RGB")) for pil_img in example_batch[tome_args.image_column_name]
        ]
        return example_batch

    def val_transforms(example_batch):
        """Apply _val_transforms across a batch."""
        example_batch["pixel_values"] = [
            _val_transforms(pil_img.convert("RGB")) for pil_img in example_batch[tome_args.image_column_name]
        ]
        return example_batch
    dataset["train"].set_transform(train_transforms)
    dataset["validation"].set_transform(val_transforms)
    print(model)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        tokenizer=image_processor,
        data_collator=collate_fn,
    )
    metrics = trainer.evaluate()
    print(metrics)
    baseline_throughput = tome.utils.benchmark(
        model,
        device="cuda:0",
        verbose=True,
        runs=50,
        batch_size=128,
        input_size=(3, 224, 224),
    )

    tome.patch.perceiver(model)
    model.r = tome_args.r
    print(model)
    tome_throughput = tome.utils.benchmark(
        model,
        device="cuda:0",
        verbose=True,
        runs=50,
        batch_size=128,
        input_size=(3, 224, 224),
    )
    print(f"Throughput improvement: {tome_throughput / baseline_throughput:.2f}x")
    metrics = trainer.evaluate()
    print(metrics)
    
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.save_model()
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    print(metrics)
    metrics = trainer.evaluate()
    trainer.save_metrics("eval", metrics)
    print(metrics)