from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer
from calflops import calculate_flops
import tome

batch_size, max_seq_length = 1, 128
model_path = "./STSB_b32_l5e5/checkpoint-3600"
config = AutoConfig.from_pretrained(model_path,
        num_labels=1,
        finetuning_task="stsb")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path,
        config=config)

flops, macs, params = calculate_flops(model=model, 
                                      input_shape=(batch_size,max_seq_length),
                                      transformer_tokenizer=tokenizer)
print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))

tome.patch.perceiver(model)
model.r = 2
flops, macs, params = calculate_flops(model=model, 
                                      input_shape=(batch_size,max_seq_length),
                                      transformer_tokenizer=tokenizer)
print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))