
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from configuration_deepseek import DeepseekConfig
from modeling_deepseek import DeepseekForCausalLM

import sys


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = DeepseekForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda:0")
model.generation_config = DeepseekConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
inputs = tokenizer(text, return_tensors="pt")
# outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)
model.eval()
outputs = model(**inputs.to(model.device))

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)