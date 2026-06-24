# test_llm.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "microsoft/Phi-3.5-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16).cuda()

prompt = "<|system|>\nYou are a helpful assistant.<|end|>\n<|user|>\nPlease locate the areas of the Knife with function of grasp.<|end|>\n<|assistant|>\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))