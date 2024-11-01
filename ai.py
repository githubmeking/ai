from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Model ve tokenizer'ı yükleme
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

# Girdi metni
input_text = "def fibonacci(n):"
inputs = tokenizer(input_text, return_tensors="pt")

# `attention_mask` ile çıktı üretme
output = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=50, temperature=0.7)

# Çıktıyı çözümleme ve yazdırma
generated_code = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_code)
