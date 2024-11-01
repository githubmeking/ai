from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Model ve tokenizer'ı yükleme
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

# Girdi metni
input_text = "def fibonacci(n):"
inputs = tokenizer(input_text, return_tensors="pt")

# Daha yapılandırılmış ve tutarlı bir çıktı almak için optimize edilmiş ayarlar
output = model.generate(
    inputs['input_ids'], 
    attention_mask=inputs['attention_mask'], 
    max_length=30,           # Çıktıyı kısa ve anlamlı tutmak için
    temperature=0.5,         # Daha düşük yaratıcılık ve daha yapılandırılmış sonuçlar
    repetition_penalty=1.15, # Tekrarları azaltma
    top_k=30,                
    top_p=0.85,              
    do_sample=True           
)

# Çıktıyı çözümleme ve yazdırma
generated_code = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_code)
