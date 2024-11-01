from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Model ve tokenizer'ı yükleme
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

# Girdi metni
input_text = "def fibonacci(n):"
inputs = tokenizer(input_text, return_tensors="pt")

# Daha yapılandırılmış ve tutarlı bir çıktı almak için ayarlar
output = model.generate(
    inputs['input_ids'], 
    attention_mask=inputs['attention_mask'], 
    max_length=50,           # Çıktı uzunluğunu sınırlı tutmak için
    temperature=0.4,         # Yaratıcılığı düşürerek daha tutarlı sonuçlar
    repetition_penalty=1.1,  
    top_k=30,                # Sınırlı seçim havuzu
    top_p=0.8,               # En yüksek olasılıklı kelimelerden seçim
    do_sample=True           
)

# Çıktıyı çözümleme ve yazdırma
generated_code = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_code)
