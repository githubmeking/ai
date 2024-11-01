from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Model ve tokenizer'ı yükleme
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

# Girdi metni
input_text = "def fibonacci_toplami(n):  # Bu fonksiyon Fibonacci dizisindeki ilk n sayısının toplamını hesaplar."
inputs = tokenizer(input_text, return_tensors="pt")

# max_new_tokens ile kısa ve yapılandırılmış bir çıktı almak için
output = model.generate(
    inputs['input_ids'], 
    attention_mask=inputs['attention_mask'], 
    max_new_tokens=30,      # max_length yerine max_new_tokens kullanarak hata önlenir
    temperature=0.3,        # Yaratıcılığı minimuma indirme
    repetition_penalty=1.2, 
    top_k=20,               
    top_p=0.75,             
    do_sample=True          
)

# Çıktıyı çözümleme ve yazdırma
generated_code = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_code)
