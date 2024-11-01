from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Model ve tokenizer'ı yükleme
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

# Girdi metni
input_text = "def fibonacci_toplami(n):  # Bu fonksiyon, Fibonacci dizisindeki ilk n sayısının toplamını hesaplar."
inputs = tokenizer(input_text, return_tensors="pt")

# Daha anlamlı bir çıktı almak için optimize edilmiş ayarlar
output = model.generate(
    inputs['input_ids'], 
    attention_mask=inputs['attention_mask'], 
    max_new_tokens=500,      
    temperature=0.3,        # Daha yapılandırılmış bir çıktı için düşük temperature
    repetition_penalty=1.2, 
    top_k=50,               # Yapılandırılmış seçimler için orta seviye top_k
    top_p=0.85,             
    do_sample=True          
)

# Çıktıyı çözümleme ve yazdırma
generated_code = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_code)
