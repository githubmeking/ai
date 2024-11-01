from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Model ve tokenizer'ı yükleme
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Giriş metni (Örnek olarak Fibonacci serisini hesaplayan bir fonksiyon başlatıyoruz)
input_text = "def fibonacci(n):"
inputs = tokenizer(input_text, return_tensors="pt")

# Modelin devam eden bir kod üretmesi için parametreler
output = model.generate(
    inputs['input_ids'], 
    max_new_tokens=50,         # Üreteceği token sayısı
    temperature=0.7,           # Daha yaratıcı bir çıktı için sıcaklık değeri
    repetition_penalty=1.2,    # Tekrarları önlemek için ceza
    top_k=50,                  # Orta seviye kelime havuzu seçimi
    top_p=0.9,                 # En yüksek olasılıklı kelimelerden seçim
    do_sample=True             # Örneklemeyi etkinleştirir
)

# Çıktıyı çözümleme ve yazdırma
generated_code = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_code)
