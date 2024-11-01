from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Model ve tokenizer'ı yükleme
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

# Girdi metni
input_text = "def fibonacci(n):"
inputs = tokenizer(input_text, return_tensors="pt")

# Yaratıcılığı artırmak için parametreler
output = model.generate(
    inputs['input_ids'], 
    attention_mask=inputs['attention_mask'], 
    max_length=100,             # Daha uzun bir çıktı için max_length artırılabilir
    temperature=1.5,            # Yaratıcılığı artırır
    repetition_penalty=1.1,     # Tekrarları azaltmak için hafif bir ceza
    top_k=0,                    # Sınırsız seçim havuzu (daha fazla yaratıcılık için)
    top_p=0.9                   # Daha geniş seçim olasılığı
)

# Çıktıyı çözümleme ve yazdırma
generated_code = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_code)
