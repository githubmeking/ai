from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Model ve tokenizer'ı yükleme
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

# Girdi metni
input_text = "def fibonacci(n):"
inputs = tokenizer(input_text, return_tensors="pt")

# Yaratıcılığı dengeli bir şekilde artırmak için parametreler
output = model.generate(
    inputs['input_ids'], 
    attention_mask=inputs['attention_mask'], 
    max_length=100,          
    temperature=1.2,         # Yaratıcılığı artırırken daha yapılandırılmış çıktılar için ayar
    repetition_penalty=1.1,  
    top_k=50,                # Daha yapılandırılmış sonuçlar için sınırlı seçim havuzu
    top_p=0.9,              
    do_sample=True           
)

# Çıktıyı çözümleme ve yazdırma
generated_code = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_code)
