from transformers import AutoTokenizer, AutoModelForCausalLM

# Modeli ve tokenizer'ı yükleyin
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

# Kod üretmek için giriş cümlesi
input_text = "def fibonacci(n):"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Kod üretimi
output = model.generate(input_ids, max_length=50, temperature=0.7, repetition_penalty=1.2)
generated_code = tokenizer.decode(output[0], skip_special_tokens=True)

# Üretilen kodu yazdır
print(generated_code)
