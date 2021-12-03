from transformers import GPT2LMHeadModel, GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
sequence = ("He began his premiership by forming a five-man war cabinet")
inputs = tokenizer.encode(sequence, return_tensors='pt')
outputs = model.generate(inputs, max_length=100, do_sample=True)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)