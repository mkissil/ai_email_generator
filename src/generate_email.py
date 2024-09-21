# src/generate_email.py

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_model():
    # Загрузка токенизатора и обученной модели
    tokenizer = GPT2Tokenizer.from_pretrained('../models/email_generator_model')
    model = GPT2LMHeadModel.from_pretrained('../models/email_generator_model')
    model.eval()
    return tokenizer, model

def generate_email(subject, keywords):
    tokenizer, model = load_model()

    # Формирование подсказки для модели
    prompt = f"<SUBJECT>{subject}</SUBJECT><KEYWORDS>{', '.join(keywords)}</KEYWORDS><EMAIL>"

    inputs = tokenizer.encode(prompt, return_tensors='pt')

    # Генерация текста
    outputs = model.generate(
        inputs,
        max_length=1024,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    # Декодирование и обработка результата
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_email = text.split('<EMAIL>')[-1]
    return generated_email.strip()

if __name__ == "__main__":
    subject = input("Введите тему письма: ")
    keywords = input("Введите ключевые слова (через запятую): ").split(',')

    email_content = generate_email(subject, keywords)
    print("\nСгенерированное письмо:")
    print(email_content)
