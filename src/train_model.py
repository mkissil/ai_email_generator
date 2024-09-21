# src/train_model.py

import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

def train():
    # Загрузка токенизатора и модели
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Подготовка датасета
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path='../data/emails_dataset.txt',
        block_size=128
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # Параметры обучения
    training_args = TrainingArguments(
        output_dir='../models/email_generator_model',
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
    )

    # Инициализация тренера
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    # Обучение модели
    trainer.train()

    # Сохранение модели
    trainer.save_model('../models/email_generator_model')

if __name__ == "__main__":
    train()
