# src/data.py
from datasets import load_dataset
from transformers import AutoTokenizer

def get_tokenized_dataset(tokenizer_name="gpt2", max_length=512):
    """Loads and preprocesses the GSM8K dataset."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    raw_datasets = load_dataset("gsm8k", "main")

    def preprocess_function(examples):
        text = [f"Question: {q}\nAnswer: {a}{tokenizer.eos_token}"
                for q, a in zip(examples['question'], examples['answer'])]
        return tokenizer(text, truncation=True, max_length=max_length, padding="max_length")

    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=raw_datasets["train"].column_names
    )
    return tokenized_datasets, tokenizer