# src/data.py
"""
Handles loading, preprocessing, and tokenizing the GSM8K dataset for model training.
"""

from datasets import load_dataset
from transformers import AutoTokenizer

def get_tokenized_dataset(max_length=512):
    """Load and tokenize the WikiText dataset."""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"
    
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-v1")
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    # Apply tokenization
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # Filter out empty examples
    def filter_empty(example):
        return len(example["input_ids"]) > 0
    
    tokenized_datasets = tokenized_datasets.filter(filter_empty)
    
    return tokenized_datasets, tokenizer