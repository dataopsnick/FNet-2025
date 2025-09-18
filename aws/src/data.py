# src/data.py
"""
Handles loading, preprocessing, and tokenizing the GSM8K dataset for model training.
"""

from datasets import load_dataset
from transformers import AutoTokenizer

def get_tokenized_dataset(tokenizer_name="gpt2", max_length=512):
    """
    Loads the GSM8K dataset, formats it, and tokenizes it for causal language modeling.

    Args:
        tokenizer_name (str): The name of the tokenizer to use from the Hugging Face Hub.
        max_length (int): The maximum sequence length for truncation and padding.

    Returns:
        tuple: A tuple containing:
            - tokenized_datasets (datasets.DatasetDict): The processed dataset.
            - tokenizer (transformers.PreTrainedTokenizer): The tokenizer instance.
    """
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # For causal language models, the pad token is often set to the end-of-sequence token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("Loading GSM8K dataset...")
    raw_datasets = load_dataset("gsm8k", "main")

    def preprocess_function(examples):
        """Applies the standard Question/Answer formatting and tokenization."""
        # Format each example into a single string for the language model.
        # The EOS token signals the end of a complete sequence.
        text = [f"Question: {q}\nAnswer: {a}{tokenizer.eos_token}"
                for q, a in zip(examples['question'], examples['answer'])]
        
        # Tokenize the formatted text.
        return tokenizer(
            text, 
            truncation=True, 
            max_length=max_length, 
            padding="max_length"
        )

    print("Mapping and tokenizing the dataset...")
    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=4,  # Use multiple processes to speed up mapping
        remove_columns=raw_datasets["train"].column_names # Clean up old columns
    )
    
    print(f"Dataset ready. Train samples: {len(tokenized_datasets['train'])}, Test samples: {len(tokenized_datasets['test'])}")
    return tokenized_datasets, tokenizer