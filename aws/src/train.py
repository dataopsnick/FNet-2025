# src/train.py
import argparse
import os
import json
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from src.model import CausalFNetForCausalLM, FNetConfig
from src.data import get_tokenized_dataset
#import smdistributed.dataparallel.torch.torch_smddp # For distributed training

def main():
    
    parser = argparse.ArgumentParser()
    
    # --- SageMaker environment variables ---
    # SageMaker provides paths for model output, data channels, etc.
    parser.add_argument("--model-dir", type=str, default=os.getenv("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.getenv("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.getenv("SM_CHANNEL_TEST"))

    # --- Model Hyperparameters (passed from the SageMaker Tuner) ---
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=6)
    parser.add_argument("--intermediate_size", type=int, default=2048)
    parser.add_argument("--stft_window_size", type=int, default=256)
    
    # --- Training Hyperparameters ---
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--num_train_epochs", type=int, default=15)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)

    args, _ = parser.parse_known_args()
    script_dir = os.path.dirname(__file__)
    deepspeed_config_path = os.path.join(script_dir, "ds_config.json")
    with open(deepspeed_config_path, "r") as f:
        deepspeed_config = json.load(f)

    print("--- Loading Dataset ---")
    # We will load the dataset from the path SageMaker provides
    tokenized_datasets, tokenizer = get_tokenized_dataset(tokenizer_name="gpt2")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("--- Configuring Model ---")
    config = FNetConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        intermediate_size=args.intermediate_size,
        stft_window_size=args.stft_window_size,
    )
    model = CausalFNetForCausalLM(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"--- Model Initialized ---")
    print(f"Parameters: {total_params / 1e6:.2f}M")
    print(f"Config: H={config.hidden_size}, L={config.num_hidden_layers}, W={config.stft_window_size}")

    print("--- Setting up Training ---")
    # SageMaker requires model artifacts to be saved to SM_MODEL_DIR
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        bf16=True,
        report_to="none",
        deepspeed=deepspeed_config,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("--- Starting Training ---")
    trainer.train()

    print("--- Training Complete ---")
    # Save the final model to the path SageMaker expects
    trainer.save_model(os.path.join(args.model_dir, "best_model"))

if __name__ == "__main__":
    main()