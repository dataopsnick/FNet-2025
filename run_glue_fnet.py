import os
import sys
import logging
import datasets
from datasets import load_dataset
from evaluate.loading import load as load_metric 

import transformers
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
)
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# Import our custom FNet model and config
from fnet_model import FNetConfig, FNetForSequenceClassification
from custom_fnet_model import CustomFNetForSequenceClassification


logger = logging.getLogger(__name__)

# --- Add this at the top ---
print("--- sys.argv ---")
print(sys.argv)
print("----------------")

# --- Inspect TrainingArguments ---
print("--- TrainingArguments Fields ---")
for name in dir(TrainingArguments):
    if not name.startswith("_"):
        print(name)
print("----------------")
# -------------------------

def get_huggingface_token():
    """Get the Hugging Face token with fallback mechanisms"""
    # First try to load from .env file
    try:
        from dotenv import load_dotenv
        import os
        load_dotenv()  # Load from .env file
        env_token = os.environ.get("HF_TOKEN")
        if env_token:
            print("Using token from .env file")
            return env_token
    except ImportError:
        print("python-dotenv not installed, trying other methods")
    except Exception as e:
        print(f"Error loading from .env: {e}")
    
    # Then try environment variables directly
    import os
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if env_token:
        print("Using token from environment variables")
        return env_token
    
    # Try to get from huggingface CLI cache
    try:
        import json
        from pathlib import Path
        home = Path.home()
        token_path = home / ".huggingface" / "token"
        if token_path.exists():
            with open(token_path, "r") as f:
                hf_cache_token = f.read().strip()
                if hf_cache_token:
                    print("Using token from HuggingFace CLI cache")
                    return hf_cache_token
    except Exception as e:
        print(f"Error getting token from cache: {e}")
    
    # If all else fails, prompt the user
    print("No HuggingFace token found in any source.")
    print("Please provide a token or set it in a .env file or environment variable.")
    return None


# --- Argument Parsing ---
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="fnet-base",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models (or 'fnet-base'/'fnet-large' for default configs)"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default="bert-base-uncased", metadata={"help": "Pretrained tokenizer name or path if not the same as model_name (FNet uses existing tokenizers)"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    # FNet specific config overrides
    hidden_size: Optional[int] = field(default=None, metadata={"help": "Override FNet hidden size"})
    num_hidden_layers: Optional[int] = field(default=None, metadata={"help": "Override FNet num layers"})
    intermediate_size: Optional[int] = field(default=None, metadata={"help": "Override FNet intermediate size"})

# Define the list of valid GLUE tasks
VALID_GLUE_TASKS = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task_name: Optional[str] = field(
        default="cola",
        metadata={
            "help": "The name of the glue task to train on.",
            "choices": VALID_GLUE_TASKS
        })
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

# GLUE task specifics
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    
    # Handle JSON config file if provided
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # Create a training args object first to check its fields
        try:
            dummy_training_args = TrainingArguments(output_dir="./dummy")
            print("--- TrainingArguments instance attributes ---")
            for key, value in vars(dummy_training_args).items():
                print(f"{key}: {value}")
            print("-" * 40)
        except Exception as e:
            print(f"Error inspecting training args: {e}")
        
        # Skip argument preprocessing and use manual creation
        training_args = TrainingArguments(
            output_dir="/tmp/fnet_cola_output",
            do_train=True,
            do_eval=True,
            per_device_train_batch_size=32,
            learning_rate=2e-5,
            num_train_epochs=3,
            # We'll try using eval_strategy and save_strategy instead
            eval_strategy="epoch",      # Try this instead of evaluation_strategy
            save_strategy="epoch",
            # Also try the boolean flag with value True
            load_best_model_at_end=True,
            metric_for_best_model="matthews_correlation",
            overwrite_output_dir=True,
            logging_steps=100
        )
        
        # Parse the other args normally
        args_to_parse = []
        for i, arg in enumerate(sys.argv):
            if arg.startswith("--") and arg not in ["--evaluation_strategy", "--save_strategy", "--load_best_model_at_end", "--output_dir", "--do_train", "--do_eval", "--per_device_train_batch_size", "--learning_rate", "--num_train_epochs", "--metric_for_best_model", "--overwrite_output_dir", "--logging_steps"]:
                args_to_parse.append(arg)
                if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
                    args_to_parse.append(sys.argv[i + 1])
        
        # Add required --output_dir for parsing
        args_to_parse.extend(["--output_dir", "/tmp/dummy"])
        
        try:
            print(f"Parsing reduced args: {args_to_parse}")
            model_parser = HfArgumentParser((ModelArguments,))
            data_parser = HfArgumentParser((DataTrainingArguments,))
            
            model_args, _ = model_parser.parse_args_into_dataclasses(args=args_to_parse, return_remaining_strings=True)
            data_args, _ = data_parser.parse_args_into_dataclasses(args=args_to_parse, return_remaining_strings=True)
        except Exception as e:
            print(f"Error during parsing: {e}")
            # Fallback to defaults
            model_args = ModelArguments(model_name_or_path="fnet-base", tokenizer_name="bert-base-uncased")
            data_args = DataTrainingArguments(task_name="cola")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log important info
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set random seed for reproducibility
    set_seed(training_args.seed)

    # --- Load Data ---
    print(f"task_name: {data_args.task_name}")
    raw_datasets = load_dataset("glue", data_args.task_name, cache_dir=model_args.cache_dir)
    is_regression = data_args.task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    # Use this token in your script
    token = get_huggingface_token()

    # When loading the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=token  # Use token instead of use_auth_token
    )

    # --- Load Model ---
    # Create FNet config
    # You can define base/large configs here or load from a json
    # --- Load Model ---
    print(f"Initializing FNet model with configuration: {model_args.model_name_or_path}")

    # Set configuration based on model size
    if model_args.model_name_or_path == "fnet-base":
        hidden_size = model_args.hidden_size or 768
        num_hidden_layers = model_args.num_hidden_layers or 12
        intermediate_size = model_args.intermediate_size or 3072
    elif model_args.model_name_or_path == "fnet-large":
        hidden_size = model_args.hidden_size or 1024
        num_hidden_layers = model_args.num_hidden_layers or 24
        intermediate_size = model_args.intermediate_size or 4096
    else:
        # Default to base if something else is specified
        hidden_size = model_args.hidden_size or 768
        num_hidden_layers = model_args.num_hidden_layers or 12
        intermediate_size = model_args.intermediate_size or 3072
        print(f"Using default 'base' configuration for unknown model: {model_args.model_name_or_path}")

    # Create config
    config = FNetConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        num_labels=num_labels,
        pad_token_id=tokenizer.pad_token_id,
        type_vocab_size=2
    )

    print(f"Created FNet config with: {num_hidden_layers} layers, {hidden_size} hidden size")

    # Create model directly from config (NO from_pretrained calls at all)
    #model = FNetForSequenceClassification(config)
    #print(f"Initialized FNet model from scratch with configuration")
    model = CustomFNetForSequenceClassification(config)
    print(f"Initialized custom FNet model from scratch with configuration")


    # --- Preprocess Data ---
    sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    padding = "max_length" if data_args.pad_to_max_length else False

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=data_args.max_seq_length, truncation=True)
        return result

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
    if data_args.task_name == "mnli":
        mnli_mismatched_eval_dataset = processed_datasets["validation_mismatched"]


    import random
    for index in random.sample(range(len(train_dataset)), min(3, len(train_dataset))):
        # Convert to regular Python int to avoid numpy.int64 issues
        index = int(index)  # Explicit conversion to Python int
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # --- Metrics ---
    metric = load_metric("glue", data_args.task_name, cache_dir=model_args.cache_dir)

    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)


    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- Training ---
    if training_args.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # --- Evaluation ---
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        # MNLI has two validation sets
        if data_args.task_name == "mnli":
            logger.info("*** Evaluate MNLI Mismatched ***")
            metrics_mm = trainer.evaluate(eval_dataset=mnli_mismatched_eval_dataset)
            max_eval_samples_mm = (
                 data_args.max_eval_samples if data_args.max_eval_samples is not None else len(mnli_mismatched_eval_dataset)
            )
            metrics_mm["eval_samples"] = min(max_eval_samples_mm, len(mnli_mismatched_eval_dataset))
            trainer.log_metrics("eval_mnli_mismatched", metrics_mm)
            trainer.save_metrics("eval_mnli_mismatched", metrics_mm)


    # --- Prediction ---
    if training_args.do_predict:
         logger.info("*** Predict ***")
         predict_dataset = processed_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
         if data_args.max_predict_samples is not None:
             predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

         predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
         predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

         output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{data_args.task_name}.txt")
         if trainer.is_world_process_zero():
             with open(output_predict_file, "w") as writer:
                 logger.info(f"***** Predict results {data_args.task_name} *****")
                 writer.write("index\tprediction\n")
                 for index, item in enumerate(predictions):
                     if is_regression:
                         writer.write(f"{index}\t{item:3.3f}\n")
                     else:
                         item = label_list[item]
                         writer.write(f"{index}\t{item}\n")

if __name__ == "__main__":
    main()
