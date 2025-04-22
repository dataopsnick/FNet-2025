# FNet-2025

Implementation of FNet: Mixing Tokens with Fourier Transforms [1] (https://arxiv.org/abs/2105.03824) in PyTorch w/ a basic GLUE benchmarking script using the `transformers` library.

## Paper Citation

[1] James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, Santiago Ontanon

arXiv:2105.03824v4 [cs.CL] 26 May 2022 

https://doi.org/10.48550/arXiv.2105.03824

## Repository Contents

**1. FNet Model Implementation (fnet_model.py)**

**2. GLUE Benchmarking Script (run_glue_fnet.py)**

## 3. How to Run

1.  **Install Dependencies:**
    ```bash
    pip install torch transformers datasets evaluate accelerate # Accelerate needed for Trainer
    pip install scikit-learn scipy # Needed for some GLUE metrics
    pip install evaluate==0.4.0 # or a higher compatible version
    ```
2.  **Run Fine-tuning:** Execute the script from your terminal. You need to specify the task name and an output directory.

    *Example: Fine-tune FNet-Base on CoLA*
    ```bash
    python run_glue_fnet.py \
      --model_name_or_path fnet-base \
      --tokenizer_name bert-base-uncased \
      --task_name cola \
      --do_train \
      --do_eval \
      --max_seq_length 128 \
      --per_device_train_batch_size 32 \
      --learning_rate 2e-5 \
      --num_train_epochs 3 \
      --output_dir /tmp/fnet_cola_output \
      --overwrite_output_dir \
      --logging_steps 100 \
      --evaluation_strategy epoch \
      --save_strategy epoch \
      --load_best_model_at_end True \
      --metric_for_best_model matthews_correlation # Metric specific to CoLA
      # Add --fp16 if you have CUDA and want mixed precision
    ```

    *Example: Fine-tune FNet-Base on MRPC*
    ```bash
    python run_glue_fnet.py \
      --model_name_or_path fnet-base \
      --tokenizer_name bert-base-uncased \
      --task_name mrpc \
      --do_train \
      --do_eval \
      --max_seq_length 128 \
      --per_device_train_batch_size 32 \
      --learning_rate 2e-5 \
      --num_train_epochs 3 \
      --output_dir /tmp/fnet_mrpc_output \
      --overwrite_output_dir \
      --logging_steps 50 \
      --evaluation_strategy epoch \
      --save_strategy epoch \
      --load_best_model_at_end True \
      --metric_for_best_model combined_score # F1/Accuracy average for MRPC
    ```

**Explanation:**

1.  **`fnet_model.py`:**
    *   Defines `FNetConfig` inheriting from `PretrainedConfig` for compatibility.
    *   Implements the core `FourierTransformLayer` using `torch.fft.fftn().real`.
    *   Includes standard `FeedForwardLayer` and `FNetEmbeddings`.
    *   `FNetEncoderBlock` combines the Fourier and FF layers with residuals and LayerNorm.
    *   `FNetModel` stacks the encoder blocks and provides the base model output.
    *   `FNetForSequenceClassification` adds the classification head on top of the pooled output (first token's hidden state) and handles loss calculation. It inherits from `PreTrainedModel` for easy integration with `Trainer`.
2.  **`run_glue_fnet.py`:**
    *   Uses `HfArgumentParser` for command-line arguments.
    *   Loads the specified GLUE dataset using `datasets`.
    *   Loads a *standard* tokenizer (like BERT's) because FNet doesn't define its own vocabulary.
    *   Creates an `FNetConfig` instance. You can select predefined "fnet-base" / "fnet-large" sizes or load a saved one. Overrides from command line are possible.
    *   Instantiates `FNetForSequenceClassification` using the config. Crucially, since `fnet-base` isn't on the Hub, `from_pretrained` will *initialize new weights* based on the `config`, unless you provide a path to a previously saved FNet model.
    *   Preprocesses the data using the loaded tokenizer.
    *   Defines `compute_metrics` using the `datasets` library's GLUE metric loader.
    *   Uses the standard `Trainer` API to handle the training and evaluation loop.

This setup provides a functional PyTorch FNet implementation and allows you to benchmark it against standard GLUE tasks using the familiar `transformers` ecosystem. Remember that the performance will depend heavily on hyperparameters, training data (this uses standard GLUE), and the specific task.
