# Causal STFT FNet: An Autoregressive Decoder with Knowledge Distillation

This repository contains a PyTorch implementation of a novel **Causal Short-Time Fourier Transform (STFT) FNet decoder**. This work extends the original FNet architecture [](#references), which focused on encoders, to the domain of autoregressive language modeling.

The entire implementation, including model definition, training, and inference, is contained within the main Jupyter Notebook.

<a target="_blank" href="https://colab.research.google.com/github/dataopsnick/FNet-2025/blob/main/Causal_STFT_FNet_Student_Teacher_Knowledge_Distillation.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## 📋 Table of Contents
- [Overview](#-overview)
- [The Decoder Challenge](#-the-decoder-challenge)
- [Key Contributions](#-key-contributions)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Architecture Details](#-architecture-details)
-   [Causal STFT Layer](#causal-stft-layer)
-   [Knowledge Distillation](#knowledge-distillation)
- [Experiments in the Notebook](#-experiments-in-the-notebook)
- [Citation](#-citation)
- [References](#-references)

## 🔍 Overview

The original FNet paper demonstrated that replacing self-attention sublayers with unparameterized Fourier transforms could achieve 92-97% of BERT's accuracy on GLUE benchmarks while training significantly faster. However, its focus was exclusively on encoder architectures.

This project addresses the challenge of creating a practical FNet-style **decoder** for generative tasks. We introduce a `CausalSTFTLayer` that uses a sliding window Fourier Transform to mix tokens, enabling efficient and causally correct autoregressive generation. Furthermore, we demonstrate how to effectively train this student model on a mathematical reasoning dataset (GSM8K) by distilling knowledge from a powerful teacher model, **Qwen2-0.5B-Instruct**.

## ✨ Key Contributions

1.  **Novel Causal STFT Layer**: Unlike the original FNet, which applies a 2D FFT across the entire sequence, this implementation uses a windowed Short-Time Fourier Transform with causal masking. This maintains O(N log N) complexity within each window while enabling autoregressive generation.

2.  **Practical Decoder Architecture**: The notebook provides a concrete, working implementation of a causal FNet decoder for language modeling, addressing the challenge left open by the original paper.

3.  **Knowledge Distillation Framework**: It demonstrates how to train this novel architecture using knowledge distillation from a pre-trained transformer. A custom `DistillationTrainer` is implemented to handle the combined loss function (Cross-Entropy and KL-Divergence).

4.  **Generative Task Application**: The model is trained and evaluated on the GSM8K mathematical reasoning dataset, showcasing its capability on structured generative tasks, a departure from the classification-focused GLUE benchmarks.

## 📁 Repository Structure

The project is simplified into a single, self-contained notebook.

```
FNet-2025/
│
└── Causal_STFT_FNet_Student_Teacher_Knowledge_Distillation.ipynb  # Core implementation, training, and inference
│
└── README.md                                                        # This file
```

## 🚀 Getting Started

The easiest way to run this project is by using Google Colab.

1.  **Open in Colab**: Click the "Open In Colab" badge at the top of this README.
2.  **Install Dependencies**: The first code cell in the notebook contains the necessary `pip install` command to set up the environment.
    ```bash
    !pip install transformers>=4.38.0 datasets==2.18.0 accelerate>=0.21.0 evaluate==0.4.1 torch peft==0.10.0 sentencepiece -q
    ```
3.  **Run the Cells**: Execute the notebook cells sequentially to define the model, load the data, and run the training and inference experiments. A GPU runtime (like the A100 provided in Colab Pro) is recommended, especially for the knowledge distillation part.

## 🏗️ Architecture Details

### Causal STFT Layer

The core innovation is the `CausalSTFTLayer`. To maintain causality (i.e., a token at position `t` cannot see future tokens), we use a sliding window approach:

1.  **Padding**: The input sequence is padded on the left.
2.  **Striding**: `as_strided` is used to create overlapping windows for each token, where each window contains the current token and the `window_size - 1` preceding tokens.
3.  **FFT**: A 2D Fast Fourier Transform (`torch.fft.fftn`) is applied to each window across both the sequence and hidden dimensions.
4.  **Projection**: The resulting transformed windows are flattened and projected back to the original hidden size with a linear layer.

This allows the model to learn local frequency-domain relationships while respecting the autoregressive nature of text generation.

### Knowledge Distillation

To train the `CausalFNetForCausalLM` student model effectively, we use knowledge distillation. A custom `DistillationTrainer` minimizes a combined loss:

-   **Student Loss (Hard Loss)**: Standard Cross-Entropy loss between the student's predictions and the true labels.
-   **Distillation Loss (Soft Loss)**: Kullback-Leibler (KL) divergence loss between the softened probability distributions of the teacher and student logits.

The final loss is a weighted average: `Loss = α * StudentLoss + (1 - α) * DistillationLoss`.

## Experiments in the Notebook

The notebook contains two primary, end-to-end experiments:

1.  **Training from Scratch**: The `CausalFNetForCausalLM` model is trained directly on the GSM8K dataset. This establishes a baseline for the architecture's learning capability.

2.  **Knowledge Distillation with Qwen2**: A larger `CausalFNetForCausalLM` student is trained using a pre-trained **Qwen/Qwen2-0.5B-Instruct** model as the teacher. This demonstrates a practical path to achieving competitive performance with the novel architecture.

Both sections include model definition, data preprocessing, training loops, and sample inference to showcase the models' generative capabilities.

## 📝 Citation

If you use this implementation in your research, please consider citing the original FNet paper:

```bibtex
@article{lee2021fnet,
  title={FNet: Mixing Tokens with Fourier Transforms},
  author={Lee-Thorp, James and Ainslie, Joshua and Eckstein, Ilya and Ontanon, Santiago},
  journal={arXiv preprint arXiv:2105.03824},
  year={2021}
}
```

## 📚 References

 James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, Santiago Ontanon. "FNet: Mixing Tokens with Fourier Transforms." arXiv:2105.03824v4 [cs.CL] 26 May 2022. [https://doi.org/10.48550/arXiv.2105.03824](https://doi.org/10.48550/arXiv.2105.03824)

---
## 📊 Performance and Observations

The experiments in the notebook serve as a proof-of-concept for the architecture and training methodology.

*   **Training Dynamics**:
    *   **From Scratch**: The model trained from scratch shows a steady decrease in validation loss, indicating that the architecture is capable of learning. However, the generated output is largely incoherent, which is expected for a small model trained on a limited dataset without pre-training.
    *   **Knowledge Distillation**: The distilled model also shows a consistent decrease in validation loss. While the final generated output is not mathematically correct, it demonstrates that the student model is successfully learning from the teacher's probability distributions. The training process is more stable and effective compared to training from scratch.

*   **Generation Quality**: The primary goal of this notebook is to demonstrate a working causal FNet decoder and a distillation pipeline. The generated text from the trained models is not yet state-of-the-art. Achieving high-quality generation would require extensive pre-training on a large corpus, which is beyond the scope of this implementation.

*   **Resource Management**: The knowledge distillation experiment, which involves a 0.5B parameter teacher model and a custom student, is designed to be runnable in a standard Google Colab environment. This is made possible by leveraging several key `transformers` features:
    *   **Mixed Precision (`fp16`)**: Reduces memory usage and speeds up computation on compatible GPUs.
    *   **Gradient Checkpointing**: Trades compute for a significant reduction in memory, allowing for larger models and batch sizes.
    *   **`device_map` and `low_cpu_mem_usage`**: Intelligently loads the model onto the available hardware to minimize memory spikes.

## 🚨 Important Notes & Differences from Original FNet

This implementation differs significantly from the original FNet paper and its official HuggingFace counterpart.

1.  **Architecture**: This is a **decoder-only** model designed for autoregressive tasks, whereas the original FNet was an **encoder**.
2.  **Token Mixing**: We use a **Causal Short-Time Fourier Transform (STFT)** with a sliding window, not a full-sequence 2D FFT. This is the core innovation that enforces causality.
3.  **Training Method**: The primary training method explored is **knowledge distillation** from a pre-trained teacher, not pre-training on a massive corpus followed by fine-tuning.
4.  **Task**: The focus is on **autoregressive text generation** (on the GSM8K dataset) rather than NLU/classification tasks (on the GLUE benchmark).
5.  **Initialization**: The student model is always initialized with random weights.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or find any issues, please feel free to open an issue or submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

*   Thanks to the original FNet authors for their innovative approach to token mixing.
*   The HuggingFace team for their excellent `transformers`, `datasets`, and `accelerate` libraries.
*   The PyTorch team for their robust and easy-to-use FFT implementations.
*   The Alibaba Cloud team for the powerful and open-source Qwen2 models.
*   The creators of the GSM8K dataset for providing a challenging benchmark for mathematical reasoning.