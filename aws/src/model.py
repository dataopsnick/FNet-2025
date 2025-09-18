# src/model.py
"""
Contains all PyTorch class definitions for the Causal STFT FNet model architecture.
This includes the configuration, individual layers, and the final model class
compatible with the Hugging Face Trainer.
"""

import torch
import torch.nn as nn
import torch.fft
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutput
from transformers.generation import GenerationMixin
from transformers import AutoModelForCausalLM

# --- FNet Configuration Class ---
class FNetConfig(PretrainedConfig):
    """
    Configuration class for the Causal STFT FNet model.
    Inherits from Hugging Face's PretrainedConfig to be compatible with the ecosystem.
    """
    model_type = "causal_stft_fnet"

    def __init__(
        self,
        vocab_size=50257,
        hidden_size=768,
        num_hidden_layers=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        max_position_embeddings=1024,
        stft_window_size=64,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=50256,
        tie_word_embeddings=True,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.stft_window_size = stft_window_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

# --- FNet Model Architecture Components ---
class CausalSTFTLayer(nn.Module):
    """
    The core token-mixing layer using a Short-Time Fourier Transform (STFT)
    over a sliding, causally-masked window.
    """
    def __init__(self, config: FNetConfig):
        super().__init__()
        self.window_size = config.stft_window_size
        # The projection layer maps the flattened FFT window back to the hidden size.
        self.projection = nn.Linear(config.stft_window_size * config.hidden_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        
        # Pad the sequence on the left (the "past") to ensure the first token has a full window.
        padded_x = F.pad(x, (0, 0, self.window_size - 1, 0))

        # Use efficient striding to create overlapping windows without duplicating memory.
        windows = padded_x.as_strided(
            size=(batch_size, seq_len, self.window_size, hidden_size),
            stride=(padded_x.stride(0), padded_x.stride(1), padded_x.stride(1), padded_x.stride(2))
        )

        # Apply 2D FFT across the window and hidden dimensions.
        fft_windows = torch.fft.fftn(windows, dim=(-2, -1))

        # Take the real part and flatten the window and hidden dimensions.
        fft_windows_real = fft_windows.real
        fft_windows_flat = fft_windows_real.view(batch_size, seq_len, -1)

        # Project back to the original hidden_size.
        return self.projection(fft_windows_flat)

class FeedForwardLayer(nn.Module):
    """Standard two-layer feed-forward network with GELU activation."""
    def __init__(self, config: FNetConfig):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation = nn.GELU()
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        return self.dropout(self.dense2(self.activation(self.dense1(x))))

class CausalFNetEncoderBlock(nn.Module):
    """A single block of the FNet decoder, with pre-norm residual connections."""
    def __init__(self, config: FNetConfig):
        super().__init__()
        self.causal_stft = CausalSTFTLayer(config)
        self.ffn = FeedForwardLayer(config)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        # Pre-norm residual connection for the STFT layer
        x = x + self.dropout(self.causal_stft(self.norm1(x)))
        # Pre-norm residual connection for the FFN layer
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

class FNetEmbeddings(nn.Module):
    """Handles word and positional embeddings."""
    def __init__(self, config: FNetConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.pos_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False)

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        pos_ids = self.position_ids[:, :seq_len]
        embeds = self.word_embeddings(input_ids) + self.pos_embeddings(pos_ids)
        return self.dropout(self.norm(embeds))

# --- Main FNet Model for Causal Language Modeling ---
class CausalFNetForCausalLM(PreTrainedModel, GenerationMixin):
    """The complete Causal STFT FNet model for autoregressive generation."""
    config_class = FNetConfig

    def __init__(self, config: FNetConfig):
        super().__init__(config)
        self.embeddings = FNetEmbeddings(config)
        self.encoder = nn.ModuleList([CausalFNetEncoderBlock(config) for _ in range(config.num_hidden_layers)])
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init() # Ties weights if tie_word_embeddings is True

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(self, input_ids, labels=None, **kwargs):
        x = self.embeddings(input_ids)
        for block in self.encoder:
            x = block(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
        return CausalLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs: Any):
        # FNet doesn't have a past_key_values cache like Transformers,
        # so we just pass the input_ids.
        return {"input_ids": input_ids}

# --- Register with Hugging Face AutoModel ---
# This line is crucial for allowing `AutoModelForCausalLM.from_pretrained` to work.
AutoModelForCausalLM.register(FNetConfig, CausalFNetForCausalLM)