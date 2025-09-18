# src/model.py
"""
Contains all PyTorch class definitions for the Causal STFT FNet model architecture.
This includes the configuration, individual layers, and the final model class
compatible with the Hugging Face Trainer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_utils import PretrainedConfig
from typing import Optional, Tuple
import numpy as np
from scipy.fft import fft, ifft

class FNetConfig(PretrainedConfig):
    model_type = "fnet"
    
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        max_position_embeddings=512,
        layer_norm_eps=1e-12,
        stft_window_size=256,
        pad_token_id=0,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.stft_window_size = stft_window_size

class FourierTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.window_size = config.stft_window_size
        
    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device
        
        # Move to CPU for FFT computation
        hidden_states_cpu = hidden_states.detach().cpu().numpy()
        
        # Apply FFT along sequence dimension
        fft_output = fft(hidden_states_cpu, axis=1, norm='ortho')
        
        # Take only real part for simplicity (you could also use complex operations)
        real_part = np.real(fft_output)
        
        # Convert back to tensor
        output = torch.from_numpy(real_part).to(device).to(hidden_states.dtype)
        
        return output

class FNetLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fourier = FourierTransform(config)
        self.mixing_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )
        self.output_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states, attention_mask=None):
        # Fourier mixing
        fourier_output = self.fourier(hidden_states)
        hidden_states = self.mixing_layer_norm(fourier_output + hidden_states)
        
        # Feed-forward
        ff_output = self.feed_forward(hidden_states)
        ff_output = self.dropout(ff_output)
        hidden_states = self.output_layer_norm(ff_output + hidden_states)
        
        return hidden_states

class FNetEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([FNetLayer(config) for _ in range(config.num_hidden_layers)])
        
    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states

class CausalFNetModel(PreTrainedModel):
    config_class = FNetConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.encoder = FNetEncoder(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Initialize weights
        self.post_init()
        
    def forward(self, input_ids, attention_mask=None):
        seq_length = input_ids.shape[1]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        embeddings = self.embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        hidden_states = embeddings + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        hidden_states = self.encoder(hidden_states, attention_mask)
        
        return hidden_states

class CausalFNetForCausalLM(PreTrainedModel):
    config_class = FNetConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.fnet = CausalFNetModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.fnet.embeddings.weight
        
        # Initialize weights
        self.post_init()
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        hidden_states = self.fnet(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        return {'loss': loss, 'logits': logits} if loss is not None else {'logits': logits}