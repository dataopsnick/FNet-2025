import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple FNet implementation with no HuggingFace dependencies
class CustomFNetConfig:
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        num_labels=2,
        problem_type=None
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.num_labels = num_labels
        self.problem_type = problem_type

# FourierTransformLayer for token mixing
class CustomFourierTransformLayer(nn.Module):
    def forward(self, x):
        # Apply 2D FFT and take real part
        return torch.fft.fftn(x, dim=(-2, -1)).real

# Feed-forward layer
class CustomFeedForwardLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation = nn.GELU()
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        x = self.dropout(x)
        return x

# Embeddings layer
class CustomEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Position ids buffer
        self.register_buffer(
            "position_ids", 
            torch.arange(config.max_position_embeddings).expand((1, -1))
        )

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
            seq_length = input_ids.size(1)
        else:
            input_shape = inputs_embeds.size()[:-1]
            seq_length = input_shape[1]
            
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
            
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
            
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
            
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

# FNet Encoder Block
class CustomFNetEncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fourier = CustomFourierTransformLayer()
        self.ffn = CustomFeedForwardLayer(config)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, x):
        # Fourier transform with residual connection
        residual = x
        x = self.fourier(x)
        x = self.norm1(x + residual)
        
        # Feed-forward with residual connection
        residual = x
        x = self.ffn(x)
        x = self.norm2(x + residual)
        
        return x

# Complete FNet Model for Sequence Classification
class CustomFNetForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        
        # Embeddings
        self.embeddings = CustomEmbeddings(config)
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList(
            [CustomFNetEncoderBlock(config) for _ in range(config.num_hidden_layers)]
        )
        
        # Pooler
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()
        
        # Classification head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def forward(
        self, 
        input_ids=None, 
        attention_mask=None,  # Not used in FNet but included for compatibility
        token_type_ids=None, 
        position_ids=None, 
        inputs_embeds=None,
        labels=None,
        output_hidden_states=None,
        return_dict=None
    ):
        # Process embeddings
        embedding_output = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds
        )
        
        # Process through encoder blocks
        hidden_states = embedding_output
        all_hidden_states = () if output_hidden_states else None
        
        for block in self.encoder_blocks:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = block(hidden_states)
            
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        # Pool [CLS] token 
        first_token = hidden_states[:, 0]
        pooled_output = self.pooler(first_token)
        pooled_output = self.pooler_activation(pooled_output)
        
        # Classification head
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and labels.dtype in [torch.long, torch.int]:
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
                    
            if self.config.problem_type == "regression":
                loss_fn = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fn(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fn(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fn = nn.BCEWithLogitsLoss()
                loss = loss_fn(logits, labels)
        
        # Build output based on return_dict flag
        if return_dict:
            return {
                "loss": loss,
                "logits": logits,
                "hidden_states": all_hidden_states,
                "pooler_output": pooled_output
            }
        else:
            output = (logits,)
            if output_hidden_states:
                output = output + (all_hidden_states,)
            return ((loss,) + output) if loss is not None else output
