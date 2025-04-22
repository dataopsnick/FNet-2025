# fnet_model.py
import torch
import torch.nn as nn
import torch.fft
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling, SequenceClassifierOutput
import math

# Configuration Class
class FNetConfig(PretrainedConfig):
    model_type = "fnet" # Important for AutoModel registration

    def __init__(
        self,
        vocab_size=30522, # Example: BERT vocab size
        hidden_size=768,
        num_hidden_layers=12,
        intermediate_size=3072, # Typically 4 * hidden_size
        hidden_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        # FNet specific params (if any needed in future)
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

# Fourier Transform Layer
# Applies 2D FFT (along sequence and hidden dims) and takes the real part
class FourierTransformLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        # Apply 2D FFT along the last two dimensions
        # Keep only the real part as per the paper
        return torch.fft.fftn(x, dim=(-2, -1)).real

# Standard Transformer FeedForward Layer
class FeedForwardLayer(nn.Module):
    def __init__(self, config: FNetConfig):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU() # Common activation
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

# FNet Encoder Block combining Fourier mixing and FeedForward
class FNetEncoderBlock(nn.Module):
    def __init__(self, config: FNetConfig):
        super().__init__()
        self.fourier_layer = FourierTransformLayer()
        self.ff_layer = FeedForwardLayer(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        # Fourier Mixing part
        residual = hidden_states
        mixing_output = self.fourier_layer(hidden_states)
        hidden_states = self.layer_norm1(residual + mixing_output) # Residual + Norm

        # Feed Forward part
        residual = hidden_states
        ff_output = self.ff_layer(hidden_states)
        hidden_states = self.layer_norm2(residual + ff_output) # Residual + Norm
        return hidden_states

# Standard Transformer Embeddings Layer
class FNetEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config: FNetConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # Note: Paper mentions FNet works fine without pos embeddings, but include for BERT comparison
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            # Create position_ids on the fly
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# The main FNet Model (stack of encoders) - Base Model outputting hidden states
class FNetModel(PreTrainedModel):
    config_class = FNetConfig # Link config class

    def __init__(self, config: FNetConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = FNetEmbeddings(config)
        self.encoder_blocks = nn.ModuleList([FNetEncoderBlock(config) for _ in range(config.num_hidden_layers)])
        # Standard weight initialization
        self.init_weights()

    def _init_weights(self, module):
        """Initialize the weights"""
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
        attention_mask=None, # Note: FNet doesn't use mask, but needed for HF interface
        token_type_ids=None,
        position_ids=None,
        head_mask=None, # Note: FNet doesn't use head_mask
        inputs_embeds=None,
        output_attentions=None, # Note: FNet doesn't have attentions
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = embedding_output
        all_hidden_states = () if output_hidden_states else None

        for i, layer_module in enumerate(self.encoder_blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = layer_module(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Simple pooling: Take the [CLS] token representation (first token)
        # This matches common practice for BERT-like sequence classification
        pooled_output = hidden_states[:, 0]

        if not return_dict:
            return (hidden_states, pooled_output) + (all_hidden_states,)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled_output,
            hidden_states=all_hidden_states,
            attentions=None, # FNet has no attentions
        )


# FNet for Sequence Classification (adds a classification head)
class FNetForSequenceClassification(PreTrainedModel):
    config_class = FNetConfig

    def __init__(self, config: FNetConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.fnet = FNetModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.init_weights()

    def _init_weights(self, module):
        """ Initialize the weights """
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
        attention_mask=None, # Ignored by FNetModel core but might be used by trainer/padding
        token_type_ids=None,
        position_ids=None,
        head_mask=None,      # Ignored
        inputs_embeds=None,
        labels=None,
        output_attentions=None, # Ignored
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.fnet(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1] # Use pooled output (CLS token)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions, # Will be None
        )
