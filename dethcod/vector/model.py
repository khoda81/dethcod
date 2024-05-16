import torch.nn as nn
import transformers
import transformers.modeling_outputs


class VectorCompressionConfig(transformers.T5Config):
    def __init__(
        self,
        vocab_size=32128,
        d_model=512,
        compressed_size: tuple[int, int] = (16, 16),
        d_kv=64,
        d_ff=2048,
        num_layers=6,
        num_decoder_layers=None,
        num_heads=8,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="relu",
        is_encoder_decoder=True,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        classifier_dropout=0.0,
        **kwargs,
    ):
        self.compressed_size = compressed_size

        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            d_kv=d_kv,
            d_ff=d_ff,
            num_layers=num_layers,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            relative_attention_num_buckets=relative_attention_num_buckets,
            relative_attention_max_distance=relative_attention_max_distance,
            dropout_rate=dropout_rate,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_factor=initializer_factor,
            feed_forward_proj=feed_forward_proj,
            is_encoder_decoder=is_encoder_decoder,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            classifier_dropout=classifier_dropout,
            **kwargs,
        )


class VectorCompressionModel(transformers.T5ForConditionalGeneration):
    def __init__(self, config: VectorCompressionConfig):
        super().__init__(config)

        self.config = config

        model_dim = self.encoder.config.d_model
        num_tokens, token_embed_dim = self.config.compressed_size
        internal_dim = num_tokens * token_embed_dim
        self.pooling_layer = nn.Linear(model_dim, internal_dim)
        self.unpooling_layer = nn.Linear(token_embed_dim, model_dim)

    def forward(self, input_ids, **kwargs):
        compressed_form = self.compress(input_ids)

        last_hidden_state = self.unpooling_layer.forward(compressed_form)
        encoder_outputs = transformers.modeling_outputs.BaseModelOutput(
            last_hidden_state=last_hidden_state,
        )

        return super().forward(
            encoder_outputs=encoder_outputs,
            **kwargs,
        )

    def compress(self, input_ids):
        encoder_output = self.encoder.forward(input_ids=input_ids)
        hiddens = encoder_output.last_hidden_state
        pooled = self.pooling_layer.forward(hiddens).mean(dim=-2)

        return pooled.unflatten(-1, self.config.compressed_size)
