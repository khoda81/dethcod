from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import transformers
import transformers.modeling_outputs


class CompressionConfig(transformers.T5Config): ...


@dataclass
class CompressionOutput(transformers.modeling_outputs.Seq2SeqLMOutput):
    value_predictions: Optional[Tuple[torch.FloatTensor, ...]] = None


class CompressionModel(transformers.T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

        self.critic_head = nn.Linear(config.d_model, 1)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], CompressionOutput]:
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if output.decoder_hidden_states is not None:
            last_hidden_state = output.decoder_hidden_states[-1]
            value_predictions = self.critic_head(last_hidden_state)
        else:
            value_predictions = None

        return CompressionOutput(
            value_predictions=value_predictions,
            logits=output.logits,
            past_key_values=output.past_key_values,
            decoder_hidden_states=output.decoder_hidden_states,
            decoder_attentions=output.decoder_attentions,
            cross_attentions=output.cross_attentions,
            encoder_last_hidden_state=output.encoder_last_hidden_state,
            encoder_hidden_states=output.encoder_hidden_states,
            encoder_attentions=output.encoder_attentions,
        )
