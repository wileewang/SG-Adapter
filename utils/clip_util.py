from typing import Any, Optional, Tuple, Union
import torch
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling

def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

def register_attention_clip(model):
    def custom_forward(self):
        def forward(
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple, BaseModelOutputWithPooling]:
            r"""
            Returns:

            """
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if input_ids is None:
                raise ValueError("You have to specify input_ids")

            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])

            hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

            # CLIP's text model uses causal mask, prepare it here.
            # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
            causal_attention_mask = _make_causal_mask(input_shape, hidden_states.dtype, device=hidden_states.device)
            
            if attention_mask is not None:
                if len(attention_mask.shape) > 2:
                    causal_attention_mask = attention_mask.to(dtype=hidden_states.dtype, device=hidden_states.device)
            # expand attention_mask
            # if attention_mask is not None:
            #     # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            #     attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

            encoder_outputs = self.encoder(
                inputs_embeds=hidden_states,
                attention_mask=None,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            last_hidden_state = encoder_outputs[0]
            last_hidden_state = self.final_layer_norm(last_hidden_state)

            if self.eos_token_id == 2:
                # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
                # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
                # ------------------------------------------------------------
                # text_embeds.shape = [batch_size, sequence_length, transformer.width]
                # take features from the eot embedding (eot_token is the highest number in each sequence)
                # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
                pooled_output = last_hidden_state[
                    torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                    input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
                ]
            else:
                # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
                pooled_output = last_hidden_state[
                    torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                    # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                    (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id)
                    .int()
                    .argmax(dim=-1),
                ]

            if not return_dict:
                return (last_hidden_state, pooled_output) + encoder_outputs[1:]

            return BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooled_output,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )
        return forward
    
    model.base_model.text_model.forward = custom_forward(model.base_model.text_model)