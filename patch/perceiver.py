from typing import Tuple

import torch
from transformers.models.perceiver.modeling_perceiver import PerceiverSelfAttention, PerceiverAttention, PerceiverLayer, PerceiverForSequenceClassification, PerceiverForImageClassificationConvProcessing, PerceiverClassifierOutput
from transformers.pytorch_utils import apply_chunking_to_forward

from tome.merge import bipartite_soft_matching, merge_source, merge_wavg
from tome.utils import parse_r
from typing import Optional, Union
import math
import torch.nn as nn

class ToMeLayer(PerceiverLayer):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs: Optional[torch.FloatTensor] = None,
        inputs_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        attention_outputs, metric = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            inputs,
            inputs_mask,
            output_attentions,
            self._tome_info["size"]
        )
        attention_output = attention_outputs[0]

        ### ADDED ###
        r = self._tome_info["r"].pop(0)
        if r > 0:
            # Apply ToMe here
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, attention_output, self._tome_info["source"]
                )
            attention_output, self._tome_info["size"] = merge_wavg(merge, attention_output, self._tome_info["size"])
        ### END ###

        outputs = attention_outputs[1:]  # add attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        layer_output = layer_output + attention_output  # residual connection

        outputs = (layer_output,) + outputs

        return outputs

class ToMeAttention(PerceiverAttention):
    """
    Modifications:
     - Return the mean of k over heads from attention
    """
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs: Optional[torch.FloatTensor] = None,
        inputs_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        size: torch.Tensor = None,
    ) -> Tuple[torch.Tensor]:
        self_outputs, metric = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            inputs,
            inputs_mask,
            output_attentions,
            size,
        )

        # Output projection
        attention_output = self.output(self_outputs[0])

        # Optionally include a residual to the original queries.
        # Consider omitting the residual if the semantics of query and output
        # are different, e.g. if queries are positions and outputs are pixels.
        if self.use_query_residual:
            attention_output = attention_output + hidden_states

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs, metric


class ToMeSelfAttention(PerceiverSelfAttention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs: Optional[torch.FloatTensor] = None,
        inputs_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        size: torch.Tensor = None,
    ) -> Tuple[torch.Tensor]:
        hidden_states = self.layernorm1(hidden_states)
        inputs = self.layernorm2(inputs)

        # Project queries, keys and values to a common feature dimension. If this is instantiated as a cross-attention module,
        # the keys and values come from the inputs; the attention mask needs to be such that the inputs's non-relevant tokens are not attended to.
        is_cross_attention = inputs is not None
        queries = self.query(hidden_states)

        if is_cross_attention:
            keys = self.key(inputs)
            values = self.value(inputs)
            attention_mask = inputs_mask
        else:
            keys = self.key(hidden_states)
            values = self.value(hidden_states)

        # Reshape channels for multi-head attention.
        # We reshape from (batch_size, time, channels) to (batch_size, num_heads, time, channels per head)
        queries = self.transpose_for_scores(queries, self.qk_channels_per_head)
        keys = self.transpose_for_scores(keys, self.qk_channels_per_head)
        values = self.transpose_for_scores(values, self.v_channels_per_head)

        # Take the dot product between the queries and keys to get the raw attention scores.
        attention_scores = torch.matmul(queries, keys.transpose(-1, -2))

        batch_size, num_heads, seq_len, q_head_dim = queries.shape
        _, _, _, v_head_dim = values.shape
        hiddens = self.num_heads * v_head_dim

        attention_scores = attention_scores / math.sqrt(q_head_dim)

        if attention_mask is not None:
            # Apply the attention mask (precomputed for all layers in PerceiverModel forward() function)
            attention_scores = attention_scores + attention_mask
        
        ### ADDED ###
        # Apply proportional attention
        if size is not None:
            attention_scores = attention_scores + size.log()[:, None, None, :, 0]
        ### END ###

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, values)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (hiddens,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs, keys.mean(1)


def make_tome_class(transformer_class):
    class ToMePerceiver(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(
            self,
            inputs: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            labels: Optional[torch.Tensor] = None,
            return_dict: Optional[bool] = None,
            # input_ids: Optional[torch.Tensor] = None,
            pixel_values: Optional[torch.Tensor] = None,
        ) -> Union[Tuple, PerceiverClassifierOutput]:
            self._tome_info["r"] = parse_r(48, self.r)
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            return super().forward(
                inputs=inputs,
                attention_mask=attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                labels=labels,
                return_dict=return_dict,
                # input_ids=input_ids,
                pixel_values=pixel_values,
            )

    return ToMePerceiver


def apply_patch(
    # model: PerceiverForSequenceClassification, trace_source: bool = False, prop_attn: bool = True
    model: PerceiverForImageClassificationConvProcessing, trace_source: bool = False, prop_attn: bool = True
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    ToMePerceiver = make_tome_class(model.__class__)

    model.__class__ = ToMePerceiver
    model.r = 0
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": True,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.perceiver.encoder.self_attends.modules():
        if isinstance(module, PerceiverLayer):
            module.__class__ = ToMeLayer
            module._tome_info = model._tome_info
        elif isinstance(module, PerceiverAttention):
            module.__class__ = ToMeAttention
        elif isinstance(module, PerceiverSelfAttention):
            module.__class__ = ToMeSelfAttention
