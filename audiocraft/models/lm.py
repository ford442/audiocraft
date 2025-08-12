# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import partial
import logging
import math
import typing as tp

import torch
from torch import nn

from ..utils import utils
from ..modules.streaming import StreamingModule, State
from ..modules.transformer import StreamingTransformer, create_norm_fn
from ..modules.conditioners import (
    ConditionFuser,
    ClassifierFreeGuidanceDropout,
    AttributeDropout,
    ConditioningProvider,
    ConditioningAttributes,
    ConditionType,
    _drop_description_condition
)
from ..modules.codebooks_patterns import CodebooksPatternProvider
from ..modules.activations import get_activation_fn


logger = logging.getLogger(__name__)
ConditionTensors = tp.Dict[str, ConditionType]
CFGConditions = tp.Union[ConditionTensors, tp.Tuple[ConditionTensors, ConditionTensors]]


def get_init_fn(method: str, input_dim: int, init_depth: tp.Optional[int] = None):
    """LM layer initialization.
    Inspired from xlformers: https://github.com/fairinternal/xlformers

    Args:
        method (str): Method name for init function. Valid options are:
            'gaussian', 'uniform'.
        input_dim (int): Input dimension of the initialized module.
        init_depth (int, optional): Optional init depth value used to rescale
            the standard deviation if defined.
    """
    # Compute std
    std = 1 / math.sqrt(input_dim)
    # Rescale with depth
    if init_depth is not None:
        std = std / math.sqrt(2 * init_depth)

    if method == 'gaussian':
        return partial(
            torch.nn.init.trunc_normal_, mean=0.0, std=std, a=-3 * std, b=3 * std
        )
    elif method == 'uniform':
        bound = math.sqrt(3) * std  # ensure the standard deviation is `std`
        return partial(torch.nn.init.uniform_, a=-bound, b=bound)
    else:
        raise ValueError("Unsupported layer initialization method")


def init_layer(m: nn.Module,
               method: str,
               init_depth: tp.Optional[int] = None,
               zero_bias_init: bool = False):
    """Wrapper around ``get_init_fn`` for proper initialization of LM modules.

    Args:
        m (nn.Module): Module to initialize.
        method (str): Method name for the init function.
        init_depth (int, optional): Optional init depth value used to rescale
            the standard deviation if defined.
        zero_bias_init (bool): Whether to initialize the bias to 0 or not.
    """
    if isinstance(m, nn.Linear):
        init_fn = get_init_fn(method, m.in_features, init_depth=init_depth)
        if m.weight.device.type == 'cpu' and m.weight.dtype == torch.float16:
            weight = m.weight.float()
            init_fn(weight)
            m.weight.data[:] = weight.half()
        else:
            init_fn(m.weight)
        if zero_bias_init and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        init_fn = get_init_fn(method, m.embedding_dim, init_depth=None)
        if m.weight.device.type == 'cpu' and m.weight.dtype == torch.float16:
            weight = m.weight.float()
            init_fn(weight)
            m.weight.data[:] = weight.half()
        else:
            init_fn(m.weight)


class ScaledEmbedding(nn.Embedding):
    """Boost learning rate for embeddings (with `scale`).
    """
    def __init__(self, *args, lr=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr

    def make_optim_group(self):
        group = {"params": list(self.parameters())}
        if self.lr is not None:
            group["lr"] = self.lr
        return group


@dataclass
class LMOutput:
    # The logits are already re-aligned with the input codes
    # hence no extra shift is required, e.g. when computing CE
    logits: torch.Tensor  # [B, K, T, card]
    mask: torch.Tensor  # [B, K, T]


class LMModel(StreamingModule):
    """Transformer-based language model on multiple streams of codes.

    Args:
        pattern_provider (CodebooksPatternProvider): Pattern provider for codebook interleaving.
        condition_provider (MusicConditioningProvider): Conditioning provider from metadata.
        fuser (ConditionFuser): Fuser handling the fusing of conditions with language model input.
        n_q (int): Number of parallel streams to model.
        card (int): Cardinality, vocabulary size.
        dim (int): Dimension of the transformer encoder.
        num_heads (int): Number of heads for the transformer encoder.
        hidden_scale (int): Scale for hidden feed forward dimension of the transformer encoder.
        norm (str): Normalization method.
        norm_first (bool): Use pre-norm instead of post-norm.
        emb_lr (float, optional): Embedding-specific learning rate.
        bias_proj (bool): Use bias for output projections.
        weight_init (str, optional): Method for weight initialization.
        depthwise_init (str, optional): Method for depthwise weight initialization.
        zero_bias_init (bool): If true and bias in Linears, initialize bias to zeros.
        cfg_dropout (float): Classifier-free guidance dropout.
        cfg_coef (float): Classifier-free guidance coefficient.
        attribute_dropout (dict): Attribute dropout probabilities.
        two_step_cfg (bool): Whether to run classifier free-guidance with 2 distinct steps.
        **kwargs: Additional parameters for the transformer encoder.
    """
    def __init__(self, pattern_provider: CodebooksPatternProvider, condition_provider: ConditioningProvider,
                 fuser: ConditionFuser, n_q: int = 8, card: int = 1024, dim: int = 128, num_heads: int = 8,
                 hidden_scale: int = 4, norm: str = 'layer_norm', norm_first: bool = False,
                 emb_lr: tp.Optional[float] = None, bias_proj: bool = True,
                 weight_init: tp.Optional[str] = None, depthwise_init: tp.Optional[str] = None,
                 zero_bias_init: bool = False, cfg_dropout: float = 0, cfg_coef: float = 1.0,
                 attribute_dropout: tp.Dict[str, tp.Dict[str, float]] = {}, two_step_cfg: bool = False,
                 **kwargs):
        super().__init__()
        self.cfg_coef = cfg_coef
        self.cfg_dropout = ClassifierFreeGuidanceDropout(p=cfg_dropout)
        self.att_dropout = AttributeDropout(p=attribute_dropout)
        self.condition_provider = condition_provider
        self.fuser = fuser
        self.card = card
        embed_dim = self.card + 1
        self.n_q = n_q
        self.dim = dim
        self.pattern_provider = pattern_provider
        self.two_step_cfg = two_step_cfg
        self.emb = nn.ModuleList([ScaledEmbedding(embed_dim, dim, lr=emb_lr) for _ in range(n_q)])
        if 'activation' in kwargs:
            kwargs['activation'] = get_activation_fn(kwargs['activation'])
        self.transformer = StreamingTransformer(
            d_model=dim, num_heads=num_heads, dim_feedforward=int(hidden_scale * dim),
            norm=norm, norm_first=norm_first, **kwargs)
        self.out_norm: tp.Optional[nn.Module] = None
        if norm_first:
            self.out_norm = create_norm_fn(norm, dim)
        self.linears = nn.ModuleList([nn.Linear(dim, self.card, bias=bias_proj) for _ in range(n_q)])
        self._init_weights(weight_init, depthwise_init, zero_bias_init)
        self._fsdp: tp.Optional[nn.Module]
        self.__dict__['_fsdp'] = None

    def _init_weights(self, weight_init: tp.Optional[str], depthwise_init: tp.Optional[str], zero_bias_init: bool):
        """Initialization of the transformer module weights.

        Args:
            weight_init (str, optional): Weight initialization strategy. See ``get_init_fn`` for valid options.
            depthwise_init (str, optional): Depthwise initialization strategy. The following options are valid:
                'current' where the depth corresponds to the current layer index or 'global' where the total number
                of layer is used as depth. If not set, no depthwise initialization strategy is used.
            zero_bias_init (bool): Whether to initialize bias to zero or not.
        """
        assert depthwise_init is None or depthwise_init in ['current', 'global']
        assert depthwise_init is None or weight_init is not None, \
            "If 'depthwise_init' is defined, a 'weight_init' method should be provided."
        assert not zero_bias_init or weight_init is not None, \
            "If 'zero_bias_init', a 'weight_init' method should be provided"

        if weight_init is None:
            return

        for emb_layer in self.emb:
            init_layer(emb_layer, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)

        for layer_idx, tr_layer in enumerate(self.transformer.layers):
            depth = None
            if depthwise_init == 'current':
                depth = layer_idx + 1
            elif depthwise_init == 'global':
                depth = len(self.transformer.layers)
            init_fn = partial(init_layer, method=weight_init, init_depth=depth, zero_bias_init=zero_bias_init)
            tr_layer.apply(init_fn)

        for linear in self.linears:
            init_layer(linear, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)

    @property
    def special_token_id(self) -> int:
        return self.card

    @property
    def num_codebooks(self) -> int:
        return self.n_q

    def forward(self, sequence: torch.Tensor,
                conditions: tp.List[ConditioningAttributes],
                condition_tensors: tp.Optional[ConditionTensors] = None,
                stage: int = -1) -> torch.Tensor:
        """Apply language model on sequence and conditions.
        Given a tensor of sequence of shape [B, K, S] with K the number of codebooks and
        S the sequence steps, return the logits with shape [B, card, K, S].

        Args:
            indices (torch.Tensor): Indices of the codes to model.
            conditions (list of ConditioningAttributes): Conditions to use when modeling
                the given codes. Note that when evaluating multiple time with the same conditioning
                you should pre-compute those and pass them as `condition_tensors`.
            condition_tensors (dict[str, ConditionType], optional): Pre-computed conditioning
                tensors, see `conditions`.
            stage (int): The codebook level that is being predicted. Relevant for MAGNeT
                in which prediction is done in a codebook-by-codebook manner.
                Takes values in range(n_q), and ignored by default.
        Returns:
            torch.Tensor: Logits.
        """
        B, K, S = sequence.shape
        assert K == self.num_codebooks, "Sequence shape must match the specified number of codebooks"
        input_ = sum([self.emb[k](sequence[:, k]) for k in range(K)])
        if condition_tensors is None:
            assert not self._is_streaming, "Conditions tensors should be precomputed when streaming."
            # apply dropout modules
            conditions = self.cfg_dropout(conditions)
            conditions = self.att_dropout(conditions)
            tokenized = self.condition_provider.tokenize(conditions)
            # encode conditions and fuse, both have a streaming cache to not recompute when generating.
            condition_tensors = self.condition_provider(tokenized)
        else:
            assert not conditions, "Shouldn't pass both conditions and condition_tensors."

        input_, cross_attention_input = self.fuser(input_, condition_tensors)

        out = self.transformer(input_, cross_attention_src=cross_attention_input,
                               src_mask=(self.attn_mask_per_stage[stage] if stage >= 0 else None))  # type: ignore
        if self.out_norm:
            out = self.out_norm(out)
        logits = torch.stack([self.linears[k](out) for k in range(K)], dim=1)  # [B, K, S, card]

        # remove the prefix from the model outputs
        if len(self.fuser.fuse2cond['prepend']) > 0:
            logits = logits[:, :, -S:]

        return logits  # [B, K, S, card]

    def compute_predictions(
            self, codes: torch.Tensor,
            conditions: tp.List[ConditioningAttributes],
            condition_tensors: tp.Optional[ConditionTensors] = None,
            stage: int = -1,
            keep_only_valid_steps: bool = True) -> LMOutput:
        """Given an input tensor of codes [B, K, T] and list of conditions, runs the model
        forward using the specified codes interleaving pattern.

        Args:
            codes (torch.Tensor): Input codes of shape [B, K, T] with B the batch size,
                K the number of codebooks and T the number of timesteps.
            conditions (list of ConditioningAttributes): conditionings to use when modeling
                the given codes. Note that when evaluating multiple time with the same conditioning
                you should pre-compute those and pass them as `condition_tensors`.
            condition_tensors (dict[str, ConditionType], optional): pre-computed conditioning
                tensors, see `conditions`.
            stage (int): The codebook level that is being predicted. Relevant for MAGNeT
                in which prediction is done in a codebook-by-codebook manner.
                Takes values in range(n_q), and ignored by default.
            keep_only_valid_steps (bool): Build a sequence from the pattern up to valid (= fully defined) steps.
                Steps that are beyond valid steps will be replaced by the special_token in that case.
        Returns:
            LMOutput: Language model outputs
                logits (torch.Tensor) of shape [B, K, T, card] corresponding to the provided codes,
                    i.e. the first item corresponds to logits to predict the first code, meaning that
                    no additional shifting of codes and logits is required.
                mask (torch.Tensor) of shape [B, K, T], mask over valid and invalid positions.
                    Given the specified interleaving strategies, parts of the logits and codes should
                    not be considered as valid predictions because of invalid context.
        """
        B, K, T = codes.shape
        codes = codes.contiguous()
        # map codes [B, K, T] into pattern sequence [B, K, S] using special_token_id for masked tokens
        pattern = self.pattern_provider.get_pattern(T)
        sequence_codes, sequence_indexes, sequence_mask = pattern.build_pattern_sequence(
            codes, self.special_token_id, keep_only_valid_steps=keep_only_valid_steps,
        )

        # apply model on pattern sequence
        model = self if self._fsdp is None else self._fsdp
        logits = model(sequence_codes, conditions, condition_tensors, stage=stage)  # [B, K, S, card]
        # map back the logits on pattern sequence to logits on original codes: [B, K, S, card] -> [B, K, T, card]
        # and provide the corresponding mask over invalid positions of tokens
        logits = logits.permute(0, 3, 1, 2)  # [B, card, K, S]
        # note: we use nans as special token to make it obvious if we feed unexpected logits
        logits, logits_indexes, logits_mask = pattern.revert_pattern_logits(
            logits, float('nan'), keep_only_valid_steps=keep_only_valid_steps
        )
        logits = logits.permute(0, 2, 3, 1)  # [B, K, T, card]
        logits_mask = logits_mask[None, :, :].expand(B, -1, -1)  # [K, T] -> [B, K, T]
        return LMOutput(logits, logits_mask)

    def _sample_next_token(self,
                           sequence: torch.Tensor,
                           cfg_conditions: CFGConditions,
                           unconditional_state: State,
                           use_sampling: bool = False,
                           temp: float = 1.0,
                           top_k: int = 0,
                           top_p: float = 0.0,
                           cfg_coef: tp.Optional[float] = None,
                           cfg_coef_beta: tp.Optional[float] = None,
                           two_step_cfg: tp.Optional[bool] = None) -> torch.Tensor:
        """Sample next token from the model given a sequence and a set of conditions. The model supports
        multiple sampling strategies (greedy sampling, softmax, top-k, top-p...).

        Args:
            sequence (torch.Tensor): Current sequence of shape [B, K, S]
                with K corresponding to the number of codebooks and S the number of sequence steps.
                S = 1 in streaming mode, except for the first step that contains a bigger prompt.
            condition_tensors (dict[str, ConditionType): Set of conditions. If CFG is used,
                should be twice the batch size, being the concatenation of the conditions + null conditions.
            use_sampling (bool): Whether to use a sampling strategy or not.
            temp (float): Sampling temperature.
            top_k (int): K for "top-k" sampling.
            top_p (float): P for "top-p" sampling.
            cfg_coef (float, optional): classifier free guidance coefficient
            cfg_coef_beta (float, optional): If None, simple classifier free guidance is used with cfg_coef.
                If not None, we apply double classifier free guidance as introduced in MusicGen-Style
                in paragraph 4.3 (https://arxiv.org/pdf/2407.12563). This beta coefficient is meant to
                push the text condition more than the style condition in the case where both text and style
                conditions are being used.
            two_step_cfg (bool): Whether to run classifier free-guidance with 2 distinct steps.

        Returns:
            next_token (torch.Tensor): Next token tensor of shape [B, K, 1].
        """
        B = sequence.shape[0]
        cfg_coef = self.cfg_coef if cfg_coef is None else cfg_coef
        model = self if self._fsdp is None else self._fsdp
        two_step_cfg = self.two_step_cfg if two_step_cfg is None else two_step_cfg
        if cfg_coef_beta is not None:
            assert isinstance(cfg_conditions, dict)
            condition_tensors = cfg_conditions
            if condition_tensors:
                # Preparing for CFG, predicting conditional text and style, conditional style
                # and unconditional
                sequence = torch.cat([sequence, sequence, sequence], dim=0)
            all_logits = model(
                sequence,
                conditions=[], condition_tensors=condition_tensors)
            if condition_tensors:
                cond_logits, wav_logits, uncond_logits = all_logits.split(B, dim=0)  # [B, K, T, card]
                logits = uncond_logits + cfg_coef * (
                    wav_logits + cfg_coef_beta * (cond_logits - wav_logits) - uncond_logits
                    )

        elif two_step_cfg and cfg_conditions != {}:
            assert isinstance(cfg_conditions, tuple), type(cfg_conditions)
            condition_tensors, null_condition_tensors = cfg_conditions
            cond_logits = model(sequence, conditions=[], condition_tensors=condition_tensors)
            state = self.get_streaming_state()
            self.set_streaming_state(unconditional_state)
            uncond_logits = model(sequence, conditions=[], condition_tensors=null_condition_tensors)
            unconditional_state.update(self.get_streaming_state())
            self.set_streaming_state(state)
            logits = uncond_logits + (cond_logits - uncond_logits) * self.cfg_coef
        else:
            assert isinstance(cfg_conditions, dict)
            condition_tensors = cfg_conditions
            if condition_tensors:
                # Preparing for CFG, predicting both conditional and unconditional logits.
                sequence = torch.cat([sequence, sequence], dim=0)
            all_logits = model(
                sequence,
                conditions=[], condition_tensors=condition_tensors)
            if condition_tensors:
                cond_logits, uncond_logits = all_logits.split(B, dim=0)  # [B, K, T, card]
                logits = uncond_logits + (cond_logits - uncond_logits) * cfg_coef
            else:
                logits = all_logits

        logits = logits.permute(0, 1, 3, 2)  # [B, K, card, T]
        logits = logits[..., -1]  # [B x K x card]

        # Apply softmax for sampling if temp > 0. Else, do greedy sampling to avoid zero division error.
        if use_sampling and temp > 0.0:
            probs = torch.softmax(logits / temp, dim=-1)
            if top_p > 0.0:
                next_token = utils.sample_top_p(probs, p=top_p)
            elif top_k > 0:
                next_token = utils.sample_top_k(probs, k=top_k)
            else:
                next_token = utils.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        return next_token

    @torch.no_grad()
    def generate_in_chunks(self,
                           prompt: tp.Optional[torch.Tensor] = None,
                           conditions: tp.List[ConditioningAttributes] = [],
                           num_samples: tp.Optional[int] = None,
                           chunk_len: int = 1024,
                           overlap_len: int = 128,
                           **kwargs
                           ) -> torch.Tensor:
        """
        Generates tokens in chunks to manage memory usage for long sequences.
        This is a wrapper around the `generate` method that avoids pre-allocating
        a tensor for the full `max_gen_len`, preventing CUDA out-of-memory errors.
        This version explicitly manages keyword arguments for robust operation.
        """
        assert not self.training, "generation shouldn't be used in training mode."
        assert overlap_len < chunk_len, "Overlap length must be smaller than chunk length."

        # Get max_gen_len from kwargs, with a default value. This is safer.
        max_gen_len = kwargs.pop('max_gen_len', 256)

        # Define the set of valid arguments for the underlying `generate` method
        # to avoid passing unexpected arguments like 'progress'.
        valid_generate_args = {
            'use_sampling', 'temp', 'top_k', 'top_p', 'cfg_coef', 'cfg_coef_beta',
            'two_step_cfg', 'remove_prompts', 'check', 'callback'
        }
        # Filter kwargs to only include valid arguments for the next call.
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_generate_args}

        first_param = next(iter(self.parameters()))
        device = first_param.device

        if num_samples is None:
            if prompt is not None:
                num_samples = prompt.shape[0]
            elif conditions:
                num_samples = len(conditions)
            else:
                num_samples = 1

        if prompt is None:
            assert num_samples > 0
            prompt = torch.zeros((num_samples, self.num_codebooks, 0), dtype=torch.long, device=device)

        B, K, T = prompt.shape
        all_generated_codes = []

        if T > 0:
            prompt_len = min(T, max_gen_len)
            all_generated_codes.append(prompt[..., :prompt_len])

        total_generated_len = sum(c.shape[-1] for c in all_generated_codes)
        current_prompt = prompt

        with self.streaming():
            while total_generated_len < max_gen_len:
                remaining_len = max_gen_len - total_generated_len
                len_to_generate = min(chunk_len, remaining_len)

                effective_max_gen_len = current_prompt.shape[-1] + len_to_generate

                # Call the original generate function with the filtered kwargs
                generated_chunk = self.generate(
                    prompt=current_prompt,
                    conditions=conditions,
                    num_samples=num_samples,
                    max_gen_len=effective_max_gen_len,
                    remove_prompts=True,
                    **filtered_kwargs
                )

                all_generated_codes.append(generated_chunk)
                total_generated_len += generated_chunk.shape[-1]

                print(f"Generated chunk of size {generated_chunk.shape[-1]}. Total generated: {total_generated_len}/{max_gen_len}")

                current_prompt = generated_chunk[..., -overlap_len:]
                # The incorrect line 'conditions = []' has been removed from here.

        if not all_generated_codes:
            return torch.zeros((B, K, 0), dtype=torch.long, device=device)

        final_codes = torch.cat(all_generated_codes, dim=-1)
        return final_codes[..., :max_gen_len]
