import inspect
from typing import Any, Literal

import torch
from torch import nn
from transformers import GemmaForCausalLM
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma


class FTP1PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        tactile_expert_config=None,
        *,
        use_tactile_input: bool = False,
        use_adarms=None,
        use_state_adarms_cond=False,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        self.vlm_config = vlm_config
        self.action_expert_config = action_expert_config
        self.tactile_expert_config = tactile_expert_config
        self.use_tactile_input = use_tactile_input
        self.use_adarms = use_adarms
        self.precision = precision

        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()
        self.use_tactile_input = use_tactile_input
        if self.use_tactile_input:
            assert tactile_expert_config is not None, (
                "tactile_expert_config must be provided when use_tactile_input is True"
            )

        # reference: https://hugging-face.cn/docs/transformers/model_doc/paligemma
        # reference: https://hugging-face.cn/docs/transformers/model_doc/gemma

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        # text_config is set as GemmaConfig by default
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        if use_adarms[1]:
            action_adarms_cond_dim = action_expert_config.width
            if use_state_adarms_cond:
                action_adarms_cond_dim = 2 * action_expert_config.width
        else:
            action_adarms_cond_dim = None

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=1,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_adarms_cond_dim,
        )

        # reference: https://hugging-face.cn/docs/transformers/model_doc/paligemma#transformers.PaliGemmaForConditionalGeneration
        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        # Conditionally create tactile expert
        if use_tactile_input and tactile_expert_config is not None:
            tactile_expert_config_hf = CONFIG_MAPPING["gemma"](
                head_dim=tactile_expert_config.head_dim,
                hidden_size=tactile_expert_config.width,
                intermediate_size=tactile_expert_config.mlp_dim,
                num_attention_heads=tactile_expert_config.num_heads,
                num_hidden_layers=tactile_expert_config.depth,
                num_key_value_heads=tactile_expert_config.num_kv_heads,
                vocab_size=1,
                hidden_activation="gelu_pytorch_tanh",
                torch_dtype="float32",
                use_adarms=False,  # Tactile expert doesn't use adarms
                adarms_cond_dim=None,
            )
            self.gemma_tactile_expert = GemmaForCausalLM(config=tactile_expert_config_hf)
            self.gemma_tactile_expert.model.embed_tokens = None
        else:
            self.gemma_tactile_expert = None

        (
            self._norm_forward_param_names,
            self._norm_forward_supports_var_kwargs,
        ) = self._infer_norm_forward_signature()

        self.to_bfloat16_for_selected_params(precision)

    def _infer_norm_forward_signature(self) -> tuple[set[str], bool]:
        """Probe GemmaRMSNorm.forward once for kwargs compatibility across transformers_replace variants."""
        sample_norm = self.gemma_expert.model.layers[0].input_layernorm
        forward_fn = getattr(sample_norm, "forward", None)
        if forward_fn is None:
            return set(), False
        try:
            sig = inspect.signature(forward_fn)
        except (TypeError, ValueError):
            return set(), False
        supports_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        return set(sig.parameters.keys()), supports_var_kwargs

    def _call_adarms_norm(
        self,
        norm_layer: nn.Module,
        hidden_states: torch.Tensor,
        cond: torch.Tensor | None,
        active_tail_tokens: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kwargs = {}
        if self._norm_forward_supports_var_kwargs or "cond" in self._norm_forward_param_names:
            kwargs["cond"] = cond
        if self._norm_forward_supports_var_kwargs or "active_tail_tokens" in self._norm_forward_param_names:
            kwargs["active_tail_tokens"] = active_tail_tokens
        return norm_layer(hidden_states, **kwargs)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def _apply_rotary_to_query_and_key(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dummy_tensor = torch.zeros(
            query_states.shape[0],
            query_states.shape[2],
            query_states.shape[-1],
            device=query_states.device,
            dtype=query_states.dtype,
        )
        cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
        return modeling_gemma.apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            unsqueeze_dim=1,
        )

    def _compute_attention_output(
        self,
        layer,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        att_output, _ = modeling_gemma.eager_attention_forward(
            layer.self_attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            layer.self_attn.scaling,
        )
        return att_output.reshape(att_output.shape[0], -1, att_output.shape[2] * att_output.shape[3])

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | Any | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
        adarms_active_tail_tokens: list[int | None] | None = None,
    ):
        # Handle different numbers of experts based on use_tactile_input
        # Expert order: [vlm, tactile, action] (indices: 0, 1, 2)
        if self.use_tactile_input and self.gemma_tactile_expert is not None:
            # Three experts: [vlm, tactile, action]
            if adarms_cond is None:
                adarms_cond = [None, None, None]
            if adarms_active_tail_tokens is None:
                adarms_active_tail_tokens = [None, None, None]
            assert len(inputs_embeds) == 3, "Inputs embeds must be a list of 3 tensors when use_tactile_input is True"
            assert len(adarms_active_tail_tokens) == 3, (
                "adarms_active_tail_tokens must have 3 entries when tactile expert is enabled"
            )

            if inputs_embeds[1] is None and inputs_embeds[2] is None:
                # Only vlm (index 0) is provided
                vlm_output = self.paligemma.language_model.forward(
                    inputs_embeds=inputs_embeds[0],
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
                    adarms_active_tail_tokens=(
                        adarms_active_tail_tokens[0] if adarms_active_tail_tokens is not None else None
                    ),
                )
                vlm_past_key_values = vlm_output.past_key_values
                vlm_output = vlm_output.last_hidden_state
                tactile_output = None
                action_output = None
                # Return order: [vlm, tactile, action]
                return [vlm_output, tactile_output, action_output], vlm_past_key_values
            if inputs_embeds[0] is None and inputs_embeds[2] is None:
                # Only tactile (index 1) is provided
                tactile_output = self.gemma_tactile_expert.model.forward(
                    inputs_embeds=inputs_embeds[1],
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
                    adarms_active_tail_tokens=(
                        adarms_active_tail_tokens[1] if adarms_active_tail_tokens is not None else None
                    ),
                )
                tactile_past_key_values = tactile_output.past_key_values
                tactile_output = tactile_output.last_hidden_state
                vlm_output = None
                action_output = None
                # Return order: [vlm, tactile, action]
                return [vlm_output, tactile_output, action_output], tactile_past_key_values
            if inputs_embeds[0] is None and inputs_embeds[1] is None:
                # Only action (index 2) is provided
                action_output = self.gemma_expert.model.forward(
                    inputs_embeds=inputs_embeds[2],
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    adarms_cond=adarms_cond[2] if adarms_cond is not None else None,
                    adarms_active_tail_tokens=(
                        adarms_active_tail_tokens[2] if adarms_active_tail_tokens is not None else None
                    ),
                )
                action_output = action_output.last_hidden_state
                vlm_output = None
                tactile_output = None
                past_key_values = None
                # Return order: [vlm, tactile, action]
                return [vlm_output, tactile_output, action_output], past_key_values
            # All three experts are provided - do cross-attention
            models = [self.paligemma.language_model, self.gemma_tactile_expert.model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers
        else:
            # Two experts: [vlm, action] (indices: 0, 1)
            if adarms_cond is None:
                adarms_cond = [None, None]
            if adarms_active_tail_tokens is None:
                adarms_active_tail_tokens = [None, None]
            assert len(inputs_embeds) == 2, "Inputs embeds must be a list of 2 tensors when use_tactile_input is False"
            assert len(adarms_active_tail_tokens) == 2, (
                "adarms_active_tail_tokens must have 2 entries when tactile expert is disabled"
            )

            if inputs_embeds[1] is None:
                # Only vlm (index 0) is provided
                vlm_output = self.paligemma.language_model.forward(
                    inputs_embeds=inputs_embeds[0],
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
                    adarms_active_tail_tokens=(
                        adarms_active_tail_tokens[0] if adarms_active_tail_tokens is not None else None
                    ),
                )
                vlm_past_key_values = vlm_output.past_key_values
                vlm_output = vlm_output.last_hidden_state
                action_output = None
                # Return order: [vlm, action]
                return [vlm_output, action_output], vlm_past_key_values
            if inputs_embeds[0] is None:
                # Only action (index 1) is provided
                action_output = self.gemma_expert.model.forward(
                    inputs_embeds=inputs_embeds[1],
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
                    adarms_active_tail_tokens=(
                        adarms_active_tail_tokens[1] if adarms_active_tail_tokens is not None else None
                    ),
                )
                action_output = action_output.last_hidden_state
                vlm_output = None
                past_key_values = None
                # Return order: [vlm, action]
                return [vlm_output, action_output], past_key_values
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

        # Process all experts together if all are provided
        # Expert order: [vlm, tactile, action] for three experts, [vlm, action] for two experts
        if self.use_tactile_input and self.gemma_tactile_expert is not None:
            all_provided = (
                inputs_embeds[0] is not None and inputs_embeds[1] is not None and inputs_embeds[2] is not None
            )
        else:
            all_provided = inputs_embeds[0] is not None and inputs_embeds[1] is not None

        if all_provided:
            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Force enable gradient checkpointing if we're in training mode and the model supports it
            if self.training and hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                if not self.gemma_expert.model.gradient_checkpointing:
                    print("Forcing gradient checkpointing to be enabled for Gemma expert model")
                    self.gemma_expert.model.gradient_checkpointing = True
                use_gradient_checkpointing = True

            # Debug gradient checkpointing status
            if hasattr(self, "_debug_gc_printed") and not self._debug_gc_printed:
                print(f"Gemma expert model gradient checkpointing: {use_gradient_checkpointing}")
                print(f"Model training mode: {self.training}")
                print(
                    f"Gemma expert model has gradient_checkpointing attr: {hasattr(self.gemma_expert.model, 'gradient_checkpointing')}"
                )
                if hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                    print(
                        f"Gemma expert model gradient_checkpointing value: {self.gemma_expert.model.gradient_checkpointing}"
                    )
                self._debug_gc_printed = True

            # Define the complete layer computation function for gradient checkpointing
            def compute_layer_complete(
                layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond, adarms_active_tail_tokens
            ):
                if self.use_tactile_input and self.gemma_tactile_expert is not None:
                    models = [self.paligemma.language_model, self.gemma_tactile_expert.model, self.gemma_expert.model]
                else:
                    models = [self.paligemma.language_model, self.gemma_expert.model]

                gates = []
                branch_states = []
                current_pos = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    if hidden_states is None:
                        continue
                    layer = models[i].layers[layer_idx]
                    normalized_hidden_states, gate = self._call_adarms_norm(
                        layer.input_layernorm,
                        hidden_states,
                        adarms_cond[i],
                        adarms_active_tail_tokens[i],
                    )
                    gates.append(gate)

                    input_shape = normalized_hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    query_state = layer.self_attn.q_proj(normalized_hidden_states).view(hidden_shape).transpose(1, 2)
                    key_state = layer.self_attn.k_proj(normalized_hidden_states).view(hidden_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(normalized_hidden_states).view(hidden_shape).transpose(1, 2)
                    branch_position_ids = position_ids[:, current_pos : current_pos + normalized_hidden_states.shape[1]]
                    query_state, key_state = self._apply_rotary_to_query_and_key(
                        query_state,
                        key_state,
                        branch_position_ids,
                    )
                    branch_self_attention_mask = (
                        None
                        if attention_mask is None
                        else attention_mask[
                            :,
                            :,
                            current_pos : current_pos + normalized_hidden_states.shape[1],
                            current_pos : current_pos + normalized_hidden_states.shape[1],
                        ]
                    )
                    branch_states.append(
                        {
                            "input_index": i,
                            "layer": layer,
                            "query_states": query_state,
                            "key_states": key_state,
                            "value_states": value_state,
                            "start_pos": current_pos,
                            "end_pos": current_pos + normalized_hidden_states.shape[1],
                            "self_attention_mask": branch_self_attention_mask,
                        }
                    )
                    current_pos += normalized_hidden_states.shape[1]

                query_states = torch.cat([branch_info["query_states"] for branch_info in branch_states], dim=2)
                key_states = torch.cat([branch_info["key_states"] for branch_info in branch_states], dim=2)
                value_states = torch.cat([branch_info["value_states"] for branch_info in branch_states], dim=2)
                att_output = self._compute_attention_output(
                    branch_states[0]["layer"],
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                )

                # Process layer outputs
                outputs_embeds = [None] * len(inputs_embeds)  # Initialize with None
                start_pos = 0
                gate_idx = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    if hidden_states is None:
                        continue
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    att_output_slice = att_output[:, start_pos:end_pos]
                    if att_output_slice.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output_slice = att_output_slice.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output_slice)

                    # first residual
                    out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[gate_idx])  # noqa: SLF001
                    after_first_residual = out_emb.clone()
                    out_emb, gate = self._call_adarms_norm(
                        layer.post_attention_layernorm,
                        out_emb,
                        adarms_cond[i],
                        adarms_active_tail_tokens[i],
                    )
                    # Convert to bfloat16 if the next layer (mlp) uses bfloat16
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)

                    out_emb = layer.mlp(out_emb)
                    # second residual
                    out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
                    outputs_embeds[i] = out_emb
                    start_pos = end_pos
                    gate_idx += 1

                return outputs_embeds

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        adarms_active_tail_tokens,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond, adarms_active_tail_tokens
                    )

                # First do cross-attention between prefix and suffix, then conduct mlp transformation independently for each expert.
                # Old code removed - now using compute_layer_complete function above

            # final norm
            # Define final norm computation function for gradient checkpointing
            def compute_final_norms(inputs_embeds, adarms_cond, adarms_active_tail_tokens):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    if hidden_states is None:
                        outputs_embeds.append(None)
                        continue
                    out_emb, _ = self._call_adarms_norm(
                        models[i].norm,
                        hidden_states,
                        adarms_cond[i],
                        adarms_active_tail_tokens[i],
                    )
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    adarms_cond,
                    adarms_active_tail_tokens,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond, adarms_active_tail_tokens)

            # Output order matches input order: [vlm, tactile, action] for three experts, [vlm, action] for two experts
            if self.use_tactile_input and self.gemma_tactile_expert is not None:
                vlm_output = outputs_embeds[0]  # index 0: vlm
                tactile_output = outputs_embeds[1]  # index 1: tactile
                action_output = outputs_embeds[2]  # index 2: action
                past_key_values = None
            else:
                vlm_output = outputs_embeds[0]  # index 0: vlm
                action_output = outputs_embeds[1]  # index 1: action
                tactile_output = None
                past_key_values = None

        # Return outputs based on number of experts
        # Order: [vlm, tactile, action] for three experts, [vlm, action] for two experts
        if self.use_tactile_input and self.gemma_tactile_expert is not None:
            return [vlm_output, tactile_output, action_output], past_key_values
        return [vlm_output, action_output], past_key_values
