import torch


def load_weights_qwen(
    model,
    param_config,
    params : dict
):

    def assign(left , right , tensor_name="unknown"):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch -> {tensor_name} | left : {left.shape} | right : {right.shape}") 

        return torch.nn.Parameter(right.clone().detach() if isinstance(right , torch.Tensor) else torch.tensor(right))

    model.tok_embed.weight = assign(model.tok_embed.weight , params["model.embed_tokens.weight"] , "model.embed_tokens.weight")

    for l in range(param_config.n_layers):
        block = model.transformer_blocs[l]
        attn = block.attn

        attn.Wq.weight = assign(
            attn.Wq.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight"
        )
        attn.Wk.weight = assign(
            attn.Wk.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight"
        )
        attn.Wv.weight = assign(
            attn.Wv.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight"
        )

        # Output projection
        attn.out_projection.weight = assign(
            attn.out_projection.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight"
        )

        # QK norms
        if hasattr(attn, "q_norm") and attn.q_norm is not None:
            attn.q_norm.weight = assign(
                attn.q_norm.weight,
                params[f"model.layers.{l}.self_attn.q_norm.weight"],
                f"model.layers.{l}.self_attn.q_norm.weight"
            )
        if hasattr(attn, "k_norm") and attn.k_norm is not None:
            attn.k_norm.weight = assign(
                attn.k_norm.weight,
                params[f"model.layers.{l}.self_attn.k_norm.weight"],
                f"model.layers.{l}.self_attn.k_norm.weight"
            )

        block.rms_norm1.weight = assign(
            block.rms_norm1.weight,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight"
        )

        block.ffn.linear_layer1.weight = assign(
            block.ffn.linear_layer1.weight,
            params[f"model.layers.{l}.mlp.gate_proj.weight"],
            f"model.layers.{l}.mlp.gate_proj.weight"
        )
        block.ffn.linear_layerP.weight = assign(
            block.ffn.linear_layerP.weight,
            params[f"model.layers.{l}.mlp.up_proj.weight"],
            f"model.layers.{l}.mlp.up_proj.weight"
        )
        block.ffn.linear_layer2.weight = assign(
            block.ffn.linear_layer2.weight,
            params[f"model.layers.{l}.mlp.down_proj.weight"],
            f"model.layers.{l}.mlp.down_proj.weight"
        )

        block.rms_norm2.weight = assign(
            block.rms_norm2.weight,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight"
        )

    model.final_rmsnorm.weight = assign(model.final_rmsnorm.weight, params["model.norm.weight"], "model.norm.weight")

    if "lm_head.weight" in params:
        model.out_head.weight = assign(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
    else:
        print("Model uses weight tying.")
        model.out_head.weight = assign(model.out_head.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")


def load_weights_fastqwen(model, param_config, params: dict):

    def assign(left, right, tensor_name="unknown"):
        if left.shape != right.shape:
            raise ValueError(
                f"Shape mismatch -> {tensor_name} | left : {left.shape} | right : {right.shape}"
            )
        tensor = (
            right.clone().detach()
            if isinstance(right, torch.Tensor)
            else torch.tensor(right)
        )
        tensor = tensor.to(torch.float16)

        return torch.nn.Parameter(tensor)

    model.tok_embed.weight = assign(
        model.tok_embed.weight,
        params["model.embed_tokens.weight"],
        "model.embed_tokens.weight",
    )

    for l in range(param_config.n_layers):
        block = model.transformer_blocs[l]
        attn = block.attn

        attn.Wq.weight = assign(
            attn.Wq.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight",
        )
        attn.Wk.weight = assign(
            attn.Wk.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight",
        )
        attn.Wv.weight = assign(
            attn.Wv.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight",
        )

        # Output projection
        attn.out_projection.weight = assign(
            attn.out_projection.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight",
        )

        # QK norms
        if hasattr(attn, "q_norm") and attn.q_norm is not None:
            attn.q_norm.weight = assign(
                attn.q_norm.weight,
                params[f"model.layers.{l}.self_attn.q_norm.weight"],
                f"model.layers.{l}.self_attn.q_norm.weight",
            )
        if hasattr(attn, "k_norm") and attn.k_norm is not None:
            attn.k_norm.weight = assign(
                attn.k_norm.weight,
                params[f"model.layers.{l}.self_attn.k_norm.weight"],
                f"model.layers.{l}.self_attn.k_norm.weight",
            )

        block.rms_norm1.weight = assign(
            block.rms_norm1.weight,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight",
        )

        block.ffn.linear_layer1 = assign(
            block.ffn.linear_layer1,
            params[f"model.layers.{l}.mlp.gate_proj.weight"],
            f"model.layers.{l}.mlp.gate_proj.weight",
        )
        block.ffn.linear_layerP = assign(
            block.ffn.linear_layerP,
            params[f"model.layers.{l}.mlp.up_proj.weight"],
            f"model.layers.{l}.mlp.up_proj.weight",
        )
        block.ffn.linear_layer2 = assign(
            block.ffn.linear_layer2,
            params[f"model.layers.{l}.mlp.down_proj.weight"],
            f"model.layers.{l}.mlp.down_proj.weight",
        )

        block.rms_norm2.weight = assign(
            block.rms_norm2.weight,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight",
        )

    model.final_rmsnorm.weight = assign(
        model.final_rmsnorm.weight, params["model.norm.weight"], "model.norm.weight"
    )

    if "lm_head.weight" in params:
        model.out_head.weight = assign(
            model.out_head.weight, params["lm_head.weight"], "lm_head.weight"
        )
    else:
        print("Model uses weight tying.")
        model.out_head.weight = assign(
            model.out_head.weight,
            params["model.embed_tokens.weight"],
            "model.embed_tokens.weight",
        )
