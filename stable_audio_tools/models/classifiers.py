import torch
import torch.nn as nn

from .dit import DiffusionTransformer


class DiTClassifier(nn.Module):
    """
    Classifier built on top of (the first half of) a DiffusionTransformer backbone.

    The DiT backbone is run with exit_layer_ix set to its final layer so that we
    get raw transformer hidden states (b, seq, embed_dim) before the postprocess
    projection.  Those are pooled and fed into a linear classifier head.

    Note: when conditioning inputs (prepend / global) are passed through, their
    tokens will be included in the sequence that gets pooled.  For most use cases
    this is acceptable; if you need audio-only pooling, strip the prepend tokens
    manually before calling forward.
    """

    def __init__(
        self,
        dit: DiffusionTransformer,
        num_classes: int,
        embed_dim: int,
        pool: str = "mean",
    ):
        """
        Args:
            dit: A DiffusionTransformer instance (typically half-depth).
            num_classes: Number of output classes.
            embed_dim: Embedding dimension of the transformer (used for the head).
            pool: Pooling strategy over the sequence dimension.
                  One of "mean", "max", or "cls" (first token).
        """
        super().__init__()
        assert pool in ("mean", "max", "cls"), f"Unknown pool strategy: {pool}"
        self.dit = dit
        self.pool = pool
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x, t=None, **kwargs):
        """
        Args:
            x: Audio latents (b, io_channels, t).
            t: Diffusion timestep tensor (b,).  Defaults to zeros if not provided.
            **kwargs: Forwarded to DiffusionTransformer (cross_attn_cond, etc.).

        Returns:
            Logits of shape (b, num_classes).
        """
        if t is None:
            t = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        # exit at the last layer to get (b, seq, embed_dim) features,
        # bypassing postprocess_conv / project_out
        depth = len(self.dit.transformer.layers)
        features = self.dit(x, t, exit_layer_ix=depth - 1, **kwargs)
        # features: (b, seq, embed_dim)

        if self.pool == "mean":
            pooled = features.mean(dim=1)
        elif self.pool == "max":
            pooled = features.max(dim=1).values
        elif self.pool == "cls":
            pooled = features[:, 0]

        return self.head(pooled)


def unstrict_load_checkpoint_weights(model, checkpoint_path):
    """
    Load weights from a checkpoint wherever the model state dict matches,
    using suffix-based key matching to handle prefix differences between
    the checkpoint and the model.

    For example, a DiffusionTransformer layer saved as
      "diffusion.model.model.transformer.layers.0.self_attn.to_qkv.weight"
    will match the classifier model key
      "dit.transformer.layers.0.self_attn.to_qkv.weight"
    because both share the suffix "transformer.layers.0.self_attn.to_qkv.weight".

    Keys with shape mismatches (e.g. the classifier head vs. the original
    diffusion output projection) are silently skipped.

    Args:
        model: nn.Module to load weights into.
        checkpoint_path: Path to the checkpoint file.

    Returns:
        (loaded_keys, skipped_keys) — lists of checkpoint parameter names.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Support common checkpoint conventions
    if "state_dict" in checkpoint:
        ckpt_state = checkpoint["state_dict"]
    elif "model" in checkpoint:
        ckpt_state = checkpoint["model"]
    else:
        ckpt_state = checkpoint

    model_state = model.state_dict()

    # Build suffix → model_key lookup so we can match regardless of prefix.
    # Longer suffixes are preferred (more specific), so we iterate from the
    # full key down to single-component suffixes and take the first match.
    suffix_to_model_key = {}
    for model_key in model_state:
        parts = model_key.split(".")
        for i in range(len(parts)):
            suffix = ".".join(parts[i:])
            # Only register the first (longest-suffix) model key for this suffix
            if suffix not in suffix_to_model_key:
                suffix_to_model_key[suffix] = model_key

    loaded, skipped = [], []

    for ckpt_key, value in ckpt_state.items():
        parts = ckpt_key.split(".")
        matched_model_key = None
        for i in range(len(parts)):
            suffix = ".".join(parts[i:])
            if suffix in suffix_to_model_key:
                matched_model_key = suffix_to_model_key[suffix]
                break

        if matched_model_key is not None and model_state[matched_model_key].shape == value.shape:
            model_state[matched_model_key] = value
            loaded.append(ckpt_key)
        else:
            skipped.append(ckpt_key)

    model.load_state_dict(model_state)
    return loaded, skipped


def create_dit_classifier_from_config(config: dict) -> "DiTClassifier":
    """
    Create a DiTClassifier from a config dict.

    Two modes are supported:

    1. **Explicit architecture** — provide the full DiT config under
       ``model.diffusion.config``.  ``checkpoint_path`` is optional.

        {
            "model_type": "dit_classifier",
            "model": {
                "diffusion": {
                    "type": "dit",
                    "config": { "io_channels": 64, "embed_dim": 1024, "depth": 16, ... }
                },
                "classifier": {
                    "num_classes": 2,
                    "pool": "mean",
                    "depth_fraction": 0.5,
                    "checkpoint_path": "/path/to/dit.ckpt"   <-- optional
                }
            }
        }

    2. **Checkpoint-inferred architecture** — omit ``model.diffusion.config``
       entirely.  The architecture is inferred automatically from the checkpoint
       weights (see ``create_dit_classifier_from_checkpoint``).
       ``checkpoint_path`` is required in this mode.

        {
            "model_type": "dit_classifier",
            "model": {
                "classifier": {
                    "num_classes": 2,
                    "pool": "mean",
                    "depth_fraction": 0.5,
                    "checkpoint_path": "/path/to/dit.ckpt",   <-- required
                    "diffusion_objective": "rectified_flow"   <-- optional
                }
            }
        }
    """
    model_config = config["model"]
    classifier_config = model_config.get("classifier", {})

    num_classes = classifier_config.get("num_classes", None)
    assert num_classes is not None, "Must specify 'num_classes' in classifier config"

    pool             = classifier_config.get("pool", "mean")
    depth_fraction   = classifier_config.get("depth_fraction", 0.5)
    checkpoint_path  = classifier_config.get("checkpoint_path", None)

    diffusion_config = model_config.get("diffusion", {})
    dit_config       = diffusion_config.get("config", None)

    if dit_config is None:
        # ---- checkpoint-inferred mode --------------------------------- #
        assert checkpoint_path is not None, (
            "Must provide 'checkpoint_path' in classifier config when "
            "'model.diffusion.config' is not specified."
        )
        diffusion_objective = classifier_config.get("diffusion_objective", "rectified_flow")
        return create_dit_classifier_from_checkpoint(
            checkpoint_path,
            num_classes=num_classes,
            depth_fraction=depth_fraction,
            pool=pool,
            diffusion_objective=diffusion_objective,
        )

    # ---- explicit architecture mode ----------------------------------- #
    full_depth   = dit_config["depth"]
    target_depth = max(1, int(full_depth * depth_fraction))
    embed_dim    = dit_config["embed_dim"]

    half_config = {**dit_config, "depth": target_depth}
    dit = DiffusionTransformer(**half_config)
    classifier = DiTClassifier(dit, num_classes, embed_dim, pool)

    if checkpoint_path is not None:
        loaded, skipped = unstrict_load_checkpoint_weights(classifier, checkpoint_path)
        print(
            f"create_dit_classifier_from_config: loaded {len(loaded)} keys, "
            f"skipped {len(skipped)} keys from '{checkpoint_path}'"
        )

    return classifier


def _infer_dit_config_from_state_dict(ckpt_state: dict) -> dict:
    """
    Infer DiffusionTransformer __init__ kwargs purely from checkpoint weight shapes
    and key presence.  Works regardless of what prefix wraps the DiffusionTransformer
    in the checkpoint (e.g. "diffusion.model.model." from a training wrapper).
    """
    # ------------------------------------------------------------------ #
    # 1. Find the DiffusionTransformer root prefix in the checkpoint.     #
    #    We anchor on "postprocess_conv.weight" which is unique to DiT.   #
    # ------------------------------------------------------------------ #
    prefix = ""
    for key in ckpt_state:
        if key.endswith("postprocess_conv.weight"):
            prefix = key[: -len("postprocess_conv.weight")]
            break
    if not prefix and "postprocess_conv.weight" not in ckpt_state:
        raise ValueError(
            "Cannot find 'postprocess_conv.weight' in checkpoint — "
            "is this a DiffusionTransformer checkpoint?"
        )

    def get(suffix):
        return ckpt_state.get(prefix + suffix)

    def has(suffix):
        return (prefix + suffix) in ckpt_state

    # ------------------------------------------------------------------ #
    # 2. Core dimensions                                                   #
    # ------------------------------------------------------------------ #
    io_channels = get("postprocess_conv.weight").shape[0]
    embed_dim   = get("transformer.project_in.weight").shape[0]
    # preprocess_conv shape: (dim_in, dim_in, 1)  where dim_in = io_channels + input_concat_dim
    dim_in          = get("preprocess_conv.weight").shape[0]
    input_concat_dim = dim_in - io_channels
    # project_in shape: (embed_dim, dim_in * patch_size)
    patch_size = get("transformer.project_in.weight").shape[1] // dim_in

    # ------------------------------------------------------------------ #
    # 3. Depth                                                             #
    # ------------------------------------------------------------------ #
    layer_indices = set()
    layers_prefix = prefix + "transformer.layers."
    for key in ckpt_state:
        if key.startswith(layers_prefix):
            idx = int(key[len(layers_prefix):].split(".")[0])
            layer_indices.add(idx)
    depth = max(layer_indices) + 1

    # ------------------------------------------------------------------ #
    # 4. num_heads                                                         #
    #    Primary: q_norm.weight shape (only present when qk_norm="ln")    #
    #    Fallback: rotary pos emb inv_freq shape                           #
    # ------------------------------------------------------------------ #
    q_norm_w = get("transformer.layers.0.self_attn.q_norm.weight")
    if q_norm_w is not None:
        dim_heads = q_norm_w.shape[0]
    else:
        rope_inv_freq = get("transformer.rotary_pos_emb.inv_freq")
        if rope_inv_freq is not None:
            # RotaryEmbedding(max(dim_heads//2, 32)) → inv_freq.shape = (that_dim//2,)
            rope_dim = rope_inv_freq.shape[0] * 2        # = max(dim_heads//2, 32)
            dim_heads = max(rope_dim * 2, 64)
        else:
            dim_heads = 64  # safe default
    num_heads = embed_dim // dim_heads

    # ------------------------------------------------------------------ #
    # 5. Optional conditioning dims                                        #
    # ------------------------------------------------------------------ #
    cond_embed_w   = get("to_cond_embed.0.weight")
    cond_token_dim = cond_embed_w.shape[1] if cond_embed_w is not None else 0

    global_embed_w  = get("to_global_embed.0.weight")
    global_cond_dim = global_embed_w.shape[1] if global_embed_w is not None else 0

    prepend_embed_w  = get("to_prepend_embed.0.weight")
    prepend_cond_dim = prepend_embed_w.shape[1] if prepend_embed_w is not None else 0

    # ------------------------------------------------------------------ #
    # 6. Architecture flags inferred from key presence                    #
    # ------------------------------------------------------------------ #
    cross_attend = has("transformer.layers.0.cross_attn.to_q.weight") or \
                   has("transformer.layers.0.cross_attn.to_kv.weight")

    if has("transformer.layers.0.self_attn.q_norm.weight"):
        qk_norm = "ln"
    elif has("transformer.layers.0.self_attn.q_norm.alpha"):
        qk_norm = "dyt"
    else:
        qk_norm = "none"

    global_cond_type = "adaLN" if has("transformer.layers.0.to_scale_shift_gate") else "prepend"

    config = {
        "io_channels":      io_channels,
        "embed_dim":        embed_dim,
        "depth":            depth,
        "num_heads":        num_heads,
        "patch_size":       patch_size,
        "input_concat_dim": input_concat_dim,
        "cond_token_dim":   cond_token_dim,
        "global_cond_dim":  global_cond_dim,
        "prepend_cond_dim": prepend_cond_dim,
        "transformer_type": "continuous_transformer",
        "global_cond_type": global_cond_type,
    }

    if qk_norm != "none":
        config["attn_kwargs"] = {"qk_norm": qk_norm}

    return config


def create_dit_classifier_from_checkpoint(
    checkpoint_path: str,
    num_classes: int,
    depth_fraction: float = 0.5,
    pool: str = "mean",
    diffusion_objective: str = "rectified_flow",
) -> "DiTClassifier":
    """
    Dynamically create a DiTClassifier by inferring the full architecture from
    a DiffusionTransformer checkpoint — no config file needed.

    Architecture parameters (embed_dim, depth, num_heads, io_channels,
    conditioning dims, qk_norm, etc.) are all read from weight shapes and
    key presence in the checkpoint.  The backbone depth is then scaled by
    depth_fraction (default 0.5 → first half), and matching weights are
    loaded via suffix-based matching.

    Args:
        checkpoint_path: Path to a DiffusionTransformer (or training wrapper)
                         checkpoint.
        num_classes: Number of output classes.
        depth_fraction: Fraction of the full depth to use (default 0.5).
        pool: Pooling strategy — "mean", "max", or "cls".
        diffusion_objective: Noise schedule used during training —
                             "rectified_flow" (default) or "v".
                             Not inferable from weights, must match the
                             original model's training objective.

    Returns:
        DiTClassifier with pre-trained backbone weights loaded.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "state_dict" in checkpoint:
        ckpt_state = checkpoint["state_dict"]
    elif "model" in checkpoint:
        ckpt_state = checkpoint["model"]
    else:
        ckpt_state = checkpoint

    dit_config = _infer_dit_config_from_state_dict(ckpt_state)
    dit_config["diffusion_objective"] = diffusion_objective

    full_depth    = dit_config["depth"]
    target_depth  = max(1, int(full_depth * depth_fraction))
    embed_dim     = dit_config["embed_dim"]

    half_config = {**dit_config, "depth": target_depth}
    dit = DiffusionTransformer(**half_config)
    classifier = DiTClassifier(dit, num_classes, embed_dim, pool)

    loaded, skipped = unstrict_load_checkpoint_weights(classifier, checkpoint_path)

    print(
        f"create_dit_classifier_from_checkpoint:\n"
        f"  Inferred: depth={full_depth}→{target_depth}, embed_dim={embed_dim}, "
        f"num_heads={dit_config['num_heads']}, io_channels={dit_config['io_channels']}\n"
        f"  Loaded {len(loaded)} / {len(loaded) + len(skipped)} checkpoint keys"
    )

    return classifier


def create_half_ditclass_from_full(
    dit_config: dict,
    checkpoint_path: str,
    num_classes: int,
    pool: str = "mean",
) -> DiTClassifier:
    """
    Build a DiTClassifier from the first half of a full DiffusionTransformer,
    loading matching weights from an existing checkpoint.

    The transformer depth is halved; all other architecture kwargs are preserved.
    Weight keys that do not exist in the half-depth model (e.g. layers in the
    second half, the diffusion output projection) are ignored.

    Args:
        dit_config: Keyword-argument dict for DiffusionTransformer.
                    Must include at least 'depth' and 'embed_dim'.
        checkpoint_path: Path to a full DiT checkpoint to bootstrap from.
        num_classes: Number of output classes for the classifier head.
        pool: Pooling strategy passed to DiTClassifier ("mean", "max", "cls").

    Returns:
        DiTClassifier with pre-trained weights loaded where available.
    """
    full_depth = dit_config["depth"]
    half_depth = max(1, full_depth // 2)
    embed_dim = dit_config["embed_dim"]

    half_config = {**dit_config, "depth": half_depth}
    dit = DiffusionTransformer(**half_config)

    classifier = DiTClassifier(dit, num_classes, embed_dim, pool)

    loaded, skipped = unstrict_load_checkpoint_weights(classifier, checkpoint_path)
    print(
        f"create_half_ditclass_from_full: loaded {len(loaded)} keys, "
        f"skipped {len(skipped)} keys from '{checkpoint_path}'"
    )

    return classifier
