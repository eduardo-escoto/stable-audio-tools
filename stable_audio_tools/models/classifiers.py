import torch
import torch.nn as nn

from .dit import DiffusionTransformer


class DiTClassifier(nn.Module):
    """
    Binary classifier built on top of (the first half of) a DiffusionTransformer backbone.

    The DiT backbone is run with exit_layer_ix set to its final layer so that we
    get raw transformer hidden states (b, seq, embed_dim) before the postprocess
    projection.  The sequence is mean-pooled and fed into a single linear binary head
    (outputs a scalar logit).  Train with binary cross-entropy loss.
    """

    def __init__(
        self,
        dit: DiffusionTransformer,
        embed_dim: int,
    ):
        """
        Args:
            dit: A DiffusionTransformer instance (typically half-depth).
            embed_dim: Embedding dimension of the transformer (used for the head).
        """
        super().__init__()
        self.dit = dit
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x, t=None, **kwargs):
        """
        Args:
            x: Audio latents (b, io_channels, t).
            t: Diffusion timestep tensor (b,).  Defaults to zeros if not provided.
            **kwargs: Forwarded to DiffusionTransformer (cross_attn_cond, etc.).

        Returns:
            Logits of shape (b,) — raw binary logits (no sigmoid applied).
        """
        if t is None:
            t = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        # exit at the last layer to get (b, seq, embed_dim) features,
        # bypassing postprocess_conv / project_out
        depth = len(self.dit.transformer.layers)
        features = self.dit(x, t, exit_layer_ix=depth - 1, **kwargs)
        # features: (b, seq, embed_dim) — mean-pool over sequence
        pooled = features.mean(dim=1)

        return self.head(pooled).squeeze(-1)  # (b,)


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

    Architecture must be provided explicitly via ``model.diffusion.config``.
    ``checkpoint_path`` is optional — when provided, matching weights are loaded
    from the checkpoint via suffix-based matching.

    Example config shape::

        {
            "model_type": "dit_classifier",
            "model": {
                "diffusion": {
                    "type": "dit",
                    "config": { "io_channels": 64, "embed_dim": 1024, "depth": 16, ... }
                },
                "classifier": {
                    "depth_fraction": 0.5,
                    "checkpoint_path": "/path/to/dit.ckpt"   <-- optional
                }
            }
        }
    """
    model_config = config["model"]
    classifier_config = model_config.get("classifier", {})

    depth_fraction  = classifier_config.get("depth_fraction", 0.5)
    checkpoint_path = classifier_config.get("checkpoint_path", None)

    diffusion_config = model_config.get("diffusion", {})
    dit_config       = diffusion_config.get("config", None)
    assert dit_config is not None, (
        "Must provide 'model.diffusion.config' with the full DiT architecture."
    )

    full_depth   = dit_config["depth"]
    target_depth = max(1, int(full_depth * depth_fraction))
    embed_dim    = dit_config["embed_dim"]

    half_config = {**dit_config, "depth": target_depth}
    dit = DiffusionTransformer(**half_config)
    classifier = DiTClassifier(dit, embed_dim)

    if checkpoint_path is not None:
        loaded, skipped = unstrict_load_checkpoint_weights(classifier, checkpoint_path)
        print(
            f"create_dit_classifier_from_config: loaded {len(loaded)} keys, "
            f"skipped {len(skipped)} keys from '{checkpoint_path}'"
        )

    return classifier


