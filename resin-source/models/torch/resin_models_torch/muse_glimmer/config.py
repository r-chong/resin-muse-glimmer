"""Muse Glimmer checkpoint configuration, read without Transformers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, kw_only=True)
class MuseGlimmerTextConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    post_norm_eps: float
    qk_scale_factor: float
    sliding_window: int
    layer_types: tuple[str, ...]
    layer_rope_theta: tuple[float, ...]
    final_logit_softcapping: float
    output_multiplier: float
    eos_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must name every decoder layer")
        if len(self.layer_rope_theta) != self.num_hidden_layers:
            raise ValueError("layer_rope_theta must name every decoder layer")
        if set(self.layer_types) - {"full_attention", "sliding_attention"}:
            raise ValueError(f"unsupported Muse layer types: {set(self.layer_types)}")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("query heads must be a multiple of key/value heads")
        if self.sliding_window < 1:
            raise ValueError("sliding_window must be positive")


@dataclass(frozen=True, kw_only=True)
class MuseGlimmerVisionConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    patch_size: int
    patch_temporal: int
    merge_size: int
    pos_emb_height: int
    pos_emb_width: int
    layer_norm_eps: float
    rope_theta: float
    layer_types: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("vision layer_types must name every layer")
        if set(self.layer_types) - {"full_attention", "window_attention"}:
            raise ValueError("unsupported Muse vision layer types")
        if self.pos_emb_height != self.pos_emb_width:
            raise ValueError("Muse vision position tables must be square")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def patch_width(self) -> int:
        return self.patch_temporal * 3 * self.patch_size**2

    @property
    def window_patches(self) -> int:
        # HF: window_size = pos_emb_height * patch_size, measured in pixels,
        # divided by patch_size with spatial_merge_size=1.
        return self.pos_emb_height


@dataclass(frozen=True, kw_only=True)
class MuseGlimmerConfig:
    text: MuseGlimmerTextConfig
    vision: MuseGlimmerVisionConfig
    image_token_id: int
    video_token_id: int
    out_hidden_size: int
    projector_hidden_size: int
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_dict(
        cls,
        config: Mapping[str, Any],
        generation_config: Mapping[str, Any] | None = None,
    ) -> MuseGlimmerConfig:
        if config.get("model_type") != "muse_glimmer":
            raise ValueError(
                f"expected a Muse Glimmer checkpoint, got {config.get('model_type')!r}"
            )
        text = config["text_config"]
        vision = config["vision_config"]
        if text.get("attention_bias"):
            raise ValueError("Muse attention projections are expected to be bias-free")
        if text.get("hidden_activation") != "silu":
            raise ValueError("Muse text MLP is expected to use SiLU")
        if text.get("tie_word_embeddings"):
            raise ValueError("Muse checkpoints are expected to have an untied lm_head")
        rope = text.get("rope_parameters") or {}
        if rope.get("rope_type", "default") != "default":
            raise ValueError("only default text RoPE is supported")
        if (
            config.get("projector_hidden_act") != "gelu"
            or vision.get("hidden_act") != "gelu"
        ):
            raise ValueError("Muse vision activations are expected to be GELU")
        eos = (generation_config or {}).get("eos_token_id", text.get("eos_token_id"))
        eos_ids = (
            tuple(eos) if isinstance(eos, list) else (() if eos is None else (eos,))
        )
        num_layers = text["num_hidden_layers"]
        theta = rope["rope_theta"]
        layer_types = text.get("layer_types") or [
            "full_attention" if (num_layers - 1 - i) % 4 == 0 else "sliding_attention"
            for i in range(num_layers)
        ]
        layer_theta = text.get("layer_rope_theta") or [
            0 if (num_layers - 1 - i) % 4 == 0 else theta for i in range(num_layers)
        ]
        vision_layers = vision["num_hidden_layers"]
        vision_types = vision.get("layer_types") or [
            "full_attention"
            if (i + 1) % 4 == 0 or i == vision_layers - 1
            else "window_attention"
            for i in range(vision_layers)
        ]
        return cls(
            text=MuseGlimmerTextConfig(
                vocab_size=text["vocab_size"],
                hidden_size=text["hidden_size"],
                intermediate_size=text["intermediate_size"],
                num_hidden_layers=num_layers,
                num_attention_heads=text["num_attention_heads"],
                num_key_value_heads=text["num_key_value_heads"],
                head_dim=text["head_dim"],
                rms_norm_eps=text["rms_norm_eps"],
                post_norm_eps=text["post_norm_eps"],
                qk_scale_factor=text["qk_scale_factor"],
                sliding_window=text["sliding_window"],
                layer_types=tuple(layer_types),
                layer_rope_theta=tuple(float(value) for value in layer_theta),
                final_logit_softcapping=text["final_logit_softcapping"],
                output_multiplier=text["output_multiplier"],
                eos_token_ids=tuple(int(value) for value in eos_ids),
            ),
            vision=MuseGlimmerVisionConfig(
                hidden_size=vision["hidden_size"],
                intermediate_size=vision["intermediate_size"],
                num_hidden_layers=vision_layers,
                num_attention_heads=vision["num_attention_heads"],
                patch_size=vision["patch_size"],
                patch_temporal=vision["patch_temporal"],
                merge_size=vision["merge_size"],
                pos_emb_height=vision["pos_emb_height"],
                pos_emb_width=vision["pos_emb_width"],
                layer_norm_eps=vision["layer_norm_eps"],
                rope_theta=(vision.get("rope_parameters") or {}).get(
                    "rope_theta", 10_000.0
                ),
                layer_types=tuple(vision_types),
            ),
            image_token_id=config["image_token_id"],
            video_token_id=config["video_token_id"],
            out_hidden_size=config["out_hidden_size"],
            projector_hidden_size=config["projector_hidden_size"],
            raw=config,
        )
