"""Load and configure the target LLM for hidden-state extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from sdq.instrumentation.reproducibility import set_seed

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass
class ModelBundle:
    """Everything needed to run instrumented inference."""

    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    device: torch.device
    dtype: torch.dtype
    config: dict

    @property
    def num_layers(self) -> int:
        return self.model.config.num_hidden_layers

    @property
    def hidden_dim(self) -> int:
        return self.model.config.hidden_size


def load_model(
    config_path: str | Path = "configs/model.yaml",
    model_path_override: str | None = None,
) -> ModelBundle:
    """Load model and tokenizer from a YAML config file.

    Args:
        config_path: Path to the YAML config file.
        model_path_override: If provided, overrides config["model"]["name"].
    """
    config_path = Path(config_path)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if model_path_override is not None:
        config["model"]["name"] = model_path_override

    model_cfg = config["model"]
    inf_cfg = config.get("inference", {})

    set_seed(inf_cfg.get("seed", 42))

    device = torch.device(model_cfg.get("device", "cuda"))
    dtype = _DTYPE_MAP.get(model_cfg.get("dtype", "bfloat16"), torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"],
        cache_dir=model_cfg.get("cache_dir"),
        revision=model_cfg.get("revision"),
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        torch_dtype=dtype,
        cache_dir=model_cfg.get("cache_dir"),
        revision=model_cfg.get("revision"),
        output_hidden_states=True,
    ).to(device)
    model.eval()

    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        config=config,
    )
