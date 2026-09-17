"""Model loading, hidden-state capture, run storage, reproducibility.

Primary entry points:
    load_model(config_path) -> ModelBundle
    capture_hidden_states(model_bundle, text) -> CaptureResult
    capture_generation_hidden_states(model_bundle, text) -> GenerationCaptureResult
    load_run(run_dir) -> Run | GenerationRun
    collect_runs(runs_dir, prompt_ids) -> dict[str, Run | GenerationRun]
"""

from sdq.instrumentation.model_loader import ModelBundle, load_model
from sdq.instrumentation.hidden_capture import (
    CaptureResult,
    GenerationCaptureResult,
    capture_hidden_states,
    capture_generation_hidden_states,
)
from sdq.instrumentation.run_storage import (
    Run,
    GenerationRun,
    load_run,
    collect_runs,
    save_run,
)
from sdq.instrumentation.reproducibility import set_seed

__all__ = [
    "ModelBundle",
    "load_model",
    "CaptureResult",
    "GenerationCaptureResult",
    "capture_hidden_states",
    "capture_generation_hidden_states",
    "Run",
    "GenerationRun",
    "load_run",
    "collect_runs",
    "save_run",
    "set_seed",
]
