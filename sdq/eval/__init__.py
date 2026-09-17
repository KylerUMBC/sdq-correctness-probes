"""Evaluation and certification for SDQ (§16).

Structural certification tests:
    A. Alignment validity       — (via alignment module metrics)
    B. Transport validity       — transport_rank.transport_validity_score
    C. Cycle validity           — (via cycle_consistency module)
    D. Transfer validity        — transfer.evaluate_transfer
    E. Same-answer control      — same_answer_diff_reasoning.same_answer_test
    F. Hard-negative control    — hard_negatives.hard_negative_test
    G. Event preservation       — event_stability.event_stability_test

Plus:
    - Holdout robustness        — holdout.evaluate_holdout
    - Transport rank analysis   — transport_rank.rank_analysis
    - Early-warning metrics     — early_warning_metrics.evaluate_early_warning
"""

from sdq.eval.transport_rank import rank_analysis, batch_rank_analysis, transport_validity_score
from sdq.eval.transfer import evaluate_transfer, TransferResult
from sdq.eval.holdout import evaluate_holdout, HoldoutMetrics
from sdq.eval.same_answer_diff_reasoning import same_answer_test, SameAnswerTestResult
from sdq.eval.hard_negatives import hard_negative_test, HardNegativeResult
from sdq.eval.event_stability import event_stability_test, EventStabilityResult
from sdq.eval.composition import (
    triple_composition_test,
    lowrank_composition_error,
    batch_composition_test,
    CompositionResult,
)
from sdq.eval.retrieval import retrieval_accuracy, RetrievalResult
from sdq.eval.motion_similarity import motion_similarity_eval, MotionSimilarityResult
from sdq.eval.bird_debug import family_debug, FamilyDebugResult
from sdq.eval.semantic_scorecard import compute_scorecard, SemanticScorecard
from sdq.eval.tube_analysis import (
    full_tube_analysis,
    format_tube_report,
    TubeAnalysisReport,
    TubeCoherenceResult,
    TransverseContractionResult,
    ReasoningStageResult,
    TubeSeparationResult,
    TubeInterventionResult,
    DynamicsTubeResult,
    dynamics_tube_coherence,
)
from sdq.eval.early_warning_metrics import (
    EarlyWarningReport,
    evaluate_early_warning,
    format_early_warning_report,
    compute_auroc,
    compute_auprc,
    auroc_per_timestep,
    average_lead_time,
    calibration_analysis,
)

__all__ = [
    "rank_analysis",
    "batch_rank_analysis",
    "transport_validity_score",
    "evaluate_transfer",
    "TransferResult",
    "evaluate_holdout",
    "HoldoutMetrics",
    "same_answer_test",
    "SameAnswerTestResult",
    "hard_negative_test",
    "HardNegativeResult",
    "event_stability_test",
    "EventStabilityResult",
    "triple_composition_test",
    "lowrank_composition_error",
    "batch_composition_test",
    "CompositionResult",
    "retrieval_accuracy",
    "RetrievalResult",
    "motion_similarity_eval",
    "MotionSimilarityResult",
    "family_debug",
    "FamilyDebugResult",
    "compute_scorecard",
    "SemanticScorecard",
    "full_tube_analysis",
    "format_tube_report",
    "TubeAnalysisReport",
    "TubeCoherenceResult",
    "TransverseContractionResult",
    "ReasoningStageResult",
    "TubeSeparationResult",
    "TubeInterventionResult",
    "DynamicsTubeResult",
    "dynamics_tube_coherence",
    "EarlyWarningReport",
    "evaluate_early_warning",
    "format_early_warning_report",
    "compute_auroc",
    "compute_auprc",
    "auroc_per_timestep",
    "average_lead_time",
    "calibration_analysis",
]
