from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class MetricEstimate:
    metric_name: str
    predicted_value: float | None = None
    uncertainty: float | None = None
    missing: bool = False


@dataclass
class BottleneckSketch:
    primary_label: str | None = None
    secondary_labels: list[str] = field(default_factory=list)
    confidence: float | None = None
    evidence_metrics: list[str] = field(default_factory=list)


@dataclass
class AcquisitionSketch:
    predicted_benefit: float | None = None
    uncertainty_score: float | None = None
    profile_cost_estimate: float | None = None
    value_of_profile: float | None = None
    decision: str | None = None
    prediction_error_shadow: float | None = None


@dataclass
class GuidanceSketch:
    recommended_action_family: str | None = None
    discouraged_action_family: list[str] = field(default_factory=list)
    rationale: str | None = None
    prompt_safe_text: str | None = None


@dataclass
class ProfileSketch:
    task_id: int
    candidate_id: str
    parent_candidate_id: str | None = None
    cheap_features: dict[str, Any] = field(default_factory=dict)
    predicted_metrics: list[MetricEstimate] = field(default_factory=list)
    measured_metrics: dict[str, float] = field(default_factory=dict)
    bottleneck: BottleneckSketch = field(default_factory=BottleneckSketch)
    acquisition: AcquisitionSketch = field(default_factory=AcquisitionSketch)
    guidance: GuidanceSketch = field(default_factory=GuidanceSketch)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
