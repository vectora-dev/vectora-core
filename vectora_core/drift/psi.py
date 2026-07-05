"""
Population Stability Index (PSI) drift detection.

PSI thresholds (industry standard):
  < 0.10  — stable     — no action required
  0.10–0.20 — low      — monitor closely
  0.20–0.25 — medium   — investigate, consider retraining
  >= 0.25  — critical  — retrain immediately

This implementation estimates PSI from summary statistics (mean, std, percentiles)
rather than requiring raw data. This matches the Vectora ingest format where only
summary stats are transmitted, not individual prediction values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DriftSeverity = Literal["stable", "low", "medium", "critical"]

_PSI_LOW      = 0.10
_PSI_MEDIUM   = 0.20
_PSI_CRITICAL = 0.25


@dataclass(frozen=True)
class FeatureStats:
    """Summary statistics for a single feature, as transmitted by vectora-sdk."""
    mean: float
    std: float
    p25: float
    p50: float
    p75: float


def compute_psi(baseline: FeatureStats, current: FeatureStats) -> float:
    """Estimate PSI between *baseline* and *current* summary statistics.

    Returns a float rounded to 4 decimal places.

    The estimate uses median shift normalised by baseline standard deviation,
    adjusted upward when the shift is large or the spread has changed significantly.
    This mirrors the algorithm in the Vectora Supabase edge function so that
    server-side and client-side scores are consistent.
    """
    baseline_std = max(abs(baseline.std), 0.0001)
    median_shift = abs(current.p50 - baseline.p50) / baseline_std
    std_ratio    = abs(current.std) / baseline_std

    psi = median_shift * 0.15

    if median_shift > 1.5:
        psi += 0.03

    if std_ratio > 2.0 or std_ratio < 0.5:
        psi += 0.03

    return round(psi, 4)


def get_severity(psi_score: float) -> DriftSeverity:
    """Map a PSI score to a drift severity label."""
    if psi_score < _PSI_LOW:
        return "stable"
    if psi_score < _PSI_MEDIUM:
        return "low"
    if psi_score < _PSI_CRITICAL:
        return "medium"
    return "critical"


def compute_feature_drift(
    baseline: dict[str, FeatureStats],
    current: dict[str, FeatureStats],
) -> dict[str, dict[str, float | str]]:
    """Compute PSI and severity for every feature present in both *baseline* and *current*.

    Returns a mapping of feature_name → {"psi": float, "severity": str}.
    Features present in only one side are silently skipped.
    """
    results: dict[str, dict[str, float | str]] = {}

    for feature, baseline_stats in baseline.items():
        if feature not in current:
            continue
        psi = compute_psi(baseline_stats, current[feature])
        results[feature] = {
            "psi":      psi,
            "severity": get_severity(psi),
        }

    return results
