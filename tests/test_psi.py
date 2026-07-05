import pytest

from vectora_core.drift.psi import (
    FeatureStats,
    compute_psi,
    compute_feature_drift,
    get_severity,
)


def make_stats(mean=0.0, std=1.0, p25=-0.67, p50=0.0, p75=0.67) -> FeatureStats:
    return FeatureStats(mean=mean, std=std, p25=p25, p50=p50, p75=p75)


def test_stable_identical():
    stats = make_stats()
    assert compute_psi(stats, stats) == 0.0


def test_stable_small_shift():
    baseline = make_stats(p50=0.0)
    current  = make_stats(p50=0.05)
    assert get_severity(compute_psi(baseline, current)) == "stable"


def test_critical_large_shift():
    baseline = make_stats(std=1.0, p50=0.0)
    current  = make_stats(std=1.0, p50=3.0)
    psi = compute_psi(baseline, current)
    assert get_severity(psi) == "critical"


def test_std_change_increases_psi():
    baseline = make_stats(std=1.0, p50=0.0)
    current_narrow = make_stats(std=0.1, p50=0.0)   # std ratio < 0.5 → penalty
    current_stable = make_stats(std=1.0, p50=0.0)
    assert compute_psi(baseline, current_narrow) > compute_psi(baseline, current_stable)


def test_severity_thresholds():
    assert get_severity(0.05)  == "stable"
    assert get_severity(0.10)  == "low"
    assert get_severity(0.15)  == "low"
    assert get_severity(0.20)  == "medium"
    assert get_severity(0.25)  == "critical"


def test_compute_feature_drift_skips_missing():
    baseline = {"age": make_stats(), "income": make_stats()}
    current  = {"age": make_stats(p50=5.0)}  # income missing from current
    result = compute_feature_drift(baseline, current)
    assert "age" in result
    assert "income" not in result


def test_compute_feature_drift_structure():
    baseline = {"age": make_stats()}
    current  = {"age": make_stats(p50=0.5)}
    result = compute_feature_drift(baseline, current)
    assert "psi" in result["age"]
    assert "severity" in result["age"]
