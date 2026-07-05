"""
SHAP wrapper — shared base used by vectora-sdk.

Requires the optional `shap` dependency:
    pip install vectora-core[shap]
"""

from __future__ import annotations

import warnings
from importlib import import_module
from typing import Any

import numpy as np

_SHAP_WARNING_EMITTED = False


class SHAPWrapper:
    """Compute mean absolute SHAP values for a fitted model and a data batch.

    Returns None instead of raising when SHAP is unavailable or the model
    type is not supported — callers must handle None gracefully.
    """

    def compute(
        self,
        model: Any,
        X: Any,
        feature_names: list[str],
    ) -> dict[str, float] | None:
        shap_module = self._import_shap()
        if shap_module is None:
            return None

        array = np.asarray(X)
        if array.ndim == 1:
            array = array.reshape(1, -1)

        explainer = self._build_explainer(shap_module, model, array)
        if explainer is None:
            return None

        try:
            shap_values = explainer(array)
            values = getattr(shap_values, "values", shap_values)
        except Exception:
            return None

        normalized = self._normalize_values(values)
        if normalized is None or normalized.shape[-1] != len(feature_names):
            return None

        mean_abs = np.mean(np.abs(normalized), axis=0)
        return {
            name: float(value)
            for name, value in zip(feature_names, mean_abs)
        }

    def _build_explainer(self, shap_module: Any, model: Any, background: np.ndarray) -> Any:
        try:
            return shap_module.Explainer(model, background)
        except Exception:
            try:
                return shap_module.Explainer(model.predict, background)
            except Exception:
                return None

    def _import_shap(self) -> Any | None:
        global _SHAP_WARNING_EMITTED

        try:
            return import_module("shap")
        except Exception:
            if not _SHAP_WARNING_EMITTED:
                warnings.warn(
                    "The 'shap' package is not installed. "
                    "Install vectora-core[shap] to enable SHAP explanations.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                _SHAP_WARNING_EMITTED = True
            return None

    def _normalize_values(self, values: Any) -> np.ndarray | None:
        array = np.asarray(values)

        if array.ndim == 3:
            array = array[..., 0]

        if array.ndim == 1:
            array = array.reshape(1, -1)

        if array.ndim != 2:
            return None

        return array
