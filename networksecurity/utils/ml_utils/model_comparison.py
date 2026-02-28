"""
Utilities for offline model comparison with statistical rigor.

This module is intentionally lightweight and sklearn-friendly so it can be used
inside the existing training pipeline without changing model training code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score


@dataclass(frozen=True)
class ModelRegistration:
    """Container for a registered candidate or champion model."""

    name: str
    version: str
    model: Any


class ModelComparator:
    """
    Compare multiple trained models using bootstrap uncertainty + a paired test.

    Features:
    - Register multiple fitted estimators (sklearn-like .predict).
    - Evaluate each model via bootstrap resampling of a held-out test set to
      produce mean/std F1 and a 95% empirical confidence interval.
    - Compare two models on the same examples using McNemar's test on correctness
      (paired test) to decide whether the winner is statistically significant.
    """

    def __init__(self, significance_level: float = 0.05, random_state: int = 42):
        self.significance_level = float(significance_level)
        self.random_state = int(random_state)
        self._models: Dict[str, ModelRegistration] = {}

    def register_model(self, name: str, model: Any, version: str) -> None:
        """
        Register a fitted model for later evaluation.

        Args:
            name: Human-readable model identifier (must be unique).
            model: Fitted estimator with a .predict(X) method.
            version: Semver-like version string ("1.0.0") or any tag ("previous").
        """
        if not name or not isinstance(name, str):
            raise ValueError("Model name must be a non-empty string.")
        if name in self._models:
            raise ValueError(f"Model '{name}' is already registered.")
        if not hasattr(model, "predict"):
            raise TypeError("Registered model must implement .predict(X).")
        self._models[name] = ModelRegistration(name=name, version=str(version), model=model)

    def get_model(self, name: str) -> Any:
        """Return the underlying estimator for a registered model."""
        return self._models[name].model

    def list_models(self) -> Dict[str, str]:
        """Return mapping of model name -> version."""
        return {k: v.version for k, v in self._models.items()}

    def evaluate_all(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_bootstraps: int = 100,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all registered models using bootstrap resampling.

        Args:
            X_test: Test features, shape (n_samples, n_features).
            y_test: Test labels, shape (n_samples,).
            n_bootstraps: Number of bootstrap resamples.

        Returns:
            Dict keyed by model name with:
              - version
              - mean_f1, std_f1
              - ci95_low, ci95_high
              - n_bootstraps
        """
        if len(self._models) == 0:
            raise ValueError("No models registered. Call register_model() first.")

        X = np.asarray(X_test)
        y = np.asarray(y_test)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X_test and y_test must have the same number of rows.")
        if X.shape[0] < 2:
            raise ValueError("Need at least 2 test samples for bootstrap evaluation.")

        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]

        # Generate bootstrap indices once so model evaluations are paired.
        bootstrap_indices = [
            rng.integers(0, n, size=n, endpoint=False) for _ in range(int(n_bootstraps))
        ]

        results: Dict[str, Dict[str, Any]] = {}
        for name, reg in self._models.items():
            scores = np.empty(int(n_bootstraps), dtype=float)
            for b in range(int(n_bootstraps)):
                idx = bootstrap_indices[b]
                y_pred = reg.model.predict(X[idx])
                scores[b] = float(f1_score(y[idx], y_pred))

            mean_ = float(np.mean(scores))
            std_ = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
            ci_low, ci_high = np.percentile(scores, [2.5, 97.5]).tolist()
            results[name] = {
                "version": reg.version,
                "mean_f1": mean_,
                "std_f1": std_,
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "n_bootstraps": int(n_bootstraps),
            }
        return results

    def compare_models(
        self,
        model_a: str | Any,
        model_b: str | Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        *,
        use_exact_binomial_if_available: bool = True,
    ) -> Dict[str, Any]:
        """
        Compare two models using McNemar's test (paired test on correctness).

        Args:
            model_a: Registered model name OR a fitted estimator.
            model_b: Registered model name OR a fitted estimator.
            X_test: Test features.
            y_test: Test labels.
            use_exact_binomial_if_available: If True, use an exact binomial test
                for discordant pairs when SciPy is available; otherwise fall back
                to the chi-square approximation.

        Returns:
            Dict with:
              - chi2_stat
              - p_value
              - significant (p_value < significance_level)
              - winner ("model_a" or "model_b" or "tie")
              - contingency: {n11, n10, n01, n00}
        """
        est_a, label_a = self._resolve_model(model_a, default_label="model_a")
        est_b, label_b = self._resolve_model(model_b, default_label="model_b")

        X = np.asarray(X_test)
        y = np.asarray(y_test)

        pred_a = est_a.predict(X)
        pred_b = est_b.predict(X)

        correct_a = (pred_a == y)
        correct_b = (pred_b == y)

        n11 = int(np.sum(correct_a & correct_b))
        n10 = int(np.sum(correct_a & (~correct_b)))  # A correct, B wrong
        n01 = int(np.sum((~correct_a) & correct_b))  # A wrong, B correct
        n00 = int(np.sum((~correct_a) & (~correct_b)))

        discordant = n01 + n10
        if discordant == 0:
            chi2_stat = 0.0
            p_value = 1.0
        else:
            # Chi-square with Edwards continuity correction.
            chi2_stat = float(((abs(n01 - n10) - 1.0) ** 2) / discordant)
            p_value = float(self._mcnemar_p_value(n01, n10, chi2_stat, use_exact_binomial_if_available))

        if n01 > n10:
            winner = label_b
        elif n10 > n01:
            winner = label_a
        else:
            winner = "tie"

        return {
            "chi2_stat": float(chi2_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < self.significance_level),
            "winner": winner,
            "contingency": {"n11": n11, "n10": n10, "n01": n01, "n00": n00},
        }

    def _resolve_model(self, model_or_name: str | Any, default_label: str) -> Tuple[Any, str]:
        if isinstance(model_or_name, str):
            if model_or_name not in self._models:
                raise KeyError(f"Model '{model_or_name}' is not registered.")
            return self._models[model_or_name].model, model_or_name
        if not hasattr(model_or_name, "predict"):
            raise TypeError(f"{default_label} must be a registered name or an estimator with .predict(X).")
        return model_or_name, default_label

    @staticmethod
    def _mcnemar_p_value(n01: int, n10: int, chi2_stat: float, use_exact_binomial_if_available: bool) -> float:
        """
        Compute McNemar p-value for discordant pairs.

        We prefer an exact two-sided binomial test (more accurate for small counts),
        and fall back to the chi-square approximation.
        """
        discordant = n01 + n10
        if discordant == 0:
            return 1.0

        if use_exact_binomial_if_available:
            try:
                from scipy.stats import binomtest  # type: ignore

                k = min(n01, n10)
                return float(binomtest(k=k, n=discordant, p=0.5, alternative="two-sided").pvalue)
            except Exception:
                # SciPy not installed or older version.
                pass

        try:
            from scipy.stats import chi2  # type: ignore

            return float(chi2.sf(chi2_stat, df=1))
        except Exception:
            # Final fallback without SciPy: approximate p-value via exp(-chi2/2).
            # This is a rough approximation for df=1.
            return float(np.exp(-0.5 * chi2_stat))

