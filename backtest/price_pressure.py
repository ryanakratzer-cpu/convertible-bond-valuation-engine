"""Cross-sectional regression of issuance-window price pressure on
arbitrage-induced short-interest spikes.

Specification
-------------
For each convertible issuance event i, with CAR_i the [-5, +5]
cumulative abnormal (stock or bond) return from
``pipeline.event_study``:

    CAR_i = gamma_0 + gamma_1 * DeltaSI_i + gamma' X_i + eta_i

where
    DeltaSI_i : change in short interest (as % of shares outstanding)
                from the pre-announcement report to the first
                post-issuance report — the convertible-arbitrage
                delta-hedge proxy;
    X_i       : controls — log issue size / market cap, Amihud
                illiquidity, pre-event volatility, rating dummies.

The permanent-vs-transitory decomposition re-runs the model on the
reversal window CAR (e.g. [+6, +30]): gamma_1 < 0 on the event window
with an offsetting sign on the reversal window is the signature of
temporary price pressure rather than information.

Inference uses heteroskedasticity-robust (HC1) or issuer-clustered
standard errors via ``statsmodels``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper

__all__ = ["PricePressureSpec", "PricePressureRegression"]


@dataclass(frozen=True)
class PricePressureSpec:
    """Column mapping and options for the pressure regression."""

    car_col: str = "car"
    short_interest_col: str = "delta_short_interest"
    controls: tuple[str, ...] = ("log_issue_size", "amihud", "pre_event_vol")
    cov_type: str = "HC1"
    cluster_col: Optional[str] = None  # e.g. "issuer_id" for clustered SEs


class PricePressureRegression:
    """OLS interface tying event-study output to short-interest data.

    Parameters
    ----------
    event_panel:
        One row per event: CAR from ``EventStudyEngine.run``, merged
        with short-interest changes and issue-level controls.
    spec:
        Column mapping / covariance options.
    """

    def __init__(self, event_panel: pd.DataFrame, spec: PricePressureSpec | None = None) -> None:
        self.event_panel = event_panel
        self.spec = spec or PricePressureSpec()

    # ------------------------------------------------------------------ #
    # Design matrix
    # ------------------------------------------------------------------ #
    def _design(self, car_col: str) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
        spec = self.spec
        columns = [car_col, spec.short_interest_col, *spec.controls]
        if spec.cluster_col is not None:
            columns.append(spec.cluster_col)
        missing = [c for c in columns if c not in self.event_panel.columns]
        if missing:
            raise ValueError(f"event panel missing columns: {missing}")
        data = self.event_panel[columns].dropna()
        if data.empty:
            raise ValueError("no complete events after listwise deletion")
        y = data[car_col]
        x = sm.add_constant(data[[spec.short_interest_col, *spec.controls]])
        return y, x, data

    def build_design_matrix(self) -> tuple[pd.Series, pd.DataFrame]:
        """Return (y, X): y = CAR, X = [const, DeltaSI, controls] with
        listwise deletion of incomplete events and aligned indices."""
        y, x, _ = self._design(self.spec.car_col)
        return y, x

    # ------------------------------------------------------------------ #
    # Estimation
    # ------------------------------------------------------------------ #
    def _fit(self, car_col: str) -> RegressionResultsWrapper:
        y, x, data = self._design(car_col)
        model = sm.OLS(y, x)
        if self.spec.cluster_col is not None:
            return model.fit(
                cov_type="cluster", cov_kwds={"groups": data[self.spec.cluster_col]}
            )
        return model.fit(cov_type=self.spec.cov_type)

    def fit(self) -> RegressionResultsWrapper:
        """Estimate the event-window specification; gamma_1 is
        ``results.params[spec.short_interest_col]``."""
        return self._fit(self.spec.car_col)

    def fit_reversal(self, reversal_car_col: str = "car_reversal") -> RegressionResultsWrapper:
        """Re-estimate on the post-event reversal-window CAR to separate
        transitory pressure from permanent information effects."""
        return self._fit(reversal_car_col)
