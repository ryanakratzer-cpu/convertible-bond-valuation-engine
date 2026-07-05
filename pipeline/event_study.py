"""Event-study engine for price pressure around convertible issuance.

Methodology
-----------
Standard market-model event study (MacKinlay 1997). For each event i
(a convertible debt announcement at event-date tau_i = 0):

1. **Estimation window** [est_start, est_end] (default [-120, -21],
   trading days relative to the event): fit the market model by OLS,

       R_it = alpha_i + beta_i * R_mt + eps_it,   eps ~ (0, s_i^2)

   using ``statsmodels.api.OLS``.

2. **Event window** [pre, post] (default [-5, +5]): abnormal returns

       AR_it = R_it - (alpha_hat_i + beta_hat_i * R_mt)

   and the cumulative abnormal return

       CAR_i(t1, t2) = sum_{t = t1}^{t2} AR_it .

3. **Inference**: cross-sectional t-test on mean CAR,

       t = CAR_bar / ( s(CAR) / sqrt(N) ),

   with s(CAR) the cross-event sample standard deviation (ddof = 1) and
   the p-value from Student t with N - 1 degrees of freedom.

The same machinery applies to *bond* returns (from the cleaned TRACE
panel produced by ``pipeline.trace_cleaner``) to measure issuance-window
price pressure, and to *stock* returns to capture convertible-arbitrage
delta-hedging (shorting) flow.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

__all__ = [
    "EventWindow",
    "MarketModelFit",
    "CarResults",
    "EventStudyEngine",
]


@dataclass(frozen=True)
class EventWindow:
    """Event-time layout in trading days relative to the event date (0).

    Defaults give a [-5, +5] event window with a [-120, -21] estimation
    window, leaving a gap so announcement leakage does not contaminate
    the market-model betas.
    """

    pre: int = -5
    post: int = 5
    estimation_start: int = -120
    estimation_end: int = -21

    def __post_init__(self) -> None:
        if self.pre > 0 or self.post < 0 or self.pre >= self.post:
            raise ValueError("event window must satisfy pre <= 0 <= post, pre < post")
        if self.estimation_start >= self.estimation_end:
            raise ValueError("estimation window must have start < end")
        if self.estimation_end >= self.pre:
            raise ValueError("estimation window must end before the event window opens")

    @property
    def event_days(self) -> range:
        """Relative days spanned by the event window, inclusive."""
        return range(self.pre, self.post + 1)

    @property
    def estimation_length(self) -> int:
        """Number of trading days in the estimation window."""
        return self.estimation_end - self.estimation_start + 1


@dataclass(frozen=True)
class MarketModelFit:
    """OLS market-model estimates for one event."""

    event_id: str
    alpha: float
    beta: float
    residual_variance: float
    nobs: int


@dataclass(frozen=True)
class CarResults:
    """Cross-sectional CAR summary over the event window.

    Attributes
    ----------
    car_by_event:
        Full-window CAR(pre, post) per event, indexed by event_id.
    mean_car_path:
        Average cumulative abnormal return at each relative day (the
        classic CAR plot), indexed by relative day.
    t_stat / p_value:
        Cross-sectional t-test on CAR(pre, post); NaN when fewer than
        two events survive.
    n_events:
        Number of events entering the cross-section.
    ar_paths:
        Daily abnormal returns, shape (event days x events) — kept so
        plots can draw dispersion bands without re-running the study.
    """

    car_by_event: pd.Series
    mean_car_path: pd.Series
    t_stat: float
    p_value: float
    n_events: int
    ar_paths: pd.DataFrame


class EventStudyEngine:
    """Market-model event study over a panel of security returns.

    Parameters
    ----------
    returns:
        Long-format panel with columns (security_id, date, ret) —
        stock returns or TRACE bond-day returns.
    market_returns:
        Series of market/index returns indexed by date (e.g. CRSP VW).
    events:
        Frame with columns (event_id, security_id, event_date) marking
        convertible issuance announcements.
    window:
        Event-time layout; defaults to [-5, +5] / [-120, -21].
    """

    #: Minimum estimation-window observations for a usable beta.
    MIN_ESTIMATION_OBS: ClassVar[int] = 30

    def __init__(
        self,
        returns: pd.DataFrame,
        market_returns: pd.Series,
        events: pd.DataFrame,
        window: EventWindow | None = None,
    ) -> None:
        self.returns = returns
        self.market_returns = market_returns
        self.events = events
        self.window = window or EventWindow()

    # ------------------------------------------------------------------ #
    # Event-time alignment
    # ------------------------------------------------------------------ #
    def _event_row(self, event_id: str) -> pd.Series:
        match = self.events.loc[self.events["event_id"] == event_id]
        if match.empty:
            raise ValueError(f"unknown event_id: {event_id!r}")
        return match.iloc[0]

    def align_event_time(self, event_id: str) -> pd.DataFrame:
        """Map calendar dates to relative trading days for one event.

        Returns rows (rel_day, ret, mkt_ret) covering the estimation and
        event windows. Day 0 is the first trading day on or after the
        event date. Raises ``ValueError`` when the estimation window has
        fewer than ``MIN_ESTIMATION_OBS`` rows or the event window is
        incomplete.
        """
        event = self._event_row(event_id)
        sec = (
            self.returns.loc[self.returns["security_id"] == event["security_id"]]
            .sort_values("date")
            .reset_index(drop=True)
        )
        market = self.market_returns.rename("mkt_ret").rename_axis("date").reset_index()
        merged = sec.merge(market, on="date", how="inner").sort_values("date")
        if merged.empty:
            raise ValueError(f"event {event_id}: no overlapping return/market dates")

        dates = merged["date"].to_numpy()
        pos = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(event["event_date"]))))
        if pos >= len(dates):
            raise ValueError(f"event {event_id}: event date after last trading day")

        w = self.window
        aligned = merged.assign(rel_day=np.arange(len(merged)) - pos)
        aligned = aligned.loc[
            (aligned["rel_day"] >= w.estimation_start) & (aligned["rel_day"] <= w.post)
        ]

        n_est = aligned["rel_day"].between(w.estimation_start, w.estimation_end).sum()
        if n_est < self.MIN_ESTIMATION_OBS:
            raise ValueError(
                f"event {event_id}: {n_est} estimation obs < {self.MIN_ESTIMATION_OBS}"
            )
        have = set(aligned["rel_day"])
        if not set(w.event_days) <= have:
            raise ValueError(f"event {event_id}: incomplete [-{-w.pre}, +{w.post}] window")
        return aligned[["rel_day", "ret", "mkt_ret"]].reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Estimation
    # ------------------------------------------------------------------ #
    def fit_market_model(self, event_id: str) -> MarketModelFit:
        """OLS of security returns on market returns over the estimation
        window; returns alpha_hat, beta_hat, s^2, and the sample size."""
        w = self.window
        aligned = self.align_event_time(event_id)
        est = aligned.loc[aligned["rel_day"].between(w.estimation_start, w.estimation_end)]
        design = sm.add_constant(est["mkt_ret"])
        fit = sm.OLS(est["ret"], design).fit()
        return MarketModelFit(
            event_id=event_id,
            alpha=float(fit.params["const"]),
            beta=float(fit.params["mkt_ret"]),
            residual_variance=float(fit.mse_resid),
            nobs=int(fit.nobs),
        )

    def abnormal_returns(self, event_id: str) -> pd.Series:
        """AR_t = R_t - (alpha_hat + beta_hat R_mt) on the event window,
        indexed by relative day in ``window.event_days``."""
        w = self.window
        aligned = self.align_event_time(event_id)
        model = self.fit_market_model(event_id)
        ev = (
            aligned.loc[aligned["rel_day"].between(w.pre, w.post)]
            .set_index("rel_day")
            .sort_index()
        )
        ar = ev["ret"] - (model.alpha + model.beta * ev["mkt_ret"])
        ar.name = event_id
        return ar

    # ------------------------------------------------------------------ #
    # Aggregation and inference
    # ------------------------------------------------------------------ #
    def run(self) -> CarResults:
        """Loop events -> fit -> AR -> CAR; aggregate to the mean CAR
        path and the cross-sectional t-test. Events with insufficient
        data are skipped with a warning."""
        ar_columns: dict[str, pd.Series] = {}
        for event_id in self.events["event_id"]:
            try:
                ar_columns[str(event_id)] = self.abnormal_returns(str(event_id))
            except ValueError as exc:
                warnings.warn(f"event study: skipping {event_id}: {exc}", stacklevel=2)
        if not ar_columns:
            raise ValueError("event study: no events with sufficient data")

        ar_paths = pd.DataFrame(ar_columns).sort_index()
        car_by_event = ar_paths.sum(axis=0)
        mean_car_path = ar_paths.mean(axis=1).cumsum()
        n = int(car_by_event.size)
        if n >= 2:
            spread = float(car_by_event.std(ddof=1))
            t_stat = float(car_by_event.mean() / (spread / np.sqrt(n)))
            p_value = float(2.0 * stats.t.sf(abs(t_stat), df=n - 1))
        else:
            t_stat, p_value = float("nan"), float("nan")
        return CarResults(
            car_by_event=car_by_event,
            mean_car_path=mean_car_path,
            t_stat=t_stat,
            p_value=p_value,
            n_events=n,
            ar_paths=ar_paths,
        )

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def plot_car(self, results: CarResults, output_path: Path) -> Path:
        """Save the mean CAR path with +/- 2 SE bands as a PNG for
        embedding in Obsidian notes; returns the path written."""
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rel_days = results.mean_car_path.index.to_numpy()
        mean_path = results.mean_car_path.to_numpy()

        fig, ax = plt.subplots(figsize=(8, 4.5))
        if results.n_events >= 2:
            cum_paths = results.ar_paths.cumsum(axis=0)
            band = 2.0 * cum_paths.std(axis=1, ddof=1).to_numpy() / np.sqrt(results.n_events)
            ax.fill_between(
                rel_days, mean_path - band, mean_path + band,
                alpha=0.25, linewidth=0, label="±2 SE",
            )
        ax.plot(rel_days, mean_path, marker="o", linewidth=1.6, label="Mean CAR")
        ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_xlabel("Trading days relative to event")
        ax.set_ylabel("Cumulative abnormal return")
        ax.set_title(
            f"Mean CAR [{self.window.pre}, +{self.window.post}]  "
            f"(N={results.n_events}, t={results.t_stat:.2f}, p={results.p_value:.3f})"
        )
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path

    def export_interactive_car(self, results: CarResults, output_path: Path) -> Path:
        """Export the CAR study as a self-contained interactive HTML
        (offline Plotly, plotly.js inlined) for iframe embedding in
        Obsidian notes.

        Hovering any event day shows the mean daily abnormal return,
        the cumulative abnormal return, and the per-day cross-sectional
        t-statistic on CAR(pre, day); the ±2 SE confidence band is
        drawn as a zoomable filled region. Returns the path written.
        """
        import plotly.graph_objects as go

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rel_days = results.mean_car_path.index.to_numpy()
        mean_path = results.mean_car_path.to_numpy()
        daily_ar = results.ar_paths.mean(axis=1).to_numpy()

        n = results.n_events
        if n >= 2:
            cum_paths = results.ar_paths.cumsum(axis=0)
            se = cum_paths.std(axis=1, ddof=1).to_numpy() / np.sqrt(n)
            with np.errstate(divide="ignore", invalid="ignore"):
                t_by_day = np.where(se > 0.0, mean_path / se, np.nan)
            band = 2.0 * se
        else:
            t_by_day = np.full_like(mean_path, np.nan)
            band = np.zeros_like(mean_path)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=rel_days, y=mean_path + band, name="+2 SE",
                line={"width": 0}, hoverinfo="skip", showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=rel_days, y=mean_path - band, name="±2 SE band",
                fill="tonexty", line={"width": 0},
                fillcolor="rgba(31, 119, 180, 0.20)", hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=rel_days, y=mean_path, name="Mean CAR",
                mode="lines+markers", line={"width": 2.2},
                customdata=np.column_stack([daily_ar, t_by_day]),
                hovertemplate=(
                    "day %{x}<br>"
                    "CAR = %{y:.4f}<br>"
                    "mean daily AR = %{customdata[0]:.4f}<br>"
                    "cross-sectional t = %{customdata[1]:.2f}"
                    "<extra></extra>"
                ),
            )
        )
        fig.add_vline(x=0, line={"dash": "dash", "color": "grey", "width": 1})
        fig.add_hline(y=0, line={"color": "grey", "width": 1})
        fig.update_layout(
            title={
                "text": (
                    f"Mean CAR [{self.window.pre}, +{self.window.post}]  "
                    f"(N={n}, t={results.t_stat:.2f}, p={results.p_value:.3f})"
                )
            },
            xaxis={"title": "Trading days relative to event"},
            yaxis={"title": "Cumulative abnormal return"},
            hovermode="x unified",
            template="plotly_white",
            legend={"orientation": "h", "y": 1.08},
        )
        fig.write_html(output_path, include_plotlyjs=True, full_html=True)
        return output_path
