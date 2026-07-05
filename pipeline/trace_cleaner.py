"""TRACE corporate bond transaction cleaning and YTM-spread standardisation.

Implements a research-grade simplification of the Dick-Nielsen (2009,
2014) filter cascade for FINRA TRACE enhanced data, followed by
liquidity screens and conversion of clean prices into yield-to-maturity
spreads over a maturity-matched Treasury curve.

Cleaning cascade (order matters):
    1. Drop cancellations (trc_st in {'C', 'X'}): the cancel instruction
       itself and its matched original on
       (cusip, date, price, volume).
    2. Drop reversals (asof_cd == 'R') and their matched originals on
       the same keys.
    3. Drop agency double-counts: inter-dealer trades reported by both
       dealers (cntra_mp_id == 'D') collapse to a single economic print.
    4. Median price filter: drop trades deviating more than
       ``max_price_deviation`` from the same-day same-CUSIP median.
    5. Liquidity screens: price sanity bounds, par-volume floor (and
       optional cap), minimum trades per bond-day.

Spread standardisation:
    ytm_spread_i = ytm_i - treasury_yield(date_i, maturity_i)

where ytm is solved from the clean price with ``scipy.optimize.brentq``
(semi-annual compounding) and the Treasury yield is linearly
interpolated on the constant-maturity curve at the bond's remaining
maturity, as-of the trade date.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq

__all__ = ["TraceCleaningConfig", "TraceDataCleaner"]

_CANCEL_CODES: tuple[str, ...] = ("C", "X")


@dataclass(frozen=True)
class TraceCleaningConfig:
    """Tunable thresholds for the TRACE filter cascade.

    Attributes
    ----------
    min_par_volume:
        Minimum trade size in dollars of par (institutional screen;
        100_000 replicates the common 'institutional trades only' cut).
    max_par_volume:
        Optional cap on par volume (TRACE disseminated sizes are
        truncated at 5MM IG / 1MM HY; a cap screens synthetic outliers).
    min_trades_per_day:
        Minimum same-CUSIP prints per day for a bond-day to be kept.
    price_bounds:
        (low, high) sanity bounds on clean price as percent of par;
        trades outside are treated as data errors.
    max_price_deviation:
        Maximum fractional deviation from the same-day median price
        before a print is flagged as erroneous (0.10 = 10%).
    """

    min_par_volume: float = 10_000.0
    max_par_volume: Optional[float] = None
    min_trades_per_day: int = 1
    price_bounds: tuple[float, float] = (5.0, 250.0)
    max_price_deviation: float = 0.10


class TraceDataCleaner:
    """Ingest, filter, and standardise TRACE transaction records.

    Expected raw schema (TRACE enhanced field names):

        cusip_id        str   9-char bond CUSIP
        trd_exctn_dt    date  execution date
        trd_exctn_tm    time  execution time (optional)
        rptd_pr         float clean price, percent of par
        entrd_vol_qt    float par volume traded (dollars of face)
        trc_st          str   trade status (cancel codes 'C'/'X')
        asof_cd         str   as-of / reversal indicator ('R')
        cntra_mp_id     str   counterparty type ('D' dealer, 'C' customer)
        coupon_rate     float annual coupon (decimal) — for YTM solving
        remaining_maturity float years to maturity — for YTM solving

    The public entry point is :meth:`clean`, which applies the full
    cascade and returns a tidy volume-weighted bond-day panel.
    """

    REQUIRED_COLUMNS: ClassVar[tuple[str, ...]] = (
        "cusip_id",
        "trd_exctn_dt",
        "rptd_pr",
        "entrd_vol_qt",
    )
    #: (cusip, date, price, volume) identifies a print for cancel matching.
    MATCH_KEYS: ClassVar[tuple[str, ...]] = (
        "cusip_id",
        "trd_exctn_dt",
        "rptd_pr",
        "entrd_vol_qt",
    )
    #: Minimum estimation sample for the YTM root-finder to be attempted.
    YTM_BRACKET: ClassVar[tuple[float, float]] = (-0.5, 5.0)

    def __init__(self, config: TraceCleaningConfig | None = None) -> None:
        self.config = config or TraceCleaningConfig()

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def ingest(self, path: Path | str) -> pd.DataFrame:
        """Load raw TRACE extracts (csv/parquet by suffix), validate the
        schema against ``REQUIRED_COLUMNS``, parse dates, and coerce
        numeric dtypes. Raises ``ValueError`` on missing columns."""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            trades = pd.read_parquet(path)
        elif suffix in {".csv", ".gz", ".zip"}:
            trades = pd.read_csv(path)
        else:
            raise ValueError(f"unsupported TRACE extract format: {suffix!r}")
        return self._validate(trades)

    def _validate(self, trades: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.REQUIRED_COLUMNS if c not in trades.columns]
        if missing:
            raise ValueError(f"TRACE extract missing required columns: {missing}")
        out = trades.copy()
        out["trd_exctn_dt"] = pd.to_datetime(out["trd_exctn_dt"])
        for col in ("rptd_pr", "entrd_vol_qt"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
        return out.dropna(subset=["rptd_pr", "entrd_vol_qt"])

    # ------------------------------------------------------------------ #
    # Filters 1-2: cancels, corrections, reversals
    # ------------------------------------------------------------------ #
    def drop_cancellations_and_corrections(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Filters 1-2: remove cancel/correction instructions and the
        records they supersede, then reversals and their originals."""
        df = trades.copy()
        if "trc_st" in df.columns:
            df = self._drop_matched(df, df["trc_st"].isin(_CANCEL_CODES))
        if "asof_cd" in df.columns:
            df = self._drop_matched(df, df["asof_cd"] == "R")
        return df

    def _drop_matched(self, df: pd.DataFrame, is_instruction: pd.Series) -> pd.DataFrame:
        """Drop instruction rows plus the originals they point at,
        matched on ``MATCH_KEYS`` (simplified Dick-Nielsen matching)."""
        keys = list(self.MATCH_KEYS)
        instructions = df.loc[is_instruction, keys]
        if instructions.empty:
            return df.loc[~is_instruction]
        matched = pd.MultiIndex.from_frame(df[keys]).isin(
            pd.MultiIndex.from_frame(instructions)
        )
        return df.loc[~(matched | is_instruction.to_numpy())]

    # ------------------------------------------------------------------ #
    # Filter 3: agency double-counts
    # ------------------------------------------------------------------ #
    def drop_agency_duplicates(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Filter 3: collapse double-reported inter-dealer prints to a
        single economic trade. Customer-facing prints are never deduped."""
        df = trades.copy()
        keys = [
            c
            for c in ("cusip_id", "trd_exctn_dt", "trd_exctn_tm", "rptd_pr", "entrd_vol_qt")
            if c in df.columns
        ]
        if "cntra_mp_id" in df.columns:
            dealer = df["cntra_mp_id"] == "D"
            deduped = df.loc[dealer].drop_duplicates(subset=keys, keep="first")
            df = pd.concat([df.loc[~dealer], deduped], ignore_index=True)
            return df.sort_values(keys).reset_index(drop=True)
        return df.drop_duplicates(subset=keys, keep="first").reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Filters 4-5: price errors and liquidity screens
    # ------------------------------------------------------------------ #
    def filter_illiquid(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Filters 4-5: price sanity bounds, par-volume floor/cap, the
        same-day median price-deviation screen, and the minimum
        trades-per-day cut."""
        cfg = self.config
        df = trades.copy()
        lo, hi = cfg.price_bounds
        df = df.loc[df["rptd_pr"].between(lo, hi)]
        df = df.loc[df["entrd_vol_qt"] >= cfg.min_par_volume]
        if cfg.max_par_volume is not None:
            df = df.loc[df["entrd_vol_qt"] <= cfg.max_par_volume]
        day_median = df.groupby(["cusip_id", "trd_exctn_dt"])["rptd_pr"].transform("median")
        df = df.loc[(df["rptd_pr"] - day_median).abs() / day_median <= cfg.max_price_deviation]
        day_count = df.groupby(["cusip_id", "trd_exctn_dt"])["rptd_pr"].transform("size")
        return df.loc[day_count >= cfg.min_trades_per_day].reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # YTM and spread standardisation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _solve_ytm(
        price_pct: float, coupon_rate: float, maturity_years: float, freq: int = 2
    ) -> float:
        """Yield-to-maturity from a clean price (percent of par) with
        per-period compounding at ``freq``/year, via brentq. Returns NaN
        when no root exists inside an (expanding) bracket."""
        n = max(1, int(round(maturity_years * freq)))
        c = 100.0 * coupon_rate / freq

        def pv_minus_price(y: float) -> float:
            disc = (1.0 + y / freq) ** (-np.arange(1, n + 1, dtype=np.float64))
            return float(c * disc.sum() + 100.0 * disc[-1]) - price_pct

        lo, hi = TraceDataCleaner.YTM_BRACKET
        try:
            f_hi = pv_minus_price(hi)
            while f_hi > 0.0 and hi < 1000.0:
                hi *= 2.0
                f_hi = pv_minus_price(hi)
            if pv_minus_price(lo) < 0.0 or f_hi > 0.0:
                return float("nan")
            return float(brentq(pv_minus_price, lo, hi))
        except (ValueError, OverflowError):
            return float("nan")

    def compute_ytm_spread(
        self,
        trades: pd.DataFrame,
        treasury_curve: pd.DataFrame,
        *,
        maturity_col: str = "remaining_maturity",
    ) -> pd.DataFrame:
        """Append ``ytm`` (solved from clean price + coupon terms) and
        ``ytm_spread`` over the interpolated constant-maturity Treasury
        yield, as-of each trade date.

        Parameters
        ----------
        treasury_curve:
            Long-format frame with columns (date, tenor_years, yield),
            yields as decimals. Trade dates missing from the curve use
            the most recent prior curve date; trades before the first
            curve date get NaN spreads.
        """
        needed = ("coupon_rate", maturity_col)
        missing = [c for c in needed if c not in trades.columns]
        if missing:
            raise ValueError(f"YTM standardisation needs columns: {missing}")
        df = trades.copy()
        df["ytm"] = [
            self._solve_ytm(p, cr, mat)
            for p, cr, mat in zip(df["rptd_pr"], df["coupon_rate"], df[maturity_col])
        ]

        wide = (
            treasury_curve.pivot_table(index="date", columns="tenor_years", values="yield")
            .sort_index()
        )
        trade_dates = pd.DatetimeIndex(df["trd_exctn_dt"].unique())
        aligned = wide.reindex(wide.index.union(trade_dates)).ffill()
        tenors = wide.columns.to_numpy(dtype=np.float64)

        df["tsy_yield"] = np.nan
        for trade_date, idx in df.groupby("trd_exctn_dt").groups.items():
            curve_row = aligned.loc[trade_date].to_numpy(dtype=np.float64)
            if np.isnan(curve_row).all():
                continue
            df.loc[idx, "tsy_yield"] = np.interp(
                df.loc[idx, maturity_col].to_numpy(dtype=np.float64), tenors, curve_row
            )
        df["ytm_spread"] = df["ytm"] - df["tsy_yield"]
        return df

    # ------------------------------------------------------------------ #
    # Full cascade
    # ------------------------------------------------------------------ #
    def clean(
        self,
        trades: pd.DataFrame,
        treasury_curve: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Full cascade: filters 1-5, optional YTM-spread standardisation,
        then aggregation to a volume-weighted bond-day panel.

        Returns one row per (cusip_id, trd_exctn_dt) with columns
        vwap_price, total_volume, n_trades and, when a Treasury curve is
        supplied, volume-weighted ytm and ytm_spread.
        """
        df = self._validate(trades)
        df = self.drop_cancellations_and_corrections(df)
        df = self.drop_agency_duplicates(df)
        df = self.filter_illiquid(df)
        if treasury_curve is not None:
            df = self.compute_ytm_spread(df, treasury_curve)

        df["_pv"] = df["rptd_pr"] * df["entrd_vol_qt"]
        grouped = df.groupby(["cusip_id", "trd_exctn_dt"])
        panel = grouped.agg(
            total_volume=("entrd_vol_qt", "sum"),
            n_trades=("rptd_pr", "size"),
            _pv=("_pv", "sum"),
        )
        panel["vwap_price"] = panel.pop("_pv") / panel["total_volume"]
        if "ytm_spread" in df.columns:
            panel["ytm"] = grouped["ytm"].mean()
            panel["ytm_spread"] = grouped["ytm_spread"].mean()
        return panel.reset_index().sort_values(["cusip_id", "trd_exctn_dt"]).reset_index(
            drop=True
        )
