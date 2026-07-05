"""End-to-end synthetic demo of the convertible bond engine.

Runs the complete analytical loop on seeded synthetic data:

    1. Price a 5y semi-annual convertible with the Tsiveriotis-Fernandes
       Crank-Nicolson solver; save the V = B + C decomposition plot.
    2. Clean a mock TRACE transaction log (with injected cancels,
       inter-dealer duplicates, fat-finger prints, and odd lots) into a
       volume-weighted bond-day panel with YTM spreads.
    3. Run a [-5, +5] market-model event study over synthetic issuance
       events with injected arbitrage price pressure; save the CAR plot.
    4. Regress event CARs on short-interest spikes (HC1 errors) plus a
       reversal-window falsification regression.

Usage (from the engine root):

    py run_pipeline_demo.py

All figures land in outputs/ for Obsidian embedding. Exit code 0 means
every stage ran clean end-to-end.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ENGINE_ROOT = Path(__file__).resolve().parent
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from backtest.price_pressure import PricePressureRegression, PricePressureSpec
from models.pde_solver import (
    ConvertibleBondSpec,
    FiniteDifferenceGrid,
    MarketParams,
    TsiveriotisFernandesSolver,
)
from models.visualizer import export_interactive_grid
from pipeline.event_study import EventStudyEngine
from pipeline.trace_cleaner import TraceCleaningConfig, TraceDataCleaner
from tests.test_synthetic_harness import build_synthetic_event_inputs

OUTPUTS = ENGINE_ROOT / "outputs"
SEED = 20260705


def stage_1_valuation() -> None:
    print("=" * 68)
    print("STAGE 1 | Tsiveriotis-Fernandes PDE valuation")
    print("=" * 68)
    bond = ConvertibleBondSpec(
        face_value=1000.0,
        coupon_rate=0.025,
        coupon_frequency=2,
        conversion_ratio=20.0,
        maturity=5.0,
        credit_spread=0.03,
        recovery_rate=0.40,
    )
    market = MarketParams(risk_free_rate=0.04, volatility=0.35, dividend_yield=0.01)
    grid = FiniteDifferenceGrid.for_bond(bond, n_space=400, n_time=400)
    solver = TsiveriotisFernandesSolver(grid=grid, bond=bond, market=market)
    result = solver.solve()

    spot = bond.conversion_price  # at-the-money conversion
    v = result.price(spot)
    b = float(np.interp(spot, grid.s_nodes, result.cash_component))
    straight = solver.lower_boundary_value(0.0)
    print(f"  spot = conversion price = {spot:.2f}")
    print(f"  convertible value V     = {v:,.2f}")
    print(f"  cash-only component B   = {b:,.2f}")
    print(f"  equity component C      = {v - b:,.2f}")
    print(f"  straight-debt floor     = {straight:,.2f}")
    print(f"  conversion value        = {bond.conversion_ratio * spot:,.2f}")
    assert v > straight and v > bond.conversion_ratio * spot, "hybrid sandwich violated"

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(grid.s_nodes, result.total_value, label="V(S, 0) total", linewidth=1.8)
    ax.plot(grid.s_nodes, result.cash_component, label="B(S, 0) cash-only", linewidth=1.2)
    ax.plot(grid.s_nodes, result.equity_component, label="C(S, 0) equity", linewidth=1.2)
    ax.plot(
        grid.s_nodes, bond.conversion_ratio * grid.s_nodes,
        linestyle=":", color="grey", linewidth=1.0, label="conversion value κS",
    )
    ax.set_xlim(0.0, 2.5 * bond.conversion_price)
    ax.set_ylim(0.0, 2.5 * bond.face_value)
    ax.set_xlabel("Stock price S")
    ax.set_ylabel("Value")
    ax.set_title("Tsiveriotis-Fernandes decomposition at t = 0")
    ax.legend(frameon=False)
    fig.tight_layout()
    out = OUTPUTS / "tf_decomposition_t0.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved plot -> {out.relative_to(ENGINE_ROOT)}")

    html = export_interactive_grid(solver, OUTPUTS / "interactive_tf_grid.html")
    print(f"  saved interactive grid -> {html.relative_to(ENGINE_ROOT)} "
          f"({html.stat().st_size / 1e6:.1f} MB, self-contained)")


def _mock_trace_log(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic TRACE prints with injected data pathologies, plus a
    matching constant-maturity Treasury curve."""
    cusips = [f"12345{k}AB{k}" for k in range(6)]
    dates = pd.bdate_range("2025-03-03", periods=40)
    coupon_by_cusip = {c: float(rng.uniform(0.02, 0.06)) for c in cusips}
    maturity_by_cusip = {c: float(rng.uniform(2.0, 9.0)) for c in cusips}

    rows = []
    for cusip in cusips:
        base = 100.0 + rng.normal(0.0, 3.0)
        for date in dates:
            base += rng.normal(0.0, 0.30)
            for _ in range(int(rng.poisson(6)) + 2):
                rows.append(
                    {
                        "cusip_id": cusip,
                        "trd_exctn_dt": date,
                        "trd_exctn_tm": f"{rng.integers(9, 17):02d}:"
                        f"{rng.integers(0, 60):02d}:{rng.integers(0, 60):02d}",
                        "rptd_pr": base + rng.normal(0.0, 0.15),
                        "entrd_vol_qt": float(np.round(rng.lognormal(11.5, 1.0), -2)),
                        "trc_st": "T",
                        "asof_cd": "",
                        "cntra_mp_id": str(rng.choice(["C", "D"], p=[0.45, 0.55])),
                        "coupon_rate": coupon_by_cusip[cusip],
                        "remaining_maturity": maturity_by_cusip[cusip],
                    }
                )
    trades = pd.DataFrame(rows)

    # Pathology 1: inter-dealer double counts (exact copies of D prints).
    dealer_rows = trades.loc[trades["cntra_mp_id"] == "D"]
    dupes = dealer_rows.sample(frac=0.05, random_state=1)
    # Pathology 2: cancellations — copy prints and flag the copy 'C';
    # the cleaner must drop both the instruction and its original.
    cancels = trades.sample(frac=0.03, random_state=2).assign(trc_st="C")
    # Pathology 3: fat-finger prints far from the same-day median.
    fat = trades.sample(frac=0.01, random_state=3).copy()
    fat["rptd_pr"] = fat["rptd_pr"] * 1.35
    # Pathology 4: sub-institutional odd lots below the volume floor.
    odd = trades.sample(frac=0.02, random_state=4).copy()
    odd["entrd_vol_qt"] = 1_000.0
    trades = pd.concat([trades, dupes, cancels, fat, odd], ignore_index=True)

    curve = pd.DataFrame(
        [
            {"date": date, "tenor_years": tenor,
             "yield": 0.030 + 0.004 * np.log1p(tenor) + rng.normal(0.0, 0.0004)}
            for date in dates
            for tenor in (0.5, 2.0, 5.0, 10.0)
        ]
    )
    return trades, curve


def stage_2_trace_cleaning(rng: np.random.Generator) -> None:
    print("=" * 68)
    print("STAGE 2 | TRACE cleaning cascade -> bond-day panel")
    print("=" * 68)
    trades, curve = _mock_trace_log(rng)
    cleaner = TraceDataCleaner(TraceCleaningConfig(min_par_volume=10_000.0))
    panel = cleaner.clean(trades, treasury_curve=curve)

    print(f"  raw prints              = {len(trades):,}")
    print(f"  bond-day panel rows     = {len(panel):,}")
    print(f"  bonds x days            = {panel['cusip_id'].nunique()} x "
          f"{panel['trd_exctn_dt'].nunique()}")
    print(f"  mean VWAP price         = {panel['vwap_price'].mean():.2f}")
    print(f"  mean YTM spread         = {panel['ytm_spread'].mean() * 1e4:,.1f} bp")
    assert not panel["vwap_price"].isna().any(), "NaN VWAP in cleaned panel"
    assert (panel["n_trades"] >= 1).all()


def stage_3_and_4_event_study(rng: np.random.Generator) -> None:
    print("=" * 68)
    print("STAGE 3 | [-5, +5] event study with injected price pressure")
    print("=" * 68)
    n_events = 24
    si_spike = rng.uniform(0.5, 3.5, n_events)  # dSI, % of shares outstanding
    injected = list(-0.006 * si_spike)          # true pressure: -60bp per ΔSI unit
    returns, market, events, _ = build_synthetic_event_inputs(
        n_events=n_events, n_days=280, pressure_by_event=injected, seed=SEED + 1
    )
    engine = EventStudyEngine(returns=returns, market_returns=market, events=events)
    results = engine.run()
    out = engine.plot_car(results, OUTPUTS / "car_minus5_plus5.png")
    html = engine.export_interactive_car(results, OUTPUTS / "interactive_car_study.html")
    print(f"  events used             = {results.n_events}")
    print(f"  mean CAR [-5, +5]       = {results.car_by_event.mean() * 100:.2f}%")
    print(f"  cross-sectional t       = {results.t_stat:.2f}  (p = {results.p_value:.4f})")
    print(f"  saved plot -> {out.relative_to(ENGINE_ROOT)}")
    print(f"  saved interactive CAR  -> {html.relative_to(ENGINE_ROOT)} "
          f"({html.stat().st_size / 1e6:.1f} MB, self-contained)")

    print("=" * 68)
    print("STAGE 4 | Price-pressure regression: CAR ~ dSI + controls")
    print("=" * 68)
    event_panel = pd.DataFrame(
        {
            "event_id": results.car_by_event.index,
            "car": results.car_by_event.to_numpy(),
            "delta_short_interest": si_spike,
            "log_issue_size": rng.normal(5.0, 1.0, n_events),
            "amihud": np.abs(rng.normal(0.5, 0.2, n_events)),
            "pre_event_vol": rng.uniform(0.2, 0.6, n_events),
            # Reversal window: pressure partially unwinds (opposite sign).
            "car_reversal": -0.5 * results.car_by_event.to_numpy()
            + rng.normal(0.0, 0.01, n_events),
        }
    ).set_index("event_id")

    reg = PricePressureRegression(event_panel, PricePressureSpec(cov_type="HC1"))
    fit = reg.fit()
    gamma1 = fit.params["delta_short_interest"]
    t1 = fit.tvalues["delta_short_interest"]
    print(f"  gamma_1 (dSI)           = {gamma1:.5f}  (true injected: -0.00600)")
    print(f"  t-stat (HC1)            = {t1:.2f},  R^2 = {fit.rsquared:.3f},  "
          f"N = {int(fit.nobs)}")
    assert gamma1 < 0.0, "pressure regression failed to recover negative gamma_1"

    rev = reg.fit_reversal()
    print(f"  reversal gamma_1        = {rev.params['delta_short_interest']:.5f} "
          f"(opposite sign => transitory pressure)")


def main() -> int:
    OUTPUTS.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    stage_1_valuation()
    stage_2_trace_cleaning(rng)
    stage_3_and_4_event_study(rng)
    print("=" * 68)
    print("ALL STAGES COMPLETED CLEANLY")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
