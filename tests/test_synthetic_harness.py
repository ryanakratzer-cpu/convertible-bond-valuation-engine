"""Synthetic self-verification harness for the TF engine.

Originally written *before* the solver internals existed, to lock down
the mathematical boundary contract; now that the Crank-Nicolson march
and the event-study pipeline are implemented, every assertion runs as a
strict test. All assertions are no-arbitrage or limiting-case
identities that any correct implementation must satisfy.

Layout:
    - synthetic data generators (seeded GBM paths, market-model panel);
    - grid integrity tests;
    - terminal payoff tests            (boundary condition 3);
    - lower boundary S -> 0 tests      (boundary condition 1);
    - upper boundary S -> inf tests    (boundary condition 2);
    - live contract tests: solver surfaces are arbitrage-consistent,
      the event engine recovers a known market model on synthetic data;
    - convergence tests against closed-form limits (straight-bond
      risky-debt limit, conversion-value dominance).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from models.pde_solver import (
    ConvertibleBondSpec,
    FiniteDifferenceGrid,
    MarketParams,
    TsiveriotisFernandesSolver,
)
from pipeline.event_study import EventStudyEngine, EventWindow

RNG_SEED = 20260705


# --------------------------------------------------------------------------- #
# Synthetic data generators
# --------------------------------------------------------------------------- #
def simulate_gbm_paths(
    s0: float,
    mu: float,
    sigma: float,
    horizon: float,
    n_steps: int,
    n_paths: int,
    seed: int = RNG_SEED,
) -> np.ndarray:
    """Exact-discretisation GBM: S_{t+dt} = S_t exp((mu - sigma^2/2)dt + sigma sqrt(dt) Z)."""
    rng = np.random.default_rng(seed)
    dt = horizon / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_increments = (mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z
    log_paths = np.cumsum(log_increments, axis=1)
    return s0 * np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))


def build_synthetic_event_inputs(
    n_events: int = 8,
    n_days: int = 260,
    alpha: float = 0.0,
    pressure_by_event: list[float] | None = None,
    seed: int = RNG_SEED,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[float]]:
    """Multi-security panel generated exactly from R = alpha + beta R_m + eps.

    One event per synthetic security, staggered mid-sample so every
    event has a full [-120, +5] history. When ``pressure_by_event`` is
    given, event j's returns are shocked by pressure_j / 3 on relative
    days -1, 0, +1 — a known injected CAR for pressure regressions.

    Returns (returns_panel, market_series, events_frame, true_betas).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-02", periods=n_days)
    market = pd.Series(rng.normal(0.0004, 0.01, n_days), index=dates, name="mkt_ret")

    frames: list[pd.DataFrame] = []
    events: list[dict[str, object]] = []
    betas: list[float] = []
    for j in range(n_events):
        beta = float(0.8 + 1.2 * rng.random())
        betas.append(beta)
        ret = alpha + beta * market.to_numpy() + rng.normal(0.0, 0.005, n_days)
        pos = 130 + int(rng.integers(0, n_days - 140))
        if pressure_by_event is not None:
            ret[pos - 1 : pos + 2] += pressure_by_event[j] / 3.0
        sid = f"SYN{j:02d}"
        frames.append(pd.DataFrame({"security_id": sid, "date": dates, "ret": ret}))
        events.append(
            {"event_id": f"E{j:02d}", "security_id": sid, "event_date": dates[pos]}
        )
    return (
        pd.concat(frames, ignore_index=True),
        market,
        pd.DataFrame(events),
        betas,
    )


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def bond() -> ConvertibleBondSpec:
    """Vanilla 5y semi-annual 2.5% convertible, kappa = 20 (conv. price 50)."""
    return ConvertibleBondSpec(
        face_value=1000.0,
        coupon_rate=0.025,
        coupon_frequency=2,
        conversion_ratio=20.0,
        maturity=5.0,
        credit_spread=0.03,
        recovery_rate=0.40,
    )


@pytest.fixture
def market() -> MarketParams:
    return MarketParams(risk_free_rate=0.04, volatility=0.35, dividend_yield=0.01)


@pytest.fixture
def grid(bond: ConvertibleBondSpec) -> FiniteDifferenceGrid:
    return FiniteDifferenceGrid.for_bond(bond, n_space=200, n_time=100)


@pytest.fixture
def solver(
    grid: FiniteDifferenceGrid, bond: ConvertibleBondSpec, market: MarketParams
) -> TsiveriotisFernandesSolver:
    return TsiveriotisFernandesSolver(grid=grid, bond=bond, market=market)


# --------------------------------------------------------------------------- #
# Grid integrity
# --------------------------------------------------------------------------- #
class TestFiniteDifferenceGrid:
    def test_mesh_shapes_and_spacing(self, grid: FiniteDifferenceGrid) -> None:
        assert grid.s_nodes.shape == (grid.n_space + 1,)
        assert grid.t_nodes.shape == (grid.n_time + 1,)
        assert np.allclose(np.diff(grid.s_nodes), grid.ds)
        assert np.allclose(np.diff(grid.t_nodes), grid.dt)

    def test_mesh_endpoints(self, grid: FiniteDifferenceGrid) -> None:
        assert grid.s_nodes[0] == 0.0
        assert grid.s_nodes[-1] == pytest.approx(grid.s_max)
        assert grid.t_nodes[0] == 0.0
        assert grid.t_nodes[-1] == pytest.approx(grid.maturity)

    def test_coverage_heuristic_spans_conversion_price(
        self, grid: FiniteDifferenceGrid, bond: ConvertibleBondSpec
    ) -> None:
        # S_max must sit deep in the conversion region for BC(2) accuracy.
        assert grid.s_max >= 3.0 * bond.conversion_price

    def test_gbm_paths_stay_inside_grid_mostly(
        self, grid: FiniteDifferenceGrid, bond: ConvertibleBondSpec, market: MarketParams
    ) -> None:
        # Seeded synthetic paths starting at the conversion price should
        # almost never breach the truncation edge over the bond's life.
        paths = simulate_gbm_paths(
            s0=bond.conversion_price,
            mu=market.risk_free_rate,
            sigma=market.volatility,
            horizon=bond.maturity,
            n_steps=252,
            n_paths=2000,
        )
        breach_rate = (paths.max(axis=1) > grid.s_max).mean()
        # A ~5% running-max breach probability is expected at 4x coverage
        # with sigma = 0.35 over 5y; the Dirichlet condition V = kappa*S
        # is near-exact that deep ITM, so 10% is the sanity threshold.
        assert breach_rate < 0.10

    def test_invalid_grid_rejected(self) -> None:
        with pytest.raises(ValueError):
            FiniteDifferenceGrid(s_max=-1.0, n_space=100, n_time=100, maturity=1.0)
        with pytest.raises(ValueError):
            FiniteDifferenceGrid(s_max=100.0, n_space=2, n_time=100, maturity=1.0)


# --------------------------------------------------------------------------- #
# Boundary condition 3: terminal payoff at t = T
# --------------------------------------------------------------------------- #
class TestTerminalPayoff:
    def test_payoff_is_max_of_redemption_and_conversion(
        self, solver: TsiveriotisFernandesSolver, bond: ConvertibleBondSpec
    ) -> None:
        v_t, _ = solver.terminal_payoff()
        s = solver.grid.s_nodes
        redemption = bond.face_value + bond.coupon_amount
        expected = np.maximum(redemption, bond.conversion_ratio * s)
        np.testing.assert_allclose(v_t, expected)

    def test_payoff_at_zero_stock_is_full_redemption(
        self, solver: TsiveriotisFernandesSolver, bond: ConvertibleBondSpec
    ) -> None:
        v_t, b_t = solver.terminal_payoff()
        assert v_t[0] == pytest.approx(bond.face_value + bond.coupon_amount)
        # At S = 0 the entire terminal value is a cash claim.
        assert b_t[0] == pytest.approx(v_t[0])

    def test_deep_in_the_money_payoff_is_pure_conversion(
        self, solver: TsiveriotisFernandesSolver, bond: ConvertibleBondSpec
    ) -> None:
        v_t, b_t = solver.terminal_payoff()
        s = solver.grid.s_nodes
        deep = s > 2.0 * bond.conversion_price
        assert deep.any(), "grid must reach the deep-ITM region"
        np.testing.assert_allclose(v_t[deep], bond.conversion_ratio * s[deep])
        # Conversion delivers shares: the cash-only component must vanish.
        assert (b_t[deep] == 0.0).all()

    def test_cash_component_never_exceeds_total(
        self, solver: TsiveriotisFernandesSolver
    ) -> None:
        v_t, b_t = solver.terminal_payoff()
        assert (b_t <= v_t + 1e-12).all()
        assert (b_t >= 0.0).all() and (v_t >= 0.0).all()


# --------------------------------------------------------------------------- #
# Boundary condition 1: S -> 0 (pure risky debt with recovery floor)
# --------------------------------------------------------------------------- #
class TestLowerBoundary:
    def test_never_below_default_recovery_value(
        self, solver: TsiveriotisFernandesSolver, bond: ConvertibleBondSpec
    ) -> None:
        """Core invariant: at S = 0 the bond can never price below R * F."""
        floor = bond.recovery_rate * bond.face_value
        for t in solver.grid.t_nodes:
            assert solver.lower_boundary_value(float(t)) >= floor - 1e-12

    def test_bounded_above_by_undiscounted_cash_flows(
        self, solver: TsiveriotisFernandesSolver, bond: ConvertibleBondSpec
    ) -> None:
        n_coupons = int(round(bond.maturity * bond.coupon_frequency))
        total_cash = bond.face_value + n_coupons * bond.coupon_amount
        for t in solver.grid.t_nodes:
            assert solver.lower_boundary_value(float(t)) <= total_cash + 1e-9

    def test_converges_to_terminal_redemption_at_maturity(
        self, solver: TsiveriotisFernandesSolver, bond: ConvertibleBondSpec
    ) -> None:
        # tau -> 0: risky PV collapses to face + final coupon, matching BC(3) at S=0.
        v_at_T = solver.lower_boundary_value(solver.grid.maturity)
        assert v_at_T == pytest.approx(bond.face_value + bond.coupon_amount, rel=1e-6)

    def test_monotone_decreasing_in_credit_spread_until_floor(
        self, grid: FiniteDifferenceGrid, bond: ConvertibleBondSpec, market: MarketParams
    ) -> None:
        """Wider spreads must weakly reduce risky-debt value at S = 0."""
        values = []
        for spread in (0.0, 0.02, 0.05, 0.10, 0.50):
            b = ConvertibleBondSpec(
                face_value=bond.face_value,
                coupon_rate=bond.coupon_rate,
                coupon_frequency=bond.coupon_frequency,
                conversion_ratio=bond.conversion_ratio,
                maturity=bond.maturity,
                credit_spread=spread,
                recovery_rate=bond.recovery_rate,
            )
            s = TsiveriotisFernandesSolver(grid=grid, bond=b, market=market)
            values.append(s.lower_boundary_value(0.0))
        assert all(a >= b_ - 1e-12 for a, b_ in zip(values, values[1:]))

    def test_extreme_spread_pins_value_to_recovery_floor(
        self, grid: FiniteDifferenceGrid, bond: ConvertibleBondSpec, market: MarketParams
    ) -> None:
        distressed = ConvertibleBondSpec(
            face_value=bond.face_value,
            coupon_rate=bond.coupon_rate,
            coupon_frequency=bond.coupon_frequency,
            conversion_ratio=bond.conversion_ratio,
            maturity=bond.maturity,
            credit_spread=5.0,  # 500% spread: cash flows nearly worthless
            recovery_rate=bond.recovery_rate,
        )
        s = TsiveriotisFernandesSolver(grid=grid, bond=distressed, market=market)
        floor = distressed.recovery_rate * distressed.face_value
        assert s.lower_boundary_value(0.0) == pytest.approx(floor)


# --------------------------------------------------------------------------- #
# Boundary condition 2: S -> infinity (certain conversion)
# --------------------------------------------------------------------------- #
class TestUpperBoundary:
    def test_value_equals_conversion_value(
        self, solver: TsiveriotisFernandesSolver, bond: ConvertibleBondSpec
    ) -> None:
        for t in solver.grid.t_nodes:
            v, _ = solver.upper_boundary_value(float(t))
            assert v == pytest.approx(bond.conversion_ratio * solver.grid.s_max)

    def test_cash_component_extinguished(
        self, solver: TsiveriotisFernandesSolver
    ) -> None:
        """Credit risk is irrelevant when the issuer delivers shares: B = 0."""
        for t in solver.grid.t_nodes:
            _, b = solver.upper_boundary_value(float(t))
            assert b == 0.0

    def test_conversion_dominates_redemption_at_edge(
        self, solver: TsiveriotisFernandesSolver, bond: ConvertibleBondSpec
    ) -> None:
        # Consistency of the truncation: the coverage heuristic must put
        # s_max far enough out that kappa * s_max >> total cash claim.
        v, _ = solver.upper_boundary_value(0.0)
        assert v > 2.0 * bond.face_value


# --------------------------------------------------------------------------- #
# Live implementation contract: shape/consistency checks on the working
# solver and pipeline (promoted from the pre-implementation guards).
# --------------------------------------------------------------------------- #
class TestLiveContract:
    def test_solve_produces_arbitrage_consistent_surfaces(
        self, solver: TsiveriotisFernandesSolver, bond: ConvertibleBondSpec
    ) -> None:
        result = solver.solve()
        s = solver.grid.s_nodes
        assert result.total_value.shape == s.shape
        assert result.cash_component.shape == s.shape
        assert np.isfinite(result.total_value).all()
        assert np.isfinite(result.cash_component).all()
        # Decomposition sanity: 0 <= B <= V, hence C = V - B >= 0.
        assert (result.cash_component >= -1e-9).all()
        assert (result.cash_component <= result.total_value + 1e-9).all()
        assert (result.equity_component >= -1e-9).all()
        # The t = 0 lower/upper boundary nodes honour BC(1) and BC(2).
        assert result.total_value[0] == pytest.approx(
            solver.lower_boundary_value(0.0)
        )
        assert result.total_value[-1] == pytest.approx(
            bond.conversion_ratio * solver.grid.s_max
        )
        # Hybrid pricing sandwich at the conversion price: worth more
        # than both pure debt and pure conversion value.
        atm = result.price(bond.conversion_price)
        assert atm > solver.lower_boundary_value(0.0)
        assert atm > bond.conversion_ratio * bond.conversion_price

    def test_event_engine_recovers_market_model(self) -> None:
        """On data generated exactly by the market model with no event
        effect, the engine must recover beta and produce CARs near 0."""
        returns, market, events, betas = build_synthetic_event_inputs(n_events=8)
        engine = EventStudyEngine(
            returns=returns, market_returns=market, events=events
        )

        fit = engine.fit_market_model("E00")
        assert fit.nobs >= 30
        assert fit.beta == pytest.approx(betas[0], abs=0.2)
        assert abs(fit.alpha) < 0.005

        results = engine.run()
        assert results.n_events == 8
        assert results.ar_paths.shape == (11, 8)
        assert list(results.mean_car_path.index) == list(range(-5, 6))
        assert np.isfinite(results.car_by_event).all()
        # No injected effect: mean CAR must be statistically small
        # (|mean| < 0.03 is > 3 cross-sectional SEs at these noise levels).
        assert abs(results.car_by_event.mean()) < 0.03
        assert 0.0 <= results.p_value <= 1.0

    def test_event_window_validation_is_live(self) -> None:
        assert list(EventWindow().event_days) == list(range(-5, 6))
        with pytest.raises(ValueError):
            EventWindow(pre=1, post=5)          # window must straddle the event
        with pytest.raises(ValueError):
            EventWindow(estimation_start=-10, estimation_end=-21)
        with pytest.raises(ValueError):
            EventWindow(estimation_end=-3)      # overlaps the event window


# --------------------------------------------------------------------------- #
# Convergence tests against closed-form limits (strict assertions).
# --------------------------------------------------------------------------- #
class TestConvergence:
    def test_zero_conversion_reduces_to_straight_risky_bond(
        self, grid: FiniteDifferenceGrid, bond: ConvertibleBondSpec, market: MarketParams
    ) -> None:
        straight = ConvertibleBondSpec(
            face_value=bond.face_value,
            coupon_rate=bond.coupon_rate,
            coupon_frequency=bond.coupon_frequency,
            conversion_ratio=0.0,
            maturity=bond.maturity,
            credit_spread=bond.credit_spread,
            recovery_rate=bond.recovery_rate,
        )
        g = FiniteDifferenceGrid(s_max=grid.s_max, n_space=grid.n_space,
                                 n_time=grid.n_time, maturity=straight.maturity)
        result = TsiveriotisFernandesSolver(grid=g, bond=straight, market=market).solve()
        analytic = TsiveriotisFernandesSolver(g, straight, market).lower_boundary_value(0.0)
        # Without conversion rights every node must price as risky debt.
        assert result.price(straight.face_value / 20.0) == pytest.approx(analytic, rel=5e-3)

    def test_solution_dominates_conversion_value_everywhere(
        self, solver: TsiveriotisFernandesSolver, bond: ConvertibleBondSpec
    ) -> None:
        result = solver.solve()
        conversion = bond.conversion_ratio * solver.grid.s_nodes
        assert (result.total_value >= conversion - 1e-8).all()
