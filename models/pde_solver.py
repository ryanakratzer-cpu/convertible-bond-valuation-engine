"""Tsiveriotis-Fernandes (1998) convertible bond PDE valuation engine.

Mathematical framework
----------------------
The convertible bond value is decomposed as

    V(S, t) = B(S, t) + C(S, t)

where ``B`` is the *cash-only* component (payments the holder receives in
cash, exposed to issuer credit risk) and ``C = V - B`` is the *equity*
component (value delivered in shares, free of credit risk because the
issuer can always deliver its own stock).

The two components satisfy the coupled linear parabolic PDE system on
the domain (S, t) in [0, S_max] x [0, T]:

    dV/dt + (1/2) sigma^2 S^2 d2V/dS2 + (r - q) S dV/dS
        - r (V - B) - (r + r_c) B = 0                          (TF-1)

    dB/dt + (1/2) sigma^2 S^2 d2B/dS2 + (r - q) S dB/dS
        - (r + r_c) B = 0                                      (TF-2)

with:
    S     : underlying stock price
    t     : calendar time (t = T is maturity)
    sigma : stock volatility
    r     : continuously-compounded risk-free rate
    q     : continuous dividend yield
    r_c   : issuer credit spread applied to cash-only flows

The equity component is discounted at ``r``; the cash-only component at
the *risky* rate ``r + r_c``. This is the defining feature of the TF
decoupling of market risk from credit risk.

Free-boundary (American) constraints applied at every time level after
the linear step (projected scheme):

    Conversion :  V >= kappa * S            (holder converts)
    Call       :  V <= max(B_call, kappa*S) (issuer calls; holder may
                                             still force conversion)
    Put        :  V >= B_put                (holder puts at put price)

where ``kappa`` is the conversion ratio. Whenever a constraint binds in
favour of conversion, the cash-only component is reset: B = 0 on the
conversion region (value is received in shares, not cash).

Numerical scheme
----------------
Crank-Nicolson finite differences on a uniform mesh. Writing the spatial
discretisation of the TF differential operator as tridiagonal matrix
``A`` (sub/main/super diagonals from central differences), each backward
time step solves

    (I - (dt/2) A) u^{m} = (I + (dt/2) A) u^{m+1} + source terms

for u in {V, B}, via ``scipy.linalg.solve_banded`` (Thomas algorithm,
O(N) per step) or ``scipy.sparse.linalg.splu`` on a
``scipy.sparse.diags`` operator. The scheme is second-order accurate in
both dS and dt and unconditionally stable for the linear step.

Boundary conditions (see ``TsiveriotisFernandesSolver`` docstring for
the full statement):
    1. S -> 0        : V = B = risky PV of remaining cash flows,
                       floored at the default recovery value R * F.
    2. S -> infinity : V = kappa * S (immediate conversion), B = 0.
    3. t = T         : V = max(F + c_T, kappa * S) with the cash-only
                       component B = (F + c_T) on the redemption region
                       and B = 0 on the conversion region.

References
----------
Tsiveriotis, K. and C. Fernandes (1998), "Valuing Convertible Bonds
with Credit Risk", Journal of Fixed Income 8(2), 95-102.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_banded

FloatArray = NDArray[np.float64]

__all__ = [
    "ConvertibleBondSpec",
    "MarketParams",
    "FiniteDifferenceGrid",
    "ValuationResult",
    "TsiveriotisFernandesSolver",
]


# --------------------------------------------------------------------------- #
# Contract / market data objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ConvertibleBondSpec:
    """Static contractual terms of a convertible bond.

    Attributes
    ----------
    face_value:
        Redemption par amount ``F`` (e.g. 1000.0).
    coupon_rate:
        Annual coupon rate as a decimal (0.025 = 2.5%). Coupons of size
        ``coupon_rate * face_value / coupon_frequency`` are paid at
        equally spaced dates ending at maturity.
    coupon_frequency:
        Coupon payments per year (2 = semi-annual). Zero denotes a
        zero-coupon convertible.
    conversion_ratio:
        Shares received per bond upon conversion, ``kappa``. The implied
        conversion price is ``face_value / conversion_ratio``.
    maturity:
        Time to maturity ``T`` in years.
    credit_spread:
        Issuer credit spread ``r_c`` (continuously compounded, decimal)
        applied when discounting cash-only flows in TF-2.
    recovery_rate:
        Fraction ``R`` of face value recovered on default; sets the
        S -> 0 floor ``R * F``.
    call_price:
        Optional issuer (soft/hard) call price. ``None`` = non-callable.
    put_price:
        Optional holder put price. ``None`` = non-puttable.
    """

    face_value: float
    coupon_rate: float
    coupon_frequency: int
    conversion_ratio: float
    maturity: float
    credit_spread: float
    recovery_rate: float
    call_price: Optional[float] = None
    put_price: Optional[float] = None

    def __post_init__(self) -> None:
        if self.face_value <= 0.0:
            raise ValueError("face_value must be positive")
        if self.maturity <= 0.0:
            raise ValueError("maturity must be positive")
        if self.conversion_ratio < 0.0:
            raise ValueError("conversion_ratio must be non-negative")
        if not 0.0 <= self.recovery_rate <= 1.0:
            raise ValueError("recovery_rate must lie in [0, 1]")
        if self.coupon_rate < 0.0 or self.coupon_frequency < 0:
            raise ValueError("coupon terms must be non-negative")

    @property
    def conversion_price(self) -> float:
        """Stock price at which conversion value equals par: F / kappa."""
        if self.conversion_ratio == 0.0:
            return math.inf
        return self.face_value / self.conversion_ratio

    @property
    def coupon_amount(self) -> float:
        """Cash amount of a single coupon payment."""
        if self.coupon_frequency == 0:
            return 0.0
        return self.coupon_rate * self.face_value / self.coupon_frequency

    def remaining_coupon_times(self, tau: float) -> FloatArray:
        """Times-from-now (in years) of coupons paid within the next ``tau`` years.

        Coupon dates are anchored to maturity: T, T - 1/f, T - 2/f, ...
        Remaining times in [0, tau] are returned — cum-coupon convention:
        a coupon falling exactly on the valuation instant is included, so
        the S -> 0 boundary value is continuous with the terminal
        condition V(0, T) = F + c_T.
        """
        if self.coupon_frequency == 0 or self.coupon_amount == 0.0:
            return np.empty(0, dtype=np.float64)
        step: Final[float] = 1.0 / self.coupon_frequency
        # Payment times measured backward from valuation date t = T - tau.
        times = tau - np.arange(0.0, self.maturity, step)
        times = times[(times > -1e-12) & (times <= tau + 1e-12)]
        times = np.clip(times, 0.0, None)
        return np.sort(times).astype(np.float64)


@dataclass(frozen=True)
class MarketParams:
    """Market environment for the diffusion in TF-1 / TF-2.

    Attributes
    ----------
    risk_free_rate:
        Continuously-compounded risk-free rate ``r``.
    volatility:
        Lognormal stock volatility ``sigma`` (annualised).
    dividend_yield:
        Continuous dividend yield ``q``.
    """

    risk_free_rate: float
    volatility: float
    dividend_yield: float = 0.0

    def __post_init__(self) -> None:
        if self.volatility <= 0.0:
            raise ValueError("volatility must be positive")


# --------------------------------------------------------------------------- #
# Discretisation
# --------------------------------------------------------------------------- #
class FiniteDifferenceGrid:
    """Uniform rectangular mesh for the (S, t) domain of the TF system.

    Spatial nodes:  S_i = i * dS,  i = 0..n_space,  dS = s_max / n_space
    Temporal nodes: t_m = m * dt,  m = 0..n_time,   dt = T / n_time

    ``s_max`` should be chosen several multiples of the conversion price
    so that the S -> infinity asymptotic boundary condition
    (V = kappa * S) is accurate at the truncation edge; the classmethod
    :meth:`for_bond` applies a standard coverage heuristic.

    The mesh is deliberately uniform: Crank-Nicolson retains second-order
    accuracy and the tridiagonal operator has constant bandwidth, which
    keeps the banded solve O(n_space) per time step.
    """

    def __init__(self, s_max: float, n_space: int, n_time: int, maturity: float) -> None:
        if s_max <= 0.0:
            raise ValueError("s_max must be positive")
        if n_space < 3 or n_time < 1:
            raise ValueError("grid must have n_space >= 3 and n_time >= 1")
        if maturity <= 0.0:
            raise ValueError("maturity must be positive")

        self.s_max: Final[float] = float(s_max)
        self.n_space: Final[int] = int(n_space)
        self.n_time: Final[int] = int(n_time)
        self.maturity: Final[float] = float(maturity)

        self.ds: Final[float] = self.s_max / self.n_space
        self.dt: Final[float] = self.maturity / self.n_time

        self._s_nodes: FloatArray = np.linspace(0.0, self.s_max, self.n_space + 1)
        self._t_nodes: FloatArray = np.linspace(0.0, self.maturity, self.n_time + 1)

    @classmethod
    def for_bond(
        cls,
        bond: ConvertibleBondSpec,
        n_space: int = 400,
        n_time: int = 400,
        coverage: float = 4.0,
    ) -> "FiniteDifferenceGrid":
        """Build a grid whose S-extent covers ``coverage`` x conversion price.

        For a non-convertible bond (kappa = 0) the extent falls back to
        ``coverage * face_value`` so the domain remains finite.
        """
        anchor = bond.conversion_price if math.isfinite(bond.conversion_price) else bond.face_value
        return cls(s_max=coverage * anchor, n_space=n_space, n_time=n_time, maturity=bond.maturity)

    @property
    def s_nodes(self) -> FloatArray:
        """Spatial nodes S_0 = 0, ..., S_{n_space} = s_max (read-only view)."""
        v = self._s_nodes.view()
        v.flags.writeable = False
        return v

    @property
    def t_nodes(self) -> FloatArray:
        """Temporal nodes t_0 = 0, ..., t_{n_time} = T (read-only view)."""
        v = self._t_nodes.view()
        v.flags.writeable = False
        return v

    def nearest_spot_index(self, spot: float) -> int:
        """Index of the spatial node closest to ``spot`` (clipped to grid)."""
        i = int(round(spot / self.ds))
        return min(max(i, 0), self.n_space)


# --------------------------------------------------------------------------- #
# Results container
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ValuationResult:
    """Full solution surfaces of the TF system at t = 0.

    Attributes
    ----------
    grid:
        The mesh the solution lives on.
    total_value:
        V(S, 0) sampled on ``grid.s_nodes``.
    cash_component:
        B(S, 0) sampled on ``grid.s_nodes``.
    time_slices_total / time_slices_cash:
        Optional (n_time + 1, n_space + 1) surfaces V(S, t_m) and
        B(S, t_m) for every time node, populated only when
        ``solve(keep_time_slices=True)`` — the interactive grid
        visualizer scrubs across these. ``None`` on a standard solve to
        keep the result lightweight.
    """

    grid: FiniteDifferenceGrid
    total_value: FloatArray
    cash_component: FloatArray
    time_slices_total: Optional[FloatArray] = None
    time_slices_cash: Optional[FloatArray] = None

    @property
    def equity_component(self) -> FloatArray:
        """C(S, 0) = V - B, the credit-risk-free equity component."""
        return self.total_value - self.cash_component

    def price(self, spot: float) -> float:
        """Linear interpolation of V(., 0) at the given spot."""
        return float(np.interp(spot, self.grid.s_nodes, self.total_value))


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
class TsiveriotisFernandesSolver:
    """Crank-Nicolson solver for the coupled Tsiveriotis-Fernandes system.

    Boundary conditions
    -------------------
    1. **Lower boundary, S -> 0** (equity worthless, issuer near
       default). The equity component vanishes, C(0, t) = 0, so the bond
       degenerates to pure credit-risky debt:

           V(0, t) = B(0, t)
                   = max( R * F ,
                          F e^{-(r + r_c)(T - t)}
                          + sum_j c e^{-(r + r_c)(t_j - t)} )

       i.e. the risky present value of all remaining cash flows
       (coupons c at dates t_j plus face F at T), discounted at the
       risky rate r + r_c, floored at the default recovery value R * F.
       The floor encodes that even in default the holder recovers
       R * F immediately — the bond can never price below it.

    2. **Upper boundary, S -> infinity** (conversion certain). The bond
       behaves as kappa shares; credit risk is irrelevant because the
       issuer delivers stock:

           V(S_max, t) = kappa * S_max,      B(S_max, t) = 0.

       (With a continuous dividend yield q the asymptote is applied at
       the truncated edge S_max; grid coverage of ~4x the conversion
       price keeps the truncation error negligible.)

    3. **Terminal condition, t = T**. The holder takes the better of
       redemption and conversion:

           V(S, T) = max( F + c_T , kappa * S )

           B(S, T) = F + c_T   if  F + c_T >= kappa * S   (redeem: cash)
                   = 0         otherwise                  (convert: shares)

       where c_T is the final coupon paid at maturity (0 for a
       zero-coupon convertible).

    Algorithm (implemented in :meth:`solve`)
    ----------------------------------------
    March backward from t = T to t = 0. At each step:
      a. assemble the tridiagonal Crank-Nicolson operators for TF-1
         and TF-2 (see :meth:`_build_operator_diagonals`);
      b. solve the two banded systems (``scipy.linalg.solve_banded``);
      c. add coupon jumps B += c, V += c across coupon dates;
      d. project onto the American constraint set
         (:meth:`_apply_exercise_constraints`): conversion floor
         V >= kappa*S with B := 0 where conversion binds, call cap,
         put floor;
      e. overwrite the boundary rows with conditions (1) and (2).
    """

    def __init__(
        self,
        grid: FiniteDifferenceGrid,
        bond: ConvertibleBondSpec,
        market: MarketParams,
    ) -> None:
        if not math.isclose(grid.maturity, bond.maturity, rel_tol=1e-9):
            raise ValueError("grid maturity must match bond maturity")
        self.grid = grid
        self.bond = bond
        self.market = market

    # ------------------------------------------------------------------ #
    # Boundary conditions — the mathematical contract locked down by the
    # synthetic test harness; solve() consumes these as Dirichlet data.
    # ------------------------------------------------------------------ #
    def terminal_payoff(self) -> tuple[FloatArray, FloatArray]:
        """Terminal condition (3): payoff surfaces (V_T, B_T) at t = T.

        Returns
        -------
        (V_T, B_T):
            ``V_T[i] = max(F + c_T, kappa * S_i)`` and ``B_T`` equal to
            the redemption cash flow where redemption is optimal, zero
            on the conversion region.
        """
        s = self.grid.s_nodes
        redemption = self.bond.face_value + self.bond.coupon_amount
        conversion = self.bond.conversion_ratio * s
        v_t = np.maximum(redemption, conversion)
        b_t = np.where(redemption >= conversion, redemption, 0.0)
        return v_t.astype(np.float64), b_t.astype(np.float64)

    def lower_boundary_value(self, t: float) -> float:
        """Lower boundary (1): V(0, t) = B(0, t) as pure risky debt.

        Risky PV of face plus remaining coupons at rate r + r_c, floored
        at the recovery value R * F.
        """
        tau = self.grid.maturity - t
        risky_rate = self.market.risk_free_rate + self.bond.credit_spread
        pv = self.bond.face_value * math.exp(-risky_rate * tau)
        coupon_times = self.bond.remaining_coupon_times(tau)
        if coupon_times.size:
            pv += float(
                self.bond.coupon_amount * np.exp(-risky_rate * coupon_times).sum()
            )
        recovery_floor = self.bond.recovery_rate * self.bond.face_value
        return max(pv, recovery_floor)

    def upper_boundary_value(self, t: float) -> tuple[float, float]:
        """Upper boundary (2): (V, B) at S = s_max.

        With conversion rights and standard grid coverage, conversion is
        certain at the truncation edge: V = kappa * s_max, and the
        cash-only component is extinguished, B = 0.

        Degenerate case: if the conversion claim does not dominate the
        debt claim at the edge (e.g. kappa = 0, a straight bond), the
        value is S-independent there and the edge prices as pure risky
        debt, V = B = lower-boundary value. This keeps the Dirichlet
        data consistent with the flat (S-independent) exact solution of
        TF-2, which the straight-bond limit test exploits.
        """
        conversion = self.bond.conversion_ratio * self.grid.s_max
        debt = self.lower_boundary_value(t)
        if conversion >= debt:
            return conversion, 0.0
        return debt, debt

    # ------------------------------------------------------------------ #
    # Interior scheme
    # ------------------------------------------------------------------ #
    def _build_operator_diagonals(
        self, credit_adjusted: bool
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Sub/main/super diagonals of the spatial TF operator ``A``.

        For interior node i with S_i = i * dS, central differences give

            a_i = 0.5 * (sigma^2 i^2 - (r - q) i)          (sub)
            b_i = -(sigma^2 i^2 + rho)                     (main)
            c_i = 0.5 * (sigma^2 i^2 + (r - q) i)          (super)

        with discount load ``rho = r`` for the equity-rate equation and
        ``rho = r + r_c`` when ``credit_adjusted`` (TF-2 / the B-coupling
        term of TF-1). These feed the Crank-Nicolson pencil
        (I -+ dt/2 A) consumed by ``scipy.linalg.solve_banded``.

        Note the uniform-mesh simplification: S_i / dS = i, so the
        diagonals depend only on the node index, not on dS itself.
        Diagonal dominance of (I - dt/2 A) holds for all i because
        sigma^2 i^2 >= |(r - q)| i on the region where the convection
        term could flip the sub-diagonal sign only for i <
        (r - q)/sigma^2, where the diffusion term already dominates.
        """
        sigma = self.market.volatility
        drift = self.market.risk_free_rate - self.market.dividend_yield
        rho = self.market.risk_free_rate + (
            self.bond.credit_spread if credit_adjusted else 0.0
        )
        i = np.arange(1, self.grid.n_space, dtype=np.float64)
        sig2i2 = (sigma * i) ** 2
        sub = 0.5 * (sig2i2 - drift * i)
        main = -(sig2i2 + rho)
        sup = 0.5 * (sig2i2 + drift * i)
        return sub, main, sup

    def _apply_exercise_constraints(
        self, v: FloatArray, b: FloatArray, t: float
    ) -> tuple[FloatArray, FloatArray]:
        """Project (V, B) onto the American constraint set at time t.

        Conversion:  V := max(V, kappa * S); where the floor binds set
                     B := 0 (value delivered in shares).
        Call:        V := min(V, max(call_price, kappa * S)) when callable.
        Put:         V := max(V, put_price) when puttable; where the put
                     binds, B := put_price (cash settlement).

        Returns projected copies; inputs are not mutated. B is finally
        clipped into [0, V] so the decomposition V = B + C stays
        arbitrage-consistent after any combination of projections.
        """
        s = self.grid.s_nodes
        conversion = self.bond.conversion_ratio * s
        convert = conversion > v
        v = np.where(convert, conversion, v)
        b = np.where(convert, 0.0, b)
        if self.bond.call_price is not None:
            cap = np.maximum(self.bond.call_price, conversion)
            v = np.minimum(v, cap)
        if self.bond.put_price is not None:
            putted = v < self.bond.put_price
            v = np.where(putted, self.bond.put_price, v)
            b = np.where(putted, self.bond.put_price, b)
        b = np.clip(b, 0.0, v)
        return v.astype(np.float64), b.astype(np.float64)

    def solve(self, keep_time_slices: bool = False) -> ValuationResult:
        """Run the backward Crank-Nicolson march and return t = 0 surfaces.

        Parameters
        ----------
        keep_time_slices:
            When True, retain the full (n_time + 1, n_space + 1) V and B
            surfaces at every time node on the result (for the
            interactive time-scrubber visualizer). Default False.

        Step sequence per time level m (marching T -> 0):
          a. solve the TF-2 banded system for B (risky discounting);
          b. solve the TF-1 banded system for V (risk-free discounting)
             with the coupling source term  -r_c * B  time-centred as
             -(dt/2) r_c (B^{m} + B^{m+1});
          c. Dirichlet data during the linear step uses *ex-coupon*
             boundary values; the coupon jump c is then added to every
             node (cum-coupon storage convention), which lands the
             S = 0 node exactly on ``lower_boundary_value`` — that
             function is cum-coupon by construction;
          d. project onto the American constraint set;
          e. the projected surfaces feed the next step.

        Satisfies the invariants enforced by tests/test_synthetic_harness.py:
        recovery floor at S = 0, conversion asymptote at S_max, and the
        no-arbitrage sandwich V >= max(straight debt, conversion value).
        """
        grid = self.grid
        n = grid.n_space
        dt = grid.dt
        r_c = self.bond.credit_spread
        coupon = self.bond.coupon_amount

        sub_v, main_v, sup_v = self._build_operator_diagonals(credit_adjusted=False)
        sub_b, main_b, sup_b = self._build_operator_diagonals(credit_adjusted=True)

        def implicit_banded(sub: FloatArray, main: FloatArray, sup: FloatArray) -> FloatArray:
            """(I - dt/2 A) in solve_banded's (3, n-1) ab layout."""
            ab = np.zeros((3, n - 1), dtype=np.float64)
            ab[0, 1:] = -0.5 * dt * sup[:-1]
            ab[1, :] = 1.0 - 0.5 * dt * main
            ab[2, :-1] = -0.5 * dt * sub[1:]
            return ab

        ab_v = implicit_banded(sub_v, main_v, sup_v)
        ab_b = implicit_banded(sub_b, main_b, sup_b)

        def explicit_rhs(
            u: FloatArray, sub: FloatArray, main: FloatArray, sup: FloatArray
        ) -> FloatArray:
            """(I + dt/2 A) u^{m+1} on interior rows; the boundary columns
            of u^{m+1} enter here, so no separate explicit correction."""
            return (
                (1.0 + 0.5 * dt * main) * u[1:n]
                + 0.5 * dt * sub * u[0 : n - 1]
                + 0.5 * dt * sup * u[2 : n + 1]
            )

        # Interior coupon dates snapped to the nearest time index (the
        # coupon at T itself lives inside the terminal payoff).
        coupon_steps: set[int] = set()
        if self.bond.coupon_frequency > 0 and coupon > 0.0:
            k = 1
            t_c = self.bond.maturity - k / self.bond.coupon_frequency
            while t_c > 1e-12:
                coupon_steps.add(int(round(t_c / dt)))
                k += 1
                t_c = self.bond.maturity - k / self.bond.coupon_frequency

        v, b = self.terminal_payoff()
        slices_v: Optional[FloatArray] = None
        slices_b: Optional[FloatArray] = None
        if keep_time_slices:
            slices_v = np.empty((grid.n_time + 1, n + 1), dtype=np.float64)
            slices_b = np.empty_like(slices_v)
            slices_v[grid.n_time] = v
            slices_b[grid.n_time] = b
        for m in range(grid.n_time - 1, -1, -1):
            t_m = float(grid.t_nodes[m])
            coup = coupon if m in coupon_steps else 0.0

            # Ex-coupon Dirichlet data for the diffusion step: the cum
            # boundary functions include a coupon paid exactly at t_m,
            # which the jump in step (c) re-adds to every node.
            ex_lower = self.lower_boundary_value(t_m) - coup
            edge_conversion = self.bond.conversion_ratio * grid.s_max
            if edge_conversion >= ex_lower:
                ex_upper_v, ex_upper_b = edge_conversion, 0.0
            else:
                ex_upper_v, ex_upper_b = ex_lower, ex_lower

            rhs_b = explicit_rhs(b, sub_b, main_b, sup_b)
            rhs_b[0] += 0.5 * dt * sub_b[0] * ex_lower
            rhs_b[-1] += 0.5 * dt * sup_b[-1] * ex_upper_b
            b_int = solve_banded((1, 1), ab_b, rhs_b)

            rhs_v = explicit_rhs(v, sub_v, main_v, sup_v)
            rhs_v[0] += 0.5 * dt * sub_v[0] * ex_lower
            rhs_v[-1] += 0.5 * dt * sup_v[-1] * ex_upper_v
            rhs_v -= 0.5 * dt * r_c * (b[1:n] + b_int)
            v_int = solve_banded((1, 1), ab_v, rhs_v)

            v = np.concatenate(([ex_lower], v_int, [ex_upper_v]))
            b = np.concatenate(([ex_lower], b_int, [ex_upper_b]))
            if coup:
                v = v + coup
                b = b + coup

            v, b = self._apply_exercise_constraints(v, b, t_m)
            if slices_v is not None and slices_b is not None:
                slices_v[m] = v
                slices_b[m] = b

        return ValuationResult(
            grid=grid,
            total_value=v,
            cash_component=b,
            time_slices_total=slices_v,
            time_slices_cash=slices_b,
        )

    def price(self, spot: float) -> float:
        """Convenience wrapper: solve() then interpolate V(spot, 0)."""
        return self.solve().price(spot)
