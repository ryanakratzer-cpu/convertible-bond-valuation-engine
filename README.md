# Convertible Bond Valuation & TRACE Event-Study Engine

Quantitative companion to the convertible-debt research stream. Code lives here; theory lives in the vault's literature notes. Each module below links to the note that derives the math it implements — keep the two in sync: **the code cites the note, the note embeds the code's output.**

> [!success] Status: fully implemented & verified (2026-07-05)
> All method bodies are live: the Crank-Nicolson TF solver, the Dick-Nielsen TRACE cascade, the market-model event study, and the price-pressure regressions. The harness runs **22/22 strict tests green** (no xfails), and `run_pipeline_demo.py` exercises the entire loop end-to-end on synthetic data. See [[Convertible Engine Implementation Log]].

## Pillar 1 — Valuation: `models/`

| Module | Class | Theory note |
|---|---|---|
| `models/pde_solver.py` | `TsiveriotisFernandesSolver` | [[Tsiveriotis-Fernandes Framework]] |
| `models/pde_solver.py` | `FiniteDifferenceGrid` | [[Crank-Nicolson Schemes for Parabolic PDEs]] |
| `models/pde_solver.py` | `ConvertibleBondSpec` / `MarketParams` | [[Convertible Bond Contract Taxonomy]] |
| `models/visualizer.py` | `export_interactive_grid` | [[Tsiveriotis-Fernandes Framework]] (interactive grid explorer) |

The engine prices `V(S,t) = B(S,t) + C(S,t)`: the cash-only component `B` discounts at the risky rate `r + r_c`, the equity component `C` at the risk-free rate `r` — the TF decoupling of credit from market risk. The three boundary conditions (recovery floor at `S → 0`, conversion asymptote at `S → ∞`, `max(redemption, conversion)` at `T`) are stated in the solver docstring and **enforced by tests before any solver logic exists**.

## Pillar 2 — Empirics: `pipeline/` & `backtest/`

| Module | Class | Theory note |
|---|---|---|
| `pipeline/trace_cleaner.py` | `TraceDataCleaner` | [[TRACE Data Synthesis]], [[Dick-Nielsen Filter Cascade]] |
| `pipeline/event_study.py` | `EventStudyEngine` | [[Event Study Methodology - MacKinlay 1997]] |
| `backtest/price_pressure.py` | `PricePressureRegression` | [[Convertible Arbitrage Short-Selling Pressure]] |

Flow: raw TRACE prints → Dick-Nielsen filters → bond-day panel → market-model CARs on the `[-5, +5]` issuance window → cross-sectional regression of CAR on short-interest spikes ([[Price Pressure vs Information Effects]]). This connects directly to the reopening-liquidity results in [[Corporate Bond Reopenings]].

## Self-verification harness

```
py -m pytest tests/test_synthetic_harness.py -v
```

`tests/test_synthetic_harness.py` (22 tests, all strict) generates seeded synthetic GBM paths and multi-security market-model panels, then asserts the no-arbitrage contract: the `S = 0` value never drops below `R·F`, the truncation edge prices as pure conversion with `B = 0`, terminal payoff equals `max(F + c_T, κS)`, the solved surface dominates conversion value everywhere, the `κ = 0` limit reproduces the closed-form risky-debt PV within 0.5%, and the event engine recovers a known β from exactly-generated market-model returns. Method: [[Synthetic Test Harness Design]].

## Running the full pipeline

From the engine root:

```
py run_pipeline_demo.py
```

Stages: (1) TF valuation + `V = B + C` decomposition plot; (2) mock TRACE log with injected cancels/duplicates/fat-fingers → cleaned bond-day panel with YTM spreads; (3) `[-5, +5]` event study with injected arbitrage pressure → CAR plot; (4) `CAR ~ ΔSI + controls` regression (HC1) plus the reversal-window falsification. Exit code 0 = every stage verified. For real data, swap the synthetic generators for `TraceDataCleaner.ingest(...)` and your event file, keeping the same call sequence shown in the script.

## Interactive visualizers (Plotly, offline)

Alongside the static PNGs (kept for GitHub/PDF), the engine exports **self-contained interactive HTML** — plotly.js is inlined, so the files work fully offline and travel with the vault:

| File | Generator | What it does |
|---|---|---|
| `outputs/interactive_tf_grid.html` | `models/visualizer.py` → `export_interactive_grid(solver)` | Time-step scrubber over the FD mesh from t = T back to t = 0: watch V, B, C diffuse away from the terminal payoff kink. Hover any node for S, V/B/C, the straight-debt floor, and the option premium. |
| `outputs/interactive_car_study.html` | `EventStudyEngine.export_interactive_car(results, path)` | Zoomable mean-CAR path with ±2 SE bands; hover any event day for the daily AR, CAR, and cross-sectional t-stat. |

Embed either one inside an Obsidian markdown note with an iframe (the `src` is resolved **relative to the note's folder** — the snippets below are for a note sitting next to the engine root, e.g. this README; prefix the path accordingly from elsewhere in the vault):

```html
<iframe src="outputs/interactive_tf_grid.html" width="100%" height="650px" frameborder="0"></iframe>
```

```html
<iframe src="outputs/interactive_car_study.html" width="100%" height="550px" frameborder="0"></iframe>
```

Regenerate both at any time with `py run_pipeline_demo.py`, or call the exporters directly on real solved bonds / event studies.

## Embedding outputs in vault notes

All figures write to `outputs/` (matplotlib Agg backend — no display server needed):

```python
from pathlib import Path
OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
engine.plot_car(results, OUTPUTS / "car_minus5_plus5.png")
```

Then embed in any note with a standard image wikilink:

```
![[quant_research/convertible-bond-engine/outputs/car_minus5_plus5.png]]
```

Convention: filename = `<analysis>_<window-or-params>.png`, and every figure gets a one-line caption in the note linking back to the generating module, e.g. `Generated by pipeline/event_study.py — see [[Event Study Methodology - MacKinlay 1997]]`.

## Layout

```
convertible-bond-engine/
├── models/               # Pillar 1: TF PDE valuation
├── pipeline/             # Pillar 2: TRACE cleaning + event study
├── backtest/             # Pillar 2: price-pressure regressions
├── tests/                # synthetic self-verification harness (22 strict)
├── outputs/              # PNGs + self-contained interactive HTML (gitkeep'd)
└── run_pipeline_demo.py  # end-to-end synthetic validation of all 4 stages
```
