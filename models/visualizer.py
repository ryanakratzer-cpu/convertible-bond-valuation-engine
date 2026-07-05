"""Interactive Plotly visualizer for the TF finite-difference solution.

Produces a self-contained offline HTML file (plotly.js embedded, no CDN
or network needed) with a **time-step scrubber**: a slider over the
temporal mesh, initialised at maturity t = T on the terminal payoff
max(F + c_T, kappa * S) and scrubbing back to the valuation date t = 0,
so the user can watch the V = B + C decomposition diffuse away from the
payoff kink under the Crank-Nicolson march.

Hover tooltips on the total-value curve report, at every grid node:
the stock price S, V / B / C, the straight-debt floor (the S -> 0
risky-debt value at that time level), and the option premium
V - max(kappa * S, floor) — the value of optionality over both hard
floors.

Embed in an Obsidian note with:

    <iframe src="outputs/interactive_tf_grid.html"
            width="100%" height="650px" frameborder="0"></iframe>
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from models.pde_solver import TsiveriotisFernandesSolver

__all__ = ["export_interactive_grid"]

#: Traces added per time slice: V (rich hover), B, C.
_TRACES_PER_FRAME = 3


def export_interactive_grid(
    solver: TsiveriotisFernandesSolver,
    filepath: Path | str = "outputs/interactive_tf_grid.html",
    max_frames: int = 60,
) -> Path:
    """Solve the TF system retaining all time slices and export an
    interactive HTML grid explorer with a time-step slider.

    Parameters
    ----------
    solver:
        A configured solver; ``solve(keep_time_slices=True)`` is run
        internally, so the solver does not need to be pre-solved.
    filepath:
        Output HTML path; parent directories are created.
    max_frames:
        Cap on slider positions. The temporal mesh is subsampled evenly
        (endpoints t = 0 and t = T always included) so the exported file
        stays a few MB even on fine grids.

    Returns
    -------
    Path of the HTML file written (self-contained, plotly.js inlined).
    """
    result = solver.solve(keep_time_slices=True)
    assert result.time_slices_total is not None  # keep_time_slices=True
    assert result.time_slices_cash is not None
    grid, bond = solver.grid, solver.bond

    # Evenly subsampled time indices, ordered T -> 0 (slider scrubs
    # from the terminal payoff back to today).
    n_frames = min(max_frames, grid.n_time + 1)
    frame_idx = np.unique(np.round(np.linspace(0, grid.n_time, n_frames)).astype(int))[::-1]

    s = grid.s_nodes
    conversion = bond.conversion_ratio * s
    y_top = 1.05 * float(result.time_slices_total.max())

    fig = go.Figure()
    steps: list[dict] = []
    n_total_traces = _TRACES_PER_FRAME * len(frame_idx) + 1  # + static kappa*S line

    for k, m in enumerate(frame_idx):
        t_m = float(grid.t_nodes[m])
        v = result.time_slices_total[m]
        b = result.time_slices_cash[m]
        c = v - b
        floor = solver.lower_boundary_value(t_m)
        premium = v - np.maximum(conversion, floor)
        visible = k == 0

        fig.add_trace(
            go.Scatter(
                x=s, y=v, name="V total", visible=visible,
                line={"width": 3},
                customdata=np.column_stack([b, c, np.full_like(v, floor), premium]),
                hovertemplate=(
                    "S = %{x:.2f}<br>"
                    "V = %{y:,.2f}<br>"
                    "B (cash-only) = %{customdata[0]:,.2f}<br>"
                    "C (equity) = %{customdata[1]:,.2f}<br>"
                    "straight-debt floor = %{customdata[2]:,.2f}<br>"
                    "option premium = %{customdata[3]:,.2f}"
                    "<extra>V(S, t)</extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=s, y=b, name="B cash-only", visible=visible,
                line={"width": 1.5},
                hovertemplate="S = %{x:.2f}<br>B = %{y:,.2f}<extra>B(S, t)</extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=s, y=c, name="C equity", visible=visible,
                line={"width": 1.5},
                hovertemplate="S = %{x:.2f}<br>C = %{y:,.2f}<extra>C(S, t)</extra>",
            )
        )

        mask = [False] * n_total_traces
        base = _TRACES_PER_FRAME * k
        mask[base : base + _TRACES_PER_FRAME] = [True] * _TRACES_PER_FRAME
        mask[-1] = True  # conversion-value reference line always on
        steps.append(
            {
                "method": "update",
                "label": f"{t_m:.2f}",
                "args": [
                    {"visible": mask},
                    {"title": {
                        "text": (
                            "Tsiveriotis-Fernandes grid — valuation date "
                            f"t = {t_m:.2f}y of T = {grid.maturity:.2f}y"
                        )
                    }},
                ],
            }
        )

    fig.add_trace(
        go.Scatter(
            x=s, y=conversion, name="conversion value κS",
            line={"dash": "dot", "color": "grey", "width": 1},
            hovertemplate="S = %{x:.2f}<br>κS = %{y:,.2f}<extra>conversion</extra>",
        )
    )

    fig.update_layout(
        title={
            "text": (
                "Tsiveriotis-Fernandes grid — valuation date "
                f"t = {grid.maturity:.2f}y of T = {grid.maturity:.2f}y"
            )
        },
        xaxis={"title": "Stock price S", "range": [0.0, grid.s_max]},
        yaxis={"title": "Value", "range": [0.0, y_top]},
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "valuation date t (years) = "},
                "pad": {"t": 45},
                "steps": steps,
            }
        ],
        hovermode="x unified",
        template="plotly_white",
        legend={"orientation": "h", "y": 1.08},
    )

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(filepath, include_plotlyjs=True, full_html=True)
    return filepath
