"""Generate F0-F5 (PNG 300 dpi + PDF) in the manuscript house style (seaborn
whitegrid + deep palette; matplotlib-drawn architecture schematic).

Reads the result CSVs in results/ and the monitor logs for learning curves.
Each figure is skipped gracefully if its inputs are missing, so this is safe to
re-run as experiments complete.
"""

from __future__ import annotations

import glob
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import seaborn as sns
import matplotlib.pyplot as plt

# Shared house style, matched 1:1 to the companion manuscripts' figures:
# seaborn whitegrid with their exact custom palette (cBlue / cTeal / cAmber /
# cGreen / cRed) and slate-ink text, sans-serif.
C_BLUE = "#3B6FB6"
C_TEAL = "#2A9D8F"
C_AMBER = "#E08D2F"
C_SLATE = "#55617A"
C_GREEN = "#4C956C"
C_RED = "#D1495B"
C_INK = "#273043"
_PALETTE = [C_BLUE, C_AMBER, C_GREEN, C_RED, C_TEAL, C_SLATE]
sns.set_theme(style="whitegrid", context="paper", palette=_PALETTE)
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9,
        "text.color": C_INK,
        "axes.titlesize": 10,
        "axes.titleweight": "normal",
        "axes.titlecolor": C_INK,
        "axes.labelsize": 9.5,
        "axes.labelcolor": C_INK,
        "axes.edgecolor": "#B7BECC",
        "axes.linewidth": 0.8,
        "axes.prop_cycle": plt.cycler(color=_PALETTE),
        "grid.color": "#E3E6EC",
        "grid.linewidth": 0.8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.color": C_INK,
        "ytick.color": C_INK,
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#C7CCD6",
        "lines.linewidth": 1.6,
        "lines.markersize": 6,
        "lines.markeredgewidth": 0.6,
        "lines.markeredgecolor": "white",
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)
# Semantic handles used across panels.
C_DRL, C_HEUR, C_PURPLE = C_BLUE, "#8A93A6", C_SLATE

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
MONITOR = os.path.join(RESULTS, "monitor")
FIGS = os.path.join(ROOT, "figures")
os.makedirs(FIGS, exist_ok=True)


def _save(fig, name):
    fig.savefig(os.path.join(FIGS, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIGS, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png/.pdf", flush=True)


def _read_monitor(tag_prefix):
    """Return list of (timesteps_cumulative, episode_return) arrays per seed."""
    series = []
    for f in sorted(glob.glob(os.path.join(MONITOR, tag_prefix + "*.monitor.csv"))):
        try:
            df = pd.read_csv(f, skiprows=1)
            if "r" in df and "l" in df:
                series.append((np.cumsum(df["l"].values), df["r"].values))
        except Exception:
            continue
    return series


def _binned_mean_std(series, n_bins=60):
    """Resample each seed's return curve onto a common timestep grid, then
    return grid, mean, std across seeds."""
    if not series:
        return None
    max_t = min(s[0][-1] for s in series)
    grid = np.linspace(max_t * 0.02, max_t, n_bins)
    mat = []
    for ts, r in series:
        mat.append(np.interp(grid, ts, r))
    mat = np.array(mat)
    return grid, mat.mean(axis=0), mat.std(axis=0)


def f0_architecture():
    """Offloading decision loop as a matplotlib schematic, drawn in the same
    house style as the other manuscripts' architecture figures (the cBlue/
    cTeal/cAmber/cGreen palette, rounded boxes, slate-ink arrows, no axes)."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    # Nord-ish soft palette matched to the companion paper's house style.
    EDGE = "#2E3440"
    GREEN = "#CFE3D6"  # IoT device (light green)
    STEEL = "#5E81AC"  # orchestrator (filled blue)
    FROST = "#D7E4EC"  # edge server (light frost)
    GOLD = "#F4E2B8"  # local CPU (light amber)
    LILAC = "#E3D9E6"  # cloud (light mauve)

    def box(ax, xy, w, h, text, fc, fs=8.5, weight="normal", tc=EDGE):
        ax.add_patch(
            FancyBboxPatch(
                xy,
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                linewidth=1.2,
                facecolor=fc,
                edgecolor=EDGE,
            )
        )
        ax.text(
            xy[0] + w / 2,
            xy[1] + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fs,
            weight=weight,
            color=tc,
        )

    def arrow(ax, p1, p2, color=EDGE, style="-|>", ls="-", lw=1.2, rad=0.0):
        ax.add_patch(
            FancyArrowPatch(
                p1,
                p2,
                arrowstyle=style,
                mutation_scale=11,
                linewidth=lw,
                color=color,
                linestyle=ls,
                shrinkA=3,
                shrinkB=3,
                connectionstyle=f"arc3,rad={rad}",
            )
        )

    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    yc = 4.1  # vertical centre line of the main flow

    # ---- Boxes ----
    box(ax, (0.2, yc - 0.6), 2.2, 1.2, "IoT device\ntask state $s$", fc=GREEN)
    box(
        ax,
        (3.6, yc - 0.75),
        2.6,
        1.5,
        "Orchestrator\n$\\pi_\\theta:\\, s \\mapsto a$",
        fc=STEEL,
        fs=9.5,
        tc="white",
    )
    box(ax, (8.6, yc + 1.35), 3.0, 1.0, "Edge server\n$f_e$,  backlog $W_e$", fc=FROST)
    box(ax, (8.6, yc - 0.5), 3.0, 1.0, "Local CPU\n$f_\\ell$", fc=GOLD)
    box(ax, (8.6, yc - 2.25), 3.0, 1.0, "Cloud\n$f_c$,  WAN delay", fc=LILAC)

    # ---- Forward arrows ----
    arrow(ax, (2.4, yc), (3.6, yc))
    ax.text(3.0, yc + 0.32, "$s$", ha="center", va="bottom", fontsize=9.5)
    arrow(ax, (6.2, yc + 0.35), (8.6, yc + 1.6), rad=-0.12)
    arrow(ax, (6.2, yc), (8.6, yc))
    arrow(ax, (6.2, yc - 0.35), (8.6, yc - 1.55), rad=0.12)
    ax.text(7.4, yc + 1.18, "$a$", ha="center", va="center", fontsize=9.5)

    # ---- Feedback path (clean dashed loop along the bottom) ----
    fb = 1.05  # feedback baseline y
    fb_col = "#7B8494"
    arrow(ax, (10.1, yc - 2.25), (10.1, fb), color=fb_col, ls="--", lw=1.0)
    arrow(ax, (10.1, fb), (1.3, fb), color=fb_col, ls="--", lw=1.0)
    arrow(ax, (1.3, fb), (1.3, yc - 0.6), color=fb_col, ls="--", lw=1.0)
    ax.text(
        5.7,
        fb + 0.22,
        "latency $T$ and energy $E$ feedback",
        ha="center",
        va="bottom",
        fontsize=8,
        color=fb_col,
        style="italic",
    )

    # ---- Size-spectrum strip: clean caption-like line at the very bottom ----
    ax.text(
        6.0,
        0.12,
        "policy size spectrum:   sub-kB DQN ($10^2$ params, $\\mu$s/decision)"
        "   $\\longrightarrow$   GB LLM agent ($10^9$ params, s/decision)",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=STEEL,
        style="italic",
    )

    fig.tight_layout(pad=0.4)
    _save(fig, "F0_architecture")


def f1_learning_curves():
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    plotted = False
    for prefix, label, color in [
        ("DQN_64x64_w0.50_s", "DQN [64,64]", "C0"),
        ("PPO_64x64_w0.50_s", "PPO [64,64]", "C1"),
    ]:
        res = _binned_mean_std(_read_monitor(prefix))
        if res is None:
            continue
        g, m, s = res
        ax.plot(g, m, label=label, color=color)
        ax.fill_between(g, m - s, m + s, alpha=0.2, color=color)
        plotted = True
    if not plotted:
        return
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episode return")
    ax.set_title("Learning curves (mean $\\pm$ std over 5 seeds)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "F1_learning_curves")


def f2_cost_bars():
    p = os.path.join(RESULTS, "e1_results.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    order = [
        "AlwaysLocal",
        "AlwaysEdge",
        "Random",
        "Greedy",
        "DQN[256x256]",
        "DQN[64x64]",
        "DQN[16x16]",
        "PPO[64x64]",
    ]
    df = (
        df.set_index("method")
        .reindex([m for m in order if m in df["method"].values or m in df.index])
        .dropna(how="all")
    )
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(df))
    colors = [C_HEUR if k == "heuristic" else C_BLUE for k in df["kind"]]
    bars = ax.bar(
        x,
        df["cost_mean"],
        yerr=df["cost_std"],
        capsize=3,
        color=colors,
        edgecolor=C_INK,
        linewidth=0.6,
    )
    # Greedy reference line: the strong myopic baseline the learned policies
    # are trying to match.
    if "Greedy" in df.index:
        gcost = float(df.loc["Greedy", "cost_mean"])
        ax.axhline(gcost, color=C_RED, ls="--", lw=1.3, zorder=0)
        ax.text(
            len(df) - 0.4,
            gcost + 0.015,
            "Greedy",
            color=C_RED,
            fontsize=8,
            ha="right",
            va="bottom",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("x", "×") for m in df.index], rotation=30, ha="right")
    ax.set_ylabel("weighted cost")
    ax.set_title("Decision cost by method (lower is better)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=C_HEUR, ec=C_INK),
        plt.Rectangle((0, 0), 1, 1, color=C_BLUE, ec=C_INK),
    ]
    ax.legend(handles, ["heuristic", "learned (DRL)"], loc="upper right")
    _save(fig, "F2_cost_bars")


def f3_tradeoff():
    p = os.path.join(RESULTS, "e2_tradeoff.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for ax, zoom in [(axes[0], False), (axes[1], True)]:
        for method, mk, color in [("DQN[64,64]", "o-", "C0"), ("Greedy", "s--", "C1")]:
            d = df[df["method"] == method].sort_values("w1")
            if len(d):
                ax.plot(d["latency"], d["energy"], mk, color=color, label=method, ms=4)
                if zoom:
                    for _, r in d.iterrows():
                        ax.annotate(
                            f"{r['w1']:.1f}",
                            (r["latency"], r["energy"]),
                            fontsize=7,
                            textcoords="offset points",
                            xytext=(4, 3),
                        )
        if not zoom:
            for name, marker in [("AlwaysLocal", "^"), ("AlwaysEdge", "v")]:
                d = df[df["method"] == name]
                if len(d):
                    ax.scatter(
                        d["latency"],
                        d["energy"],
                        marker=marker,
                        s=70,
                        color="C2",
                        label=name,
                        zorder=5,
                    )
            ax.set_title("(a) full view")
            ax.legend(fontsize=8)
        else:
            dd = df[df["method"].isin(["DQN[64,64]", "Greedy"])]
            mx, Mx = dd["latency"].min(), dd["latency"].max()
            my, My = dd["energy"].min(), dd["energy"].max()
            ax.set_xlim(mx - 0.01, Mx + 0.015)
            ax.set_ylim(my - 0.005, My + 0.008)
            ax.set_title("(b) zoom on the sweep")
        ax.set_xlabel("mean latency (s)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("mean energy (J)")
    fig.suptitle("Latency\u2013energy tradeoff as $w_1$ is swept", y=1.02)
    _save(fig, "F3_tradeoff")


def f4_size_vs_cost():
    p1 = os.path.join(RESULTS, "e1_results.csv")
    if not os.path.exists(p1):
        return
    df = pd.read_csv(p1)
    drl = df[df["kind"] == "drl"].copy()
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    # DRL: size_kb in KB -> bytes for a KB..GB axis
    xs = drl["size_kb"].astype(float) * 1024.0
    ax.scatter(xs, drl["cost_mean"], s=60, color="C0", zorder=5)
    # Per-method label offsets so the lower-left cluster does not overlap.
    # DQN[64x64] (100 KB) and PPO[64x64] (141 KB) sit almost on top of each
    # other on the log axis, so fan their labels above/below; int8 (73 KB)
    # is pushed down and right of the DQN[16x16] tag.
    off = {
        "DQN[256x256]": (6, -12, "left"),
        "DQN[64x64]": (-2, 13, "right"),
        "DQN[16x16]": (-8, 2, "right"),
        "PPO[64x64]": (8, -16, "left"),
    }
    for _, r in drl.iterrows():
        dx, dy, ha = off.get(r["method"], (4, 4, "left"))
        ax.annotate(
            f"{r['method'].replace('x', '×')}\n{r['infer_us']:.0f}$\\mu$s",
            (float(r["size_kb"]) * 1024.0, r["cost_mean"]),
            fontsize=6.5,
            textcoords="offset points",
            xytext=(dx, dy),
            ha=ha,
        )
    # int8 point from E3
    p3 = os.path.join(RESULTS, "e3_lightweight.csv")
    if os.path.exists(p3):
        e3 = pd.read_csv(p3)
        q = e3[e3["variant"].str.contains("int8")]
        if len(q):
            r = q.iloc[0]
            ax.scatter(
                float(r["size_kb"]) * 1024.0,
                r["cost_mean"],
                s=60,
                marker="D",
                color="C3",
                zorder=5,
            )
            ax.annotate(
                "int8",
                (float(r["size_kb"]) * 1024.0, r["cost_mean"]),
                fontsize=6.5,
                textcoords="offset points",
                xytext=(2, -14),
                ha="center",
            )
    # LLM agents from E5
    p5s = os.path.join(RESULTS, "e5_llm_summary.csv")
    p5e = os.path.join(RESULTS, "e5_llm_episodes.csv")
    if os.path.exists(p5s) and os.path.exists(p5e):
        s = pd.read_csv(p5s)
        e = pd.read_csv(p5e)
        LLM_BYTES = 4.9e9  # llama3.1:8b q4 ~4.9 GB on disk
        costs = {}
        for _, sr in s.iterrows():
            ep = e[e["variant"] == sr["variant"]]
            if len(ep):
                cost = ep["cost"].mean()
                costs[sr["variant"]] = (cost, sr["decision_latency_s_mean"])
                ax.scatter(LLM_BYTES, cost, s=110, marker="*", color="C4", zorder=6)
        # Single combined label for the LLM cluster (both points sit at ~5 GB
        # with nearly equal cost), placed to the left to avoid overlap.
        if costs:
            ytop = max(c for c, _ in costs.values())
            lat = ", ".join(f"{v[1]:.2f}s" for v in costs.values())
            ax.annotate(
                "zero-shot LLM\n(P1, P2)\n$\\approx$5 GB, " + lat,
                (LLM_BYTES, ytop),
                fontsize=6.5,
                textcoords="offset points",
                xytext=(-10, 2),
                ha="right",
                va="center",
            )
    # Shade the device-deployable region (<= ~1 MB) to make the spectrum's
    # message visual: everything left of the band can live on the device.
    ax.axvspan(1e3, 1e6, color="C2", alpha=0.06, zorder=0)
    ax.set_xscale("log")
    ax.margins(x=0.12)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo - 0.02 * (yhi - ylo), yhi + 0.12 * (yhi - ylo))
    ax.set_xlabel("orchestrator on-disk size (bytes, log scale)")
    ax.set_ylabel("weighted cost")
    ax.set_title("Orchestrator size vs cost (KB policies $\\to$ GB LLM)")
    ax.grid(False)
    ax.grid(True, which="major", axis="y", alpha=0.3)
    ax.tick_params(which="minor", length=0)
    ax.set_axisbelow(True)
    _save(fig, "F4_size_vs_cost")


def f5_robustness():
    p = os.path.join(RESULTS, "e4_robustness.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), sharey=True)
    for ax, shift, title, panel in [
        (axes[0], "rate", "channel rate scale", "(a)"),
        (axes[1], "bgload", "background-load scale", "(b)"),
    ]:
        d = df[df["shift"] == shift]
        for method, mk, color in [("DQN[64,64]", "o-", "C0"), ("Greedy", "s--", "C1")]:
            dd = d[d["method"] == method].sort_values("scale")
            if len(dd):
                ax.errorbar(
                    dd["scale"],
                    dd["cost"],
                    yerr=dd.get("cost_std"),
                    fmt=mk,
                    color=color,
                    label=method,
                    capsize=3,
                )
        ax.axvline(1.0, color="0.6", ls=":", lw=0.8, zorder=0)
        ax.set_xlabel(title)
        ax.set_title(f"{panel} {title}")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("weighted cost")
    axes[1].legend(fontsize=8)
    fig.suptitle("Robustness to distribution shift (no retraining)")
    _save(fig, "F5_robustness")


def main():
    for fn in [
        f0_architecture,
        f1_learning_curves,
        f2_cost_bars,
        f3_tradeoff,
        f4_size_vs_cost,
        f5_robustness,
    ]:
        try:
            fn()
        except Exception as e:
            print(f"  [skip] {fn.__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
