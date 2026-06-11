#!/usr/bin/env python3
"""Read results_T*.csv and plot:
  1. Convergence heatmap per phase (one PNG each)
  2. Success rate bar chart across phases by optimizer
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

RESULTS_DIR = Path(__file__).parent / "results"
PHASES = ["T0", "T1", "T2", "T4"]
OPTIMIZERS = ["Structure", "CMA-ES", "DE", "L-BFGS-B", "least_squares"]
OPTIMIZER_LABELS = {
    "Structure": "Structure",
    "CMA-ES": "CMA-ES",
    "DE": "DE",
    "L-BFGS-B": "L-BFGS-B",
    "least_squares": "Least squares",
}

NMSE_GREEN = 1e-10
NMSE_YELLOW = 1e-2

TRANSFORM_TITLES = {
    "T0":  "T0 – Original (baseline)",
    "T1a": "T1a – Zero Term (sin·cos)",
    "T1b": "T1b – Zero Term (exp·cos)",
    "T2a": "T2a – Feature Power (var → var**c)",
    "T2b": "T2b – Expression Power ((expr)**c)",
    "T4a": "T4a – Rational Zero Term",
    "T4b": "T4b – Rational Zero Term (cross-feature)",
}
PHASE_LABELS = {
    "T0": "Original\nexpression",
    "T1": "Composite\nzero terms",
    "T2": "Power\naugmentation",
    "T4": "Rational\nzero terms",
}

COLOR_GREEN  = "#2ecc71"
COLOR_YELLOW = "#f1c40f"
COLOR_RED    = "#e74c3c"

NMSE_CONVERGED = 1e-10


# ====================================================================
# Heatmap
# ====================================================================

def _nmse_color(nmse: float) -> str:
    if np.isnan(nmse) or np.isinf(nmse) or nmse >= NMSE_YELLOW:
        return COLOR_RED
    if nmse < NMSE_GREEN:
        return COLOR_GREEN
    return COLOR_YELLOW


def _fmt_nmse(nmse: float) -> str:
    if np.isnan(nmse) or np.isinf(nmse):
        return "inf"
    if nmse == 0:
        return "0.00"
    if abs(nmse) < 1e-300:
        return "~0"
    exp = int(np.floor(np.log10(abs(nmse))))
    mantissa = nmse / 10**exp
    if exp == 0:
        return f"{nmse:.2f}"
    return f"{mantissa:.1f}e{exp}"


def _fmt_time(seconds: float) -> str:
    if np.isnan(seconds) or np.isinf(seconds):
        return ""
    if seconds < 0.1:
        return f"{seconds*1000:.0f}ms"
    if seconds < 10:
        return f"{seconds:.1f}s"
    return f"{seconds:.0f}s"


def _draw_heatmap(ax, pivot_nmse: pd.DataFrame, pivot_time: pd.DataFrame,
                  title: str):
    """pivot_nmse / pivot_time: index=equation_name, columns=optimizer."""
    equations = list(pivot_nmse.index)
    optimizers = list(pivot_nmse.columns)
    n_eq = len(equations)
    n_opt = len(optimizers)

    best_per_eq = pivot_nmse.min(axis=1)

    for i, eq in enumerate(equations):
        for j, opt in enumerate(optimizers):
            nmse = pivot_nmse.loc[eq, opt]
            color = _nmse_color(nmse)
            rect = plt.Rectangle((j, n_eq - 1 - i), 1, 1,
                                 facecolor=color, edgecolor="white", linewidth=0.5)
            ax.add_patch(rect)

            is_best = (not np.isnan(nmse) and not np.isinf(nmse)
                       and abs(nmse - best_per_eq[eq]) < 1e-30)
            if is_best:
                best_rect = plt.Rectangle((j + 0.02, n_eq - 1 - i + 0.02),
                                          0.96, 0.96, facecolor="none",
                                          edgecolor="#d4a017", linewidth=2)
                ax.add_patch(best_rect)

            txt_color = "black" if color == COLOR_YELLOW else "white"
            time_val = pivot_time.loc[eq, opt] if opt in pivot_time.columns else np.nan
            ax.text(j + 0.5, n_eq - 0.32 - i, _fmt_nmse(nmse),
                    ha="center", va="center", fontsize=5.5,
                    color=txt_color, fontweight="bold")
            ax.text(j + 0.5, n_eq - 0.72 - i, _fmt_time(time_val),
                    ha="center", va="center", fontsize=4.5,
                    color=txt_color, alpha=0.8)

    ax.set_xlim(0, n_opt)
    ax.set_ylim(0, n_eq)
    ax.set_xticks([x + 0.5 for x in range(n_opt)])
    ax.set_xticklabels(optimizers, fontsize=8, fontweight="bold")
    ax.set_yticks([n_eq - 0.5 - i for i in range(n_eq)])
    ax.set_yticklabels(equations, fontsize=6)
    ax.tick_params(length=0)

    converged = (pivot_nmse < NMSE_GREEN).sum()
    close = ((pivot_nmse >= NMSE_GREEN) & (pivot_nmse < NMSE_YELLOW)).sum()
    median_time = pivot_time.median()
    for j, opt in enumerate(optimizers):
        ax.text(j + 0.5, n_eq + 0.55, f"{converged[opt]}/{n_eq}",
                ha="center", va="bottom", fontsize=7, fontweight="bold",
                color="#27ae60")
        ax.text(j + 0.5, n_eq + 0.25, f"(+{close[opt]} close)",
                ha="center", va="bottom", fontsize=5.5, color="#b8860b")
        mt = median_time[opt] if opt in median_time.index else np.nan
        ax.text(j + 0.5, n_eq - 0.02, f"med {_fmt_time(mt)}",
                ha="center", va="bottom", fontsize=5, color="#555555")

    ax.set_title(title, fontsize=10, fontweight="bold", pad=35)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _eq_sort_key(name: str):
    alpha = "".join(c for c in name if c.isalpha())
    digits = "".join(c for c in name if c.isdigit())
    return (alpha, int(digits) if digits else 0)


def plot_heatmap(phase: str, df_phase: pd.DataFrame, out_dir: Path):
    transforms = [phase] if phase == "T0" else [f"{phase}a", f"{phase}b"]
    transforms = [t for t in transforms if t in df_phase["transform"].unique()]
    if not transforms:
        return

    n_sub = len(transforms)
    n_equations = df_phase["equation_name"].nunique()
    fig_width = 7.5 * n_sub
    fig_height = max(8, n_equations * 0.42 + 2.5)

    fig, axes = plt.subplots(1, n_sub, figsize=(fig_width, fig_height))
    if n_sub == 1:
        axes = [axes]

    fig.suptitle(
        f"{phase} Convergence Heatmap  ({n_equations} equations × {len(OPTIMIZERS)} optimizers)\n"
        "Green=converged (NMSE<1e-6)  Yellow=close (NMSE<1e-2)  Red=fail\n"
        "Gold border = best NMSE per equation  |  each cell: NMSE + solve time",
        fontsize=11, fontweight="bold", y=0.98,
    )

    for ax, tfm in zip(axes, transforms):
        sub_df = df_phase[df_phase["transform"] == tfm].copy()
        opt_order = pd.CategoricalDtype(categories=OPTIMIZERS, ordered=True)
        sub_df["optimizer"] = sub_df["optimizer"].astype(opt_order)

        pivot_nmse = sub_df.pivot_table(
            index="equation_name", columns="optimizer",
            values="final_nmse", aggfunc="first", observed=True,
        )
        pivot_time = sub_df.pivot_table(
            index="equation_name", columns="optimizer",
            values="time_s", aggfunc="first", observed=True,
        )
        eq_order = sorted(pivot_nmse.index, key=_eq_sort_key)
        pivot_nmse = pivot_nmse.reindex(index=eq_order, columns=OPTIMIZERS)
        pivot_time = pivot_time.reindex(index=eq_order, columns=OPTIMIZERS)

        _draw_heatmap(ax, pivot_nmse, pivot_time,
                      TRANSFORM_TITLES.get(tfm, tfm))

    plt.subplots_adjust(top=0.90, bottom=0.03, left=0.08, right=0.98, wspace=0.25)
    out_path = out_dir / f"{phase}_convergence_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap -> {out_path}")


# ====================================================================
# Bar chart
# ====================================================================

def plot_bar(all_df: pd.DataFrame, out_dir: Path):
    all_df = all_df.copy()
    all_df["success"] = all_df["final_nmse"] < NMSE_CONVERGED

    rows = []
    for phase in PHASES:
        phase_transforms = [phase] if phase == "T0" else [f"{phase}a", f"{phase}b"]
        phase_df = all_df[all_df["transform"].isin(phase_transforms)]
        if phase_df.empty:
            continue
        for opt in OPTIMIZERS:
            opt_df = phase_df[phase_df["optimizer"] == opt]
            rate = opt_df["success"].sum() / len(opt_df) * 100 if len(opt_df) else 0.0
            med_t = opt_df["time_s"].median() if len(opt_df) else 0.0
            rows.append({"phase": phase,
                         "transformation": PHASE_LABELS.get(phase, phase),
                         "optimizer": opt,
                         "optimizer_label": OPTIMIZER_LABELS.get(opt, opt),
                         "success_rate": rate, "median_time": med_t})

    rates_df = pd.DataFrame(rows)
    phases_present = [p for p in PHASES if p in rates_df["phase"].values]
    phase_order = [PHASE_LABELS.get(p, p) for p in phases_present]

    optimizer_labels = [OPTIMIZER_LABELS.get(opt, opt) for opt in OPTIMIZERS]

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    sns.barplot(
        data=rates_df,
        x="transformation",
        y="success_rate",
        hue="optimizer_label",
        order=phase_order,
        hue_order=optimizer_labels,
        palette=palette,
        errorbar=None,
        edgecolor="white",
        linewidth=0.8,
        ax=ax,
    )

    for container, opt_label in zip(ax.containers, optimizer_labels):
        for bar, phase_label in zip(container, phase_order):
            row = rates_df[
                (rates_df["transformation"] == phase_label)
                & (rates_df["optimizer_label"] == opt_label)
            ]
            if row.empty:
                continue
            height = bar.get_height()
            med_t = float(row["median_time"].iloc[0])
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1.2,
                _fmt_time(med_t),
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color="#333333",
            )

    ax.set_xlabel("")
    ax.set_ylabel("Success rate (%)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 112)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.legend(
        title=None,
        ncol=len(optimizer_labels),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
        frameon=False,
        fontsize=12,
        handlelength=1.8,
        columnspacing=1.2,
    )
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.subplots_adjust(top=0.80, bottom=0.23, left=0.08, right=0.99)
    out_path = out_dir / "success_rate_by_phase.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Bar chart -> {out_path}")

    print(f"\n{'Transformation':>24s}", end="")
    for opt in OPTIMIZERS:
        print(f" {opt:>14s}", end="")
    print()
    print("-" * (24 + 15 * len(OPTIMIZERS)))
    for phase in phases_present:
        phase_rows = rates_df[rates_df["phase"] == phase]
        print(f"{PHASE_LABELS.get(phase, phase).replace(chr(10), ' '):>24s}", end="")
        for opt in OPTIMIZERS:
            row = phase_rows[phase_rows["optimizer"] == opt]
            val = row["success_rate"].values[0] if len(row) else 0.0
            print(f" {val:13.1f}%", end="")
        print()


# ====================================================================
# Main entry
# ====================================================================

def main():
    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    all_frames = []
    for phase in PHASES:
        csv_path = RESULTS_DIR / f"results_{phase}.csv"
        if not csv_path.exists():
            print(f"[SKIP] {csv_path.name} does not exist")
            continue
        df = pd.read_csv(csv_path)
        df["optimizer"] = df["optimizer"].replace({"Hybrid": "Structure"})
        print(f"[{phase}] {len(df)} records, "
              f"transforms={sorted(df['transform'].unique())}")
        plot_heatmap(phase, df, out_dir)
        all_frames.append(df)

    if all_frames:
        all_df = pd.concat(all_frames, ignore_index=True)
        print(f"\nTotal {len(all_df)} records")
        plot_bar(all_df, out_dir)

    print("\nAll plots completed.")


if __name__ == "__main__":
    main()
