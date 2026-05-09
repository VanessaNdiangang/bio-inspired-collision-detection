# generate_figures.py
#
# Reads from results/ and produces all paper figures as PDF and PNG.
#
# Usage:
#   python generate_figures.py
#   python generate_figures.py --results_dir results --output_dir figures

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path


# ── Style ──────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     "--",
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
})

BLUE   = "#2166AC"
RED    = "#D6604D"
GREEN  = "#4DAC26"
ORANGE = "#F4A736"
PURPLE = "#762A83"
GREY   = "#888888"

MODEL_COLOURS = {
    "LGMD2":      RED,
    "EMD":        GREEN,
    "Hybrid SNN": BLUE,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def save(fig, output_dir, name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(output_dir / f"{name}.{ext}")
    plt.close(fig)
    print(f"  Saved {name}.pdf / .png")


def per_stimulus_metrics(results_dir, label_csv, model_key, suffix):
    """
    Compute accuracy, precision, recall, F1 per stimulus type for one model.
    Joins prediction CSVs against ground truth on Filename + Frame.
    """
    gt       = pd.read_csv(label_csv)
    rows     = []

    for csv_path in sorted(Path(results_dir).glob(f"*{suffix}.csv")):
        video_name = csv_path.stem.replace(suffix, "") + ".mp4"
        video_gt   = gt[gt["Filename"] == video_name]
        if video_gt.empty:
            continue

        df     = pd.read_csv(csv_path)
        merged = df.merge(video_gt[["Frame", "TrueLabel", "StimulusType"]],
                          on="Frame", how="inner")
        if merged.empty:
            continue

        rows.append(merged)

    if not rows:
        return pd.DataFrame()

    combined = pd.concat(rows, ignore_index=True)
    records  = []

    for stype, grp in combined.groupby("StimulusType"):
        y_true = grp["TrueLabel"].values
        y_pred = grp["Prediction"].values
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        n         = tp + fp + fn + tn
        precision = tp / (tp + fp)  if (tp + fp) > 0           else 0.0
        recall    = tp / (tp + fn)  if (tp + fn) > 0           else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0               else 0.0)
        accuracy  = (tp + tn) / n   if n > 0                   else 0.0
        records.append(dict(StimulusType=stype, Model=model_key,
                            accuracy=accuracy, precision=precision,
                            recall=recall, f1=f1))

    return pd.DataFrame(records)


# ── Figure 1 — Overall performance comparison ──────────────────────────────────

def fig_overall_performance(eval_csv, output_dir):
    df = pd.read_csv(eval_csv).set_index("Model")

    metrics = ["accuracy", "precision", "recall", "f1"]
    labels  = ["Accuracy", "Precision", "Recall", "F1"]
    models  = list(df.index)
    x       = np.arange(len(metrics))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for i, model in enumerate(models):
        vals   = [df.loc[model, m] for m in metrics]
        offset = (i - 1) * width
        ax.bar(x + offset, vals, width, label=model,
               color=MODEL_COLOURS.get(model, GREY),
               alpha=0.88, edgecolor="white", linewidth=0.8)

    ax.axhline(0.75, color=GREY, linestyle="--", linewidth=1.2,
               label="Majority class baseline (75%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Overall Performance Across Models\n"
                 "(synthetic + real videos combined)")
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)

    save(fig, output_dir, "fig_overall_performance")


# ── Figure 2 — Per-stimulus F1 breakdown ──────────────────────────────────────

def fig_per_stimulus(results_dir, label_csv, output_dir):
    model_specs = [
        ("LGMD2",      results_dir / "lgmd2",      "_lgmd_output"),
        ("EMD",        results_dir / "emd",         "_emd_output"),
        ("Hybrid SNN", results_dir / "hybrid_snn",  "_snn_output"),
    ]

    all_rows = []
    for name, rdir, suffix in model_specs:
        rows = per_stimulus_metrics(rdir, label_csv, name, suffix)
        if not rows.empty:
            all_rows.append(rows)

    if not all_rows:
        print("  [SKIP] fig_per_stimulus — no data")
        return

    combined = pd.concat(all_rows, ignore_index=True)
    stypes   = sorted(combined["StimulusType"].unique())
    models   = ["LGMD2", "EMD", "Hybrid SNN"]
    x        = np.arange(len(stypes))
    width    = 0.25

    fig, ax = plt.subplots(figsize=(11, 4.5))

    for i, model in enumerate(models):
        sub    = combined[combined["Model"] == model].set_index("StimulusType")
        vals   = [sub.loc[s, "f1"] if s in sub.index else 0.0 for s in stypes]
        offset = (i - 1) * width
        ax.bar(x + offset, vals, width, label=model,
               color=MODEL_COLOURS.get(model, GREY),
               alpha=0.88, edgecolor="white", linewidth=0.8)

    labels = [s.replace("_", "\n") for s in stypes]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score by Stimulus Type and Model")
    ax.legend(loc="upper right", fontsize=8.5, frameon=False)

    save(fig, output_dir, "fig_per_stimulus_f1")


# ── Figure 3 — LGMD membrane potential trace (real data) ─────────────────────

def fig_lgmd_trace(results_dir, output_dir):
    # Use looming_02 — has confirmed collision detections at frames 9-11
    csv_path = results_dir / "lgmd2" / "looming_02_lgmd_output.csv"
    if not csv_path.exists():
        print(f"  [SKIP] fig_lgmd_trace — {csv_path.name} not found")
        return

    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    fig.suptitle("LGMD2 — Internal State (looming_02, real output)",
                 fontsize=11, fontweight="bold")

    # Membrane potential
    axes[0].plot(df["Frame"], df["MembranePotential"],
                 color=RED, linewidth=1.6)
    axes[0].yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    axes[0].set_ylabel("Membrane\nPotential")

    # SFA
    axes[1].plot(df["Frame"], df["SFA"], color=ORANGE, linewidth=1.6)
    axes[1].axhline(0.78, color="black", linestyle="--",
                    linewidth=1.0, label="Spike threshold (Tsp=0.78)")
    axes[1].set_ylabel("SFA")
    axes[1].legend(fontsize=8, loc="upper right")

    # Collision flag
    axes[2].step(df["Frame"], df["Collision"],
                 color=RED, linewidth=1.5, where="mid")
    axes[2].fill_between(df["Frame"], df["Collision"],
                         step="mid", alpha=0.2, color=RED)
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["0", "1"])
    axes[2].set_ylabel("Collision\nFlag")
    axes[2].set_xlabel("Frame")

    # Mark collision window on all panels
    collision_frames = df[df["Collision"] == 1]["Frame"].values
    if len(collision_frames):
        for ax in axes:
            ax.axvspan(collision_frames[0], collision_frames[-1],
                       alpha=0.12, color=RED, label="Collision window")
        axes[0].text(collision_frames[0] + 0.3,
                     df["MembranePotential"].max() * 0.85,
                     f"Frames {collision_frames[0]}–{collision_frames[-1]}",
                     fontsize=8, color=RED)

    for ax in axes:
        ax.set_xlim(df["Frame"].min(), df["Frame"].max())
        ax.grid(alpha=0.25)

    axes[0].text(0.01, 0.04, "Source: real LGMD2 model output",
                 transform=axes[0].transAxes,
                 fontsize=7.5, color=GREY, style="italic")

    plt.tight_layout()
    save(fig, output_dir, "fig_lgmd_trace")


# ── Figure 4 — EMD motion energy: looming vs lateral ─────────────────────────

def fig_emd_energy(results_dir, output_dir):
    loom_path = results_dir / "emd" / "looming_02_emd_output.csv"
    lat_path  = results_dir / "emd" / "lateral_02_emd_output.csv"

    if not loom_path.exists() or not lat_path.exists():
        print("  [SKIP] fig_emd_energy — CSVs not found")
        return

    loom = pd.read_csv(loom_path)
    lat  = pd.read_csv(lat_path)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=False)
    fig.suptitle("EMD — Motion Energy by Stimulus Type",
                 fontsize=11, fontweight="bold")

    ax1.plot(loom["Frame"], loom["MotionEnergy"],
             color=GREEN, linewidth=1.6)
    ax1.axhline(1000,  color=GREY, linestyle="--", linewidth=1.0,
                label="Threshold (+1000)")
    ax1.axhline(-1000, color=GREY, linestyle="--", linewidth=1.0,
                label="Threshold (−1000)")
    ax1.axhline(0, color="black", linewidth=0.6)
    ax1.set_ylabel("Motion Energy")
    ax1.set_title("Looming stimulus — symmetric expansion, signals cancel")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_xlabel("Frame")

    ax2.plot(lat["Frame"], lat["MotionEnergy"],
             color=ORANGE, linewidth=1.6)
    ax2.axhline(1000,  color=GREY, linestyle="--", linewidth=1.0,
                label="Threshold (+1000)")
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.set_ylabel("Motion Energy")
    ax2.set_title("Lateral stimulus — strong rightward directional signal")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_xlabel("Frame")

    plt.tight_layout()
    save(fig, output_dir, "fig_emd_energy")


# ── Figure 5 — SNN training curve ─────────────────────────────────────────────

def fig_snn_training(results_dir, output_dir):
    log_path = results_dir / "snn" / "training_log.csv"
    if not log_path.exists():
        print(f"  [SKIP] fig_snn_training — training_log.csv not found")
        return

    df = pd.read_csv(log_path)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    fig.suptitle("Hybrid SNN — Training Progress",
                 fontsize=11, fontweight="bold")

    ax1.plot(df["Epoch"], df["Loss"], color=BLUE, linewidth=1.6)
    ax1.set_ylabel("Loss")
    ax1.set_title("BCEWithLogitsLoss per epoch")

    ax2.plot(df["Epoch"], df["Accuracy"], color=GREEN, linewidth=1.6)
    ax2.axhline(73.0, color=GREY, linestyle="--", linewidth=1.0,
                label="Majority class baseline (73%)")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_xlabel("Epoch")
    ax2.legend(fontsize=8.5, loc="lower right")

    for ax in (ax1, ax2):
        ax.set_xlim(df["Epoch"].min(), df["Epoch"].max())

    plt.tight_layout()
    save(fig, output_dir, "fig_snn_training")


# ── Figure 6 — SNN spike rate trace ───────────────────────────────────────────

def fig_snn_trace(results_dir, label_csv, output_dir):
    csv_path = results_dir / "hybrid_snn" / "looming_04_snn_output.csv"
    if not csv_path.exists():
        print(f"  [SKIP] fig_snn_trace — {csv_path.name} not found")
        return

    df = pd.read_csv(csv_path)

    # Get ground truth for this video
    gt       = pd.read_csv(label_csv)
    video_gt = gt[gt["Filename"] == "looming_04.mp4"]
    merged   = df.merge(video_gt[["Frame", "TrueLabel"]], on="Frame", how="left")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    fig.suptitle("Hybrid SNN — Output for Looming Stimulus (looming_04)",
                 fontsize=11, fontweight="bold")

    ax1.plot(merged["Frame"], merged["SpikeRate"],
             color=BLUE, linewidth=1.6, label="Spike Rate")
    ax1.set_ylabel("Spike Rate")
    ax1.legend(fontsize=8.5)

    ax2.step(merged["Frame"], merged["Prediction"],
             color=BLUE, linewidth=1.5, where="mid", label="Prediction")
    if "TrueLabel" in merged.columns:
        ax2.step(merged["Frame"], merged["TrueLabel"],
                 color=GREY, linewidth=1.2, linestyle="--",
                 where="mid", label="Ground truth")
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["No collision", "Collision"])
    ax2.set_xlabel("Frame (last frame of 10-frame window)")
    ax2.legend(fontsize=8.5)

    for ax in (ax1, ax2):
        ax.set_xlim(merged["Frame"].min(), merged["Frame"].max())
        ax.grid(alpha=0.25)

    plt.tight_layout()
    save(fig, output_dir, "fig_snn_trace")


# ── Figure 7 — Runtime comparison ─────────────────────────────────────────────

def fig_runtime(output_dir):
    # Runtimes from paper — approximate, different languages
    models = ["LGMD2\n(C# / .NET)", "Hybrid SNN\n(Python / PyTorch)", "EMD\n(Python)"]
    times  = [70, 5, 1]
    colours = [RED, BLUE, GREEN]

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    bars = ax.bar(models, times, color=colours, width=0.5,
                  edgecolor="white", linewidth=1.2)

    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{t} min", ha="center", va="bottom",
                fontweight="bold", fontsize=10)

    ax.set_ylabel("Approximate Runtime (minutes)")
    ax.set_title("Model Runtime — 25 Synthetic + 10 Real Videos\n"
                 "(indicative: models differ in language and environment)")
    ax.set_ylim(0, 85)
    ax.grid(axis="x", alpha=0)

    save(fig, output_dir, "fig_runtime")


# ── Entry point ────────────────────────────────────────────────────────────────

def main(results_dir, label_csv, output_dir):
    results_dir = Path(results_dir)
    label_csv   = Path(label_csv)
    output_dir  = Path(output_dir)

    if not results_dir.exists():
        print(f"[ERROR] results_dir not found: {results_dir}")
        return
    if not label_csv.exists():
        print(f"[ERROR] label_csv not found: {label_csv}")
        return

    eval_csv = results_dir / "evaluation_summary.csv"
    if not eval_csv.exists():
        print(f"[ERROR] evaluation_summary.csv not found — run compare_models.py first")
        return

    print(f"Generating figures → {output_dir}/\n")

    fig_overall_performance(eval_csv,               output_dir)
    fig_per_stimulus(results_dir, label_csv,         output_dir)
    fig_lgmd_trace(results_dir,                      output_dir)
    fig_emd_energy(results_dir,                      output_dir)
    fig_snn_training(results_dir,                    output_dir)
    fig_snn_trace(results_dir, label_csv,            output_dir)
    fig_runtime(output_dir)

    print(f"\nAll figures saved to {output_dir}/")
    print("PDF files go into Overleaf. PNG files are for preview.")


if __name__ == "__main__":
    BASE = Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Generate all paper figures from model results."
    )
    parser.add_argument("--results_dir", default=str(BASE / "results"))
    parser.add_argument("--label_csv",   default=str(BASE / "labels" / "ground_truth.csv"))
    parser.add_argument("--output_dir",  default=str(BASE / "figures"))
    args = parser.parse_args()

    main(args.results_dir, args.label_csv, args.output_dir)
