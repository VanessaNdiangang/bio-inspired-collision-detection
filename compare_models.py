# compare_models.py
#
# Runs all three models over the same videos and produces a combined
# results table.
#
# Usage:
#   python compare_models.py                           # train SNN from scratch
#   python compare_models.py --snn_weights path.pth    # skip training, load weights

import argparse
import cv2
import pandas as pd
from pathlib import Path

from emd_model        import EMDModel
from lgmd_model       import LGMDModel
from hybrid_snn_model import HybridSNN
from data_loader      import load_all_stimuli, REAL_VIDEOS

BASE = Path(__file__).parent     
ROOT = BASE.parent                    

# BASE is now article_submission/ directly
BASE = Path(__file__).parent

SYNTHETIC_DIR = BASE / "videos" / "synthetic"
REAL_DIR      = BASE / "videos" / "real"
LABEL_CSV     = BASE / "labels" / "ground_truth.csv"
RESULTS_DIR   = BASE / "results"
WEIGHTS_PATH  = BASE / "results" / "snn" / "weights.pth"
LOG_PATH      = BASE / "results" / "snn" / "training_log.csv"

# Longest prefix match — order matters (looming_camera before looming)
STIMULUS_TYPE_PREFIXES = [
    "looming_camera",
    "looming_lighting",
    "looming",
    "lateral",
    "receding",
]


# ── Ground truth builder ───────────────────────────────────────────────────────

def build_ground_truth(synthetic_dir, real_dir, label_csv):
    """
    Build a unified ground truth CSV from two sources:

    Synthetic videos — reads the per-video label CSVs that live alongside
    the .mp4 files in synthetic_dir (e.g. looming_01.csv).
    Each CSV must have a TrueLabel column, one row per frame.

    Real videos — applies the video-level label from the REAL_VIDEOS
    registry in data_loader.py to every frame. Frame count is read
    directly from the video file.

    Writes: label_csv with columns Filename, Frame, TrueLabel, StimulusType
    """
    rows = []

    # ── Synthetic ─────────────────────────────────────────────────────────────
    syn_path = Path(synthetic_dir)
    for csv_path in sorted(syn_path.glob("*.csv")):
        video_name = csv_path.stem + ".mp4"
        if not (syn_path / video_name).exists():
            continue                         # orphan CSV, skip

        df = pd.read_csv(csv_path)
        if "TrueLabel" not in df.columns:
            print(f"  [SKIP] {csv_path.name}: no TrueLabel column")
            continue

        # Infer stimulus type from filename
        stem  = csv_path.stem.lower()
        stype = "unknown"
        for prefix in STIMULUS_TYPE_PREFIXES:
            if stem.startswith(prefix):
                stype = prefix
                break

        # Use Frame column if present, otherwise use row index + 1
        has_frame_col = "Frame" in df.columns
        for i, row in df.iterrows():
            rows.append({
                "Filename":    video_name,
                "Frame":       int(row["Frame"]) if has_frame_col else i + 1,
                "TrueLabel":   int(row["TrueLabel"]),
                "StimulusType": stype,
            })

    # ── Real ──────────────────────────────────────────────────────────────────
    real_path = Path(real_dir)
    for filename, info in REAL_VIDEOS.items():
        video_path = real_path / filename
        if not video_path.exists():
            continue

        cap      = cv2.VideoCapture(str(video_path))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        for frame_num in range(1, n_frames + 1):
            rows.append({
                "Filename":    filename,
                "Frame":       frame_num,
                "TrueLabel":   info["label"],
                "StimulusType": info["stimulus_type"],
            })

    Path(label_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(label_csv, index=False)

    n_pos = sum(r["TrueLabel"] for r in rows)
    print(f"Ground truth built → {label_csv}")
    print(f"  {len(rows)} frames total  "
          f"({n_pos} positive, {len(rows) - n_pos} negative)\n")

# ── SNN training ───────────────────────────────────────────────────────────────

def train_snn(weights_path, log_path):
    snn = HybridSNN()
    sequences, labels = load_all_stimuli(
        str(SYNTHETIC_DIR), split="train", window_size=HybridSNN.TIME_WINDOW
    )
    if len(sequences) == 0:
        print(f"[ERROR] No sequences loaded. Looked in: {SYNTHETIC_DIR.resolve()}")
        print("Check that synthetic videos and their label CSVs exist there.")
        return None
    snn.train(sequences, labels, log_path=str(log_path))
    snn.save(str(weights_path))
    return snn

# ── Main comparison ────────────────────────────────────────────────────────────

def run_all(snn_weights, eval_only=False):

    if not Path(LABEL_CSV).exists():
        print("Ground truth CSV not found — building...")
        build_ground_truth(SYNTHETIC_DIR, REAL_DIR, LABEL_CSV)
    else:
        print(f"Using existing ground truth: {LABEL_CSV}\n")

    models = {
        "LGMD2":      LGMDModel(project_path=BASE / "lgmd"),
        "EMD":        EMDModel(),
        "Hybrid SNN": HybridSNN(),
    }

    if not eval_only:
        models["LGMD2"].build()
        models["Hybrid SNN"].load(snn_weights)

        print("\n" + "═" * 65)
        print("RUNNING ALL MODELS ON SYNTHETIC AND REAL VIDEOS")
        print("═" * 65)

        for name, model in models.items():
            output_dir = RESULTS_DIR / name.lower().replace(" ", "_")
            print(f"\n── {name} ──")
            model.run_on_folder(str(SYNTHETIC_DIR), output_dir)
            model.run_on_folder(str(REAL_DIR),      output_dir)

    print("\n" + "═" * 65)
    print("EVALUATION")
    print("═" * 65)

    summary_rows = []
    for name, model in models.items():
        output_dir = RESULTS_DIR / name.lower().replace(" ", "_")
        metrics    = model.evaluate(str(output_dir), str(LABEL_CSV))
        summary_rows.append({"Model": name, **metrics})

    summary  = pd.DataFrame(summary_rows).set_index("Model")
    out_path = RESULTS_DIR / "evaluation_summary.csv"
    summary.reset_index().to_csv(out_path, index=False)

    print("\n" + "═" * 65)
    print("COMBINED RESULTS")
    print("═" * 65)
    print(summary[["accuracy", "precision", "recall", "f1",
                   "tp", "fp", "fn"]].to_string())
    print(f"\nSummary saved → {out_path}")

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snn_weights", default=None)
    parser.add_argument("--rebuild_gt",  action="store_true",
                        help="Force rebuild of ground truth CSV")
    parser.add_argument("--eval_only",   action="store_true",
                        help="Skip model runs, go straight to evaluation")
    args = parser.parse_args()

    if args.rebuild_gt and Path(LABEL_CSV).exists():
        Path(LABEL_CSV).unlink()
        print("Ground truth CSV deleted — will rebuild.\n")

    if not args.eval_only:
        if WEIGHTS_PATH.exists():
            print(f"Using existing SNN weights: {WEIGHTS_PATH}\n")
        else:
            print("Training SNN from scratch...\n")
            train_snn(WEIGHTS_PATH, LOG_PATH)
        args.snn_weights = WEIGHTS_PATH

    run_all(args.snn_weights if not args.eval_only else WEIGHTS_PATH,
            eval_only=args.eval_only)