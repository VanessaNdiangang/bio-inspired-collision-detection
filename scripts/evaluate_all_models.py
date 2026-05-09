# evaluate_all_models.py
#
# Runs all three biologically inspired motion detection models on synthetic stimuli
# and computes frame-level performance metrics against programmatic ground truth labels.
#
# Models evaluated:
#   - LGMD2  : Lobula Giant Movement Detector (C# implementation, Fu et al.)
#   - EMD    : Reichardt Elementary Motion Detector (Python, simplified correlator)
#   - SNN    : Hybrid Spiking Neural Network (Python, SpikingJelly, trained)
#
# Ground truth labels are loaded from CSV files generated alongside each stimulus video.
# Labels are frame-level and binary: 1 = collision imminent, 0 = no collision.
# Collision is defined as disc radius > 16 pixels in a 64x64 frame (biologically grounded).
#
# Majority class baseline is reported explicitly.
# All metrics are computed at the frame level.

import subprocess
import pandas as pd
import torch
import cv2
from pathlib import Path
from collections import deque
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

from models.hybrid_snn_v2 import HybridSNNv2
from emd_model import run_emd_on_frames

# ============================
# Configuration
# ============================

STIMULUS_DIR = "stimulus_videos"       # folder containing mp4 and label CSV files
RESULTS_DIR = "evaluation_results"     # folder to save output CSVs
LGMD_EXE = "lgmd/ConsoleProject.csproj"  # path to LGMD C# project
SNN_WEIGHTS = "snn_trained.pth"        # path to trained SNN weights

RESIZE_TO = (64, 64)                   # all models use 64x64 input
TIME_WINDOW = 10                       # SNN processes 10 frames at a time
EMD_THRESHOLD = 1000                   # motion energy threshold for binary EMD prediction
                                       # chosen to reflect sensitivity to any directional motion
                                       # looming produces ~0, lateral produces ~112,000

# ============================
# Helper: Load Ground Truth Labels
# ============================

def load_labels(label_path):
    """
    Loads frame-level ground truth labels from a CSV file.
    Labels were generated programmatically at stimulus creation time.
    Returns a list of integers (0 or 1), one per frame.
    """
    df = pd.read_csv(label_path)
    return df["TrueLabel"].tolist()


# ============================
# Helper: Compute Metrics
# ============================

def compute_metrics(true_labels, predictions, model_name, stimulus_type):
    """
    Computes accuracy, precision, recall and F1 score for a single model
    on a single stimulus type.

    All metrics are frame-level — each frame prediction is compared
    against its corresponding frame-level ground truth label.

    Args:
        true_labels  : list of ground truth labels (0 or 1)
        predictions  : list of model predictions (0 or 1)
        model_name   : string label for the model (lgmd, emd, snn)
        stimulus_type: string label for the stimulus (looming, lateral, etc.)

    Returns:
        dict of metric values
    """
    # Handle edge case where all predictions are one class
    zero_division = 0

    return {
        "Model": model_name,
        "StimulusType": stimulus_type,
        "Frames": len(true_labels),
        "Accuracy": round(accuracy_score(true_labels, predictions), 4),
        "Precision": round(precision_score(true_labels, predictions,
                                           zero_division=zero_division), 4),
        "Recall": round(recall_score(true_labels, predictions,
                                     zero_division=zero_division), 4),
        "F1": round(f1_score(true_labels, predictions,
                             zero_division=zero_division), 4),
        "TruePositives": int(sum(p == 1 and t == 1
                                 for p, t in zip(predictions, true_labels))),
        "FalsePositives": int(sum(p == 1 and t == 0
                                  for p, t in zip(predictions, true_labels))),
        "FalseNegatives": int(sum(p == 0 and t == 1
                                  for p, t in zip(predictions, true_labels))),
    }


# ============================
# Helper: Majority Class Baseline
# ============================

def majority_baseline(true_labels):
    """
    Computes the accuracy of a naive classifier that always predicts
    the most common class. This is the minimum bar any model must beat
    to claim it is detecting something meaningful.

    If 71% of frames are labelled 0, a model that always predicts 0
    scores 71% accuracy without learning anything.
    """
    majority_class = 1 if sum(true_labels) > len(true_labels) / 2 else 0
    correct = sum(t == majority_class for t in true_labels)
    return round(correct / len(true_labels), 4)


# ============================
# Run LGMD (C# via subprocess)
# ============================

def run_lgmd_on_video(video_path, tmp_csv):
    """
    Calls the LGMD2 C# executable on a single video file.
    The executable processes frames sequentially and outputs a CSV
    containing MembranePotential, Spikes, Collision, and SFA per frame.

    The Collision column is binary (0 or 1) and is used directly
    as the frame-level prediction.

    Note: LGMD fires based on expansion rate, not magnitude.
    It may fire before the programmatic threshold (radius > 16)
    due to spike-frequency adaptation — this is biologically correct
    and is discussed as a sensitivity characteristic in the results.
    """
    subprocess.run([
        "dotnet", "run",
        "--project", LGMD_EXE,
        "--",
        str(video_path),
        str(tmp_csv)
    ], capture_output=True)

    if not Path(tmp_csv).exists():
        print(f"[WARNING] LGMD produced no output for {video_path.name}")
        return None

    df = pd.read_csv(tmp_csv)
    Path(tmp_csv).unlink()  # clean up temporary file
    return df["Collision"].tolist()


# ============================
# Run EMD (Python Reichardt Correlator)
# ============================

def run_emd_on_video(video_path):
    """
    Runs the Reichardt Elementary Motion Detector on a single video.
    Extracts frames, computes opponent motion energy per consecutive frame pair,
    and thresholds the result into binary predictions.

    EMD threshold = 1000:
    - Looming produces ~0 energy (symmetric expansion cancels)
    - Lateral produces ~112,000 energy (directional motion dominates)
    - Threshold of 1000 captures any meaningful directional motion signal
    - This reflects the EMD's biological role as a directional motion detector,
      not a collision detector

    Returns one fewer prediction than frames (first frame has no predecessor).
    A leading 0 is prepended to align with frame-level labels.
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, RESIZE_TO)
        frames.append(resized.astype(np.float32))
    cap.release()

    energies = run_emd_on_frames(frames)

    # Threshold continuous motion energy into binary predictions
    predictions = [1 if e > EMD_THRESHOLD else 0 for e in energies]

    # Prepend 0 for the first frame (no previous frame to compare)
    return [0] + predictions


# ============================
# Run SNN (Python, trained weights)
# ============================

def run_snn_on_video(video_path, model, device):
    """
    Runs the trained hybrid SNN on a single video using a sliding window.
    Each window of 10 consecutive frames produces one prediction.

    The model uses:
    - Parallel ON/OFF convolutional pathways (LGMD-inspired looming detection)
    - Spatiotemporal merge layer (EMD-inspired directional integration)
    - LIF neurons with tau=2.0, v_threshold=0.2
    - Trained weights loaded from snn_trained.pth

    Frames before the first full window (frames 0-8) are assigned
    the prediction of the first full window for alignment purposes.
    """
    cap = cv2.VideoCapture(str(video_path))
    frame_buffer = deque(maxlen=TIME_WINDOW)
    predictions = []
    first_pred = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, RESIZE_TO)
        tensor = torch.tensor(resized, dtype=torch.float32).unsqueeze(0) / 255.0
        frame_buffer.append(tensor)

        if len(frame_buffer) < TIME_WINDOW:
            predictions.append(None)  # placeholder until buffer is full
            continue

        input_seq = torch.stack(list(frame_buffer)).to(device)

        with torch.no_grad():
            _, _, logits, pred = model(input_seq)
            p = int(pred.item())
            if first_pred is None:
                first_pred = p
            predictions.append(p)

    cap.release()

    # Fill placeholder frames with first real prediction
    predictions = [first_pred if p is None else p for p in predictions]
    return predictions

# ============================
# Run real videos (using filename-based labels)
# ============================

def evaluate_real_videos(video_dir="real", 
                         labels_csv="real_video_labels.csv",
                         results_dir="evaluation_results_real"):
    """
    Evaluates all three models on real-world LGMD demonstration videos.
    Labels derived from original author filename conventions (Fu et al.)
    loom* = 1 (collision imminent), recession* = 0 (no collision)
    Note: labels are clip-level, not frame-level, because real videos
    do not have programmatic frame-level ground truth.
    All frames in a clip share the clip-level label.
    """
    Path(results_dir).mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    snn_model = HybridSNNv2(time_window=TIME_WINDOW).to(device)
    snn_model.load_state_dict(torch.load(SNN_WEIGHTS, map_location=device))
    snn_model.eval()

    labels_df = pd.read_csv(labels_csv)
    all_metrics = []
    all_frame_results = []

    for _, row in labels_df.iterrows():
        video_file = Path(video_dir) / row["Filename"]
        if not video_file.exists():
            print(f"[SKIP] {row['Filename']} not found")
            continue

        stimulus_type = row["StimulusType"]
        clip_label = int(row["TrueLabel"])

        print(f"\nProcessing: {video_file.name} ({stimulus_type})")

        # Count frames to assign clip-level label to every frame
        cap = cv2.VideoCapture(str(video_file))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        true_labels = [clip_label] * n_frames

        # Run models
        print(f"  Running LGMD...")
        tmp_csv = Path(results_dir) / f"{video_file.stem}_lgmd_tmp.csv"
        lgmd_preds = run_lgmd_on_video(video_file, tmp_csv)
        if lgmd_preds is None:
            lgmd_preds = [0] * n_frames
        lgmd_preds = lgmd_preds[:n_frames]
        if len(lgmd_preds) < n_frames:
            lgmd_preds += [0] * (n_frames - len(lgmd_preds))

        print(f"  Running EMD...")
        emd_preds = run_emd_on_video(video_file)
        emd_preds = emd_preds[:n_frames]
        if len(emd_preds) < n_frames:
            emd_preds += [0] * (n_frames - len(emd_preds))

        print(f"  Running SNN...")
        snn_preds = run_snn_on_video(video_file, snn_model, device)
        snn_preds = snn_preds[:n_frames]
        if len(snn_preds) < n_frames:
            snn_preds += [0] * (n_frames - len(snn_preds))

        # Compute metrics
        for model_name, preds in [("LGMD", lgmd_preds), 
                                   ("EMD", emd_preds), 
                                   ("SNN", snn_preds)]:
            metrics = compute_metrics(true_labels, preds, 
                                     model_name, stimulus_type)
            all_metrics.append(metrics)

        # Store frame results
        for i in range(n_frames):
            all_frame_results.append({
                "Video": video_file.name,
                "StimulusType": stimulus_type,
                "Frame": i,
                "TrueLabel": true_labels[i],
                "LGMD_Pred": lgmd_preds[i],
                "EMD_Pred": emd_preds[i],
                "SNN_Pred": snn_preds[i],
            })

    # Save results
    pd.DataFrame(all_frame_results).to_csv(
        f"{results_dir}/per_frame_results.csv", index=False)
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(f"{results_dir}/metrics_summary.csv", index=False)

    print("\n" + "="*60)
    print("REAL VIDEO RESULTS SUMMARY")
    print("="*60)
    print(f"\nMajority Class Baseline: 50% (balanced dataset)")
    print("\nMetrics by Model and Stimulus Type:")
    print(metrics_df.to_string(index=False))

    return metrics_df


# ============================
# Main Evaluation Loop
# ============================

def evaluate_all(stimulus_dir=STIMULUS_DIR, results_dir=RESULTS_DIR):
    """
    Main evaluation function. Iterates over all stimulus videos,
    runs all three models, loads ground truth labels, and computes metrics.

    Results are saved as:
    - per_video_results.csv  : frame-level predictions for all models
    - metrics_summary.csv    : accuracy, precision, recall, F1 per model per stimulus type
    - baseline_summary.csv   : majority class baseline per stimulus type
    """
    Path(results_dir).mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained SNN — uses weights from training, not random initialisation
    snn_model = HybridSNNv2(time_window=TIME_WINDOW).to(device)
    snn_model.load_state_dict(torch.load(SNN_WEIGHTS, map_location=device))
    snn_model.eval()
    print(f"SNN loaded from {SNN_WEIGHTS}")

    stimulus_path = Path(stimulus_dir)
    all_metrics = []
    all_baselines = []
    all_frame_results = []

    for video_file in sorted(stimulus_path.glob("*.mp4")):
        label_file = stimulus_path / f"{video_file.stem}.csv"

        if not label_file.exists():
            print(f"[SKIP] No label file for {video_file.name}")
            continue

        # Only evaluate on held-out test split (variants 4 and 5)
        # Variants 1-3 were used for training and are excluded
        variant_number = int(video_file.stem[-2:])
        if variant_number <= 3:
            continue

        # Extract stimulus type from filename
        # e.g. looming_01 → looming, looming_lighting_01 → looming_lighting
        stem = video_file.stem
        parts = stem.rsplit("_", 1)
        stimulus_type = parts[0] if len(parts) > 1 else stem

        print(f"\nProcessing: {video_file.name} ({stimulus_type})")

        # Load ground truth labels
        true_labels = load_labels(label_file)
        n_frames = len(true_labels)

        # Majority class baseline for this stimulus
        baseline = majority_baseline(true_labels)
        all_baselines.append({
            "StimulusType": stimulus_type,
            "Video": video_file.name,
            "Frames": n_frames,
            "PositiveFrames": sum(true_labels),
            "NegativeFrames": n_frames - sum(true_labels),
            "MajorityClassBaseline": baseline
        })

        # --- Run LGMD ---
        print(f"  Running LGMD...")
        tmp_csv = Path(results_dir) / f"{stem}_lgmd_tmp.csv"
        lgmd_preds = run_lgmd_on_video(video_file, tmp_csv)

        if lgmd_preds is None:
            lgmd_preds = [0] * n_frames

        # Align length with ground truth
        lgmd_preds = lgmd_preds[:n_frames]
        if len(lgmd_preds) < n_frames:
            lgmd_preds += [0] * (n_frames - len(lgmd_preds))

        lgmd_metrics = compute_metrics(true_labels, lgmd_preds,
                                       "LGMD", stimulus_type)
        all_metrics.append(lgmd_metrics)

        # --- Run EMD ---
        print(f"  Running EMD...")
        emd_preds = run_emd_on_video(video_file)
        emd_preds = emd_preds[:n_frames]
        if len(emd_preds) < n_frames:
            emd_preds += [0] * (n_frames - len(emd_preds))

        emd_metrics = compute_metrics(true_labels, emd_preds,
                                      "EMD", stimulus_type)
        all_metrics.append(emd_metrics)

        # --- Run SNN ---
        print(f"  Running SNN...")
        snn_preds = run_snn_on_video(video_file, snn_model, device)
        snn_preds = snn_preds[:n_frames]
        if len(snn_preds) < n_frames:
            snn_preds += [0] * (n_frames - len(snn_preds))

        snn_metrics = compute_metrics(true_labels, snn_preds,
                                      "SNN", stimulus_type)
        all_metrics.append(snn_metrics)

        # Store frame-level results for this video
        for i in range(n_frames):
            all_frame_results.append({
                "Video": video_file.name,
                "StimulusType": stimulus_type,
                "Frame": i,
                "TrueLabel": true_labels[i],
                "LGMD_Pred": lgmd_preds[i],
                "EMD_Pred": emd_preds[i],
                "SNN_Pred": snn_preds[i],
            })

    # ============================
    # Save Results
    # ============================

    # Frame-level predictions
    frame_df = pd.DataFrame(all_frame_results)
    frame_df.to_csv(f"{results_dir}/per_frame_results.csv", index=False)
    print(f"\nFrame results saved to {results_dir}/per_frame_results.csv")

    # Metrics summary
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(f"{results_dir}/metrics_summary.csv", index=False)
    print(f"Metrics saved to {results_dir}/metrics_summary.csv")

    # Baseline summary
    baseline_df = pd.DataFrame(all_baselines)
    baseline_df.to_csv(f"{results_dir}/baseline_summary.csv", index=False)
    print(f"Baseline saved to {results_dir}/baseline_summary.csv")

    # Print summary table to terminal
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"\nMajority Class Baseline (overall): "
          f"{baseline_df['MajorityClassBaseline'].mean():.1%}")
    print("\nMetrics by Model and Stimulus Type:")
    print(metrics_df.to_string(index=False))

    return metrics_df, baseline_df

# ============================
# Entry Point
# ============================

if __name__ == "__main__":
    print("=== SYNTHETIC STIMULI EVALUATION ===")
    evaluate_all()
    
    print("\n=== REAL VIDEO EVALUATION ===")
    evaluate_real_videos()