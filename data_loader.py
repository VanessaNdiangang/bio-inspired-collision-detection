# data_loader.py

import re
import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path


# ── Real video registry ────────────────────────────────────────────────────────
#
# These are the 10 LGMD-dataset videos used in the study.
# They are test-only — the SNN was trained on synthetic videos only.
#
# Key difference from synthetic videos:
#   Synthetic → per-frame label CSV (one label per frame)
#   Real      → video-level label (whole video is looming or receding)
#
# Source: LGMD2 open-source package (Fu et al.)

REAL_VIDEOS = {
    # Looming — label 1
    "loom_on_inhibition.mp4":      {"label": 1, "stimulus_type": "looming"},
    "looming_grouping.mp4":         {"label": 1, "stimulus_type": "looming"},
    "looming_off_inhibition.mp4":   {"label": 1, "stimulus_type": "looming"},
    "looming_photoreceptor.mp4":    {"label": 1, "stimulus_type": "looming"},
    "raw_loom.mp4":                 {"label": 1, "stimulus_type": "looming"},
    # Receding — label 0
    "recession_grouping.mp4":       {"label": 0, "stimulus_type": "receding"},
    "recession_off_inhibition.mp4": {"label": 0, "stimulus_type": "receding"},
    "recession_on_inhibition.mp4":  {"label": 0, "stimulus_type": "receding"},
    "recession_photoreceptor.mp4":  {"label": 0, "stimulus_type": "receding"},
    "recession_raw.mp4":            {"label": 0, "stimulus_type": "receding"},
}


# ── Internal helpers ───────────────────────────────────────────────────────────

def _read_frames(video_path):
    """
    Read all frames from a video file.
    Returns a list of (64, 64) uint8 grayscale arrays.
    """
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(cv2.resize(gray, (64, 64)))
    cap.release()
    return frames


def _make_sequences(frames, frame_labels, window_size):
    """
    Slide a window of `window_size` frames across the video.
    Each sequence is labelled by majority vote of its frame labels.

    Returns:
        sequences — list of float32 tensors, shape (window_size, 1, 64, 64)
        labels    — list of int (0 or 1)
    """
    sequences, labels = [], []
    for start in range(len(frames) - window_size + 1):
        window_frames = frames[start : start + window_size]
        window_labels = frame_labels[start : start + window_size]
        majority      = 1 if sum(window_labels) > (window_size / 2) else 0
        tensor = (
            torch.tensor(np.array(window_frames), dtype=torch.float32)
            .unsqueeze(1)   # (window_size, 1, 64, 64)
            / 255.0
        )
        sequences.append(tensor)
        labels.append(majority)
    return sequences, labels


def _parse_variant_number(stem):
    """
    Extract the trailing variant number from a synthetic filename.
      'looming_03'  → 3
      'lateral_12'  → 12
    Returns None if the filename does not end in a number.
    """
    match = re.search(r'_(\d+)$', stem)
    return int(match.group(1)) if match else None


# ── Per-video loaders ──────────────────────────────────────────────────────────

def load_synthetic_video(video_path, label_path, window_size=10):
    """
    Load one synthetic video and its per-frame label CSV.
    Raises FileNotFoundError if the CSV is missing.
    Raises ValueError if frame count and label count do not match.
    """
    frames = _read_frames(video_path)

    if not Path(label_path).exists():
        raise FileNotFoundError(
            f"Label CSV not found: {label_path}\n"
            f"Each synthetic video must have a matching CSV."
        )

    label_df   = pd.read_csv(label_path)
    labels_raw = label_df["TrueLabel"].tolist()

    if len(frames) != len(labels_raw):
        raise ValueError(
            f"{video_path.name}: {len(frames)} frames but "
            f"{len(labels_raw)} rows in label CSV — they must match exactly."
        )

    return _make_sequences(frames, labels_raw, window_size)


def load_real_video(video_path, video_label, window_size=10):
    """
    Load one real video using its video-level label.
    Every frame in the video inherits the same label.
    No per-frame CSV is needed or expected.
    """
    frames       = _read_frames(video_path)
    frame_labels = [video_label] * len(frames)
    return _make_sequences(frames, frame_labels, window_size)


# ── Main loader ────────────────────────────────────────────────────────────────

def load_all_stimuli(stimulus_dir, window_size=10, split="train"):
    """
    Load all stimulus videos from a directory.

    Synthetic videos  (e.g. looming_01.mp4)
    ─────────────────────────────────────────
    Recognised by a trailing variant number in the filename.
    Require a matching per-frame label CSV in the same folder.
    Filtered by split:
      "train" — variants 1–3 only
      "test"  — variants 4–5 only
      "all"   — all variants

    Real videos  (e.g. loom_on_inhibition.mp4)
    ───────────────────────────────────────────
    Recognised by name from the REAL_VIDEOS registry above.
    Label comes from the registry, not a CSV.
    Only included when split="test" or split="all".
    Never used for training — the SNN was trained on synthetic data only.

    Args:
        stimulus_dir  folder containing .mp4 files
        window_size   frames per sequence (default 10)
        split         "train", "test", or "all"

    Returns:
        sequences  list of tensors, each (window_size, 1, 64, 64)
        labels     list of int (0 or 1)
    """
    stimulus_path             = Path(stimulus_dir)
    all_sequences, all_labels = [], []
    n_synthetic = n_real = n_skipped = 0

    for video_file in sorted(stimulus_path.glob("*.mp4")):
        filename = video_file.name

        # ── Real video ────────────────────────────────────────────────────────
        if filename in REAL_VIDEOS:
            if split == "train":
                continue    # real videos never used for training
            info       = REAL_VIDEOS[filename]
            seqs, labs = load_real_video(video_file, info["label"], window_size)
            all_sequences.extend(seqs)
            all_labels.extend(labs)
            n_real += 1
            continue

        # ── Synthetic video ───────────────────────────────────────────────────
        variant = _parse_variant_number(video_file.stem)

        if variant is None:
            print(f"  [SKIP] {filename}: cannot parse variant number — "
                  f"expected filename ending in e.g. _01, _02")
            n_skipped += 1
            continue

        if split == "train" and variant > 3:
            continue
        if split == "test"  and variant <= 3:
            continue

        label_file = stimulus_path / f"{video_file.stem}.csv"
        try:
            seqs, labs = load_synthetic_video(video_file, label_file, window_size)
            all_sequences.extend(seqs)
            all_labels.extend(labs)
            n_synthetic += 1
        except (FileNotFoundError, ValueError) as e:
            print(f"  [SKIP] {e}")
            n_skipped += 1

    # Summary
    n_pos = sum(all_labels)
    n_neg = len(all_labels) - n_pos
    print(f"Loaded  {n_synthetic} synthetic + {n_real} real videos  "
          f"| {n_skipped} skipped  | split='{split}'")
    print(f"        {len(all_sequences)} sequences total  "
          f"({n_pos} positive, {n_neg} negative)")
    if n_skipped:
        print(f"Warning: {n_skipped} file(s) skipped — "
              f"check filenames and label CSVs above")

    return all_sequences, all_labels


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Train split (synthetic variants 1-3 only) ===")
    seqs, labs = load_all_stimuli("stimulus_videos", split="train")
    if seqs:
        print(f"Sequence shape: {seqs[0].shape}")

    print("\n=== Test split (synthetic variants 4-5 + real videos) ===")
    seqs, labs = load_all_stimuli("stimulus_videos", split="test")
    if seqs:
        print(f"Sequence shape: {seqs[0].shape}")
