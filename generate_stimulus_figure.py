# generate_stimulus_figure.py
#
# Extracts representative frames from actual videos and composes a
# single figure showing all stimulus types side by side.
#
# Run from article submission/:
#   python generate_stimulus_figure.py

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

BASE        = Path(__file__).parent
VIDEOS_SYN  = BASE / "videos" / "synthetic"
VIDEOS_REAL = BASE / "videos" / "real"
OUTPUT_DIR  = BASE / "figures"


def extract_frame(video_path, frame_number):
    """Extract a single frame from a video by frame index (0-based)."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (64, 64))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Define which frame to extract from each video ─────────────────────────
    # Frame chosen to show the stimulus clearly:
    #   looming        → frame 20 (disc large, mid-expansion)
    #   receding       → frame 5  (disc still large, beginning of recession)
    #   lateral        → frame 15 (disc mid-travel)
    #   looming+light  → frame 20 (disc large with brightness variation)
    #   looming+camera → frame 20 (disc large with positional shift)
    #   real looming   → frame 30 (somewhere mid-sequence)
    #   real receding  → frame 30

    panels = [
        # (video_path,                                  frame_idx, label,              border_colour)
        (VIDEOS_SYN  / "looming_03.mp4",               20,  "Looming\n(synthetic)",      "#2166AC"),
        (VIDEOS_SYN  / "receding_03.mp4",               5,  "Receding\n(synthetic)",     "#4DAC26"),
        (VIDEOS_SYN  / "lateral_03.mp4",               15,  "Lateral\n(synthetic)",      "#F4A736"),
        (VIDEOS_SYN  / "looming_lighting_03.mp4",      20,  "Looming +\nlighting",       "#762A83"),
        (VIDEOS_SYN  / "looming_camera_03.mp4",        20,  "Looming +\ncamera motion",  "#D6604D"),
        (VIDEOS_REAL / "looming_off_inhibition.mp4",   30,  "Looming\n(real video)",     "#2166AC"),
        (VIDEOS_REAL / "recession_grouping.mp4",       30,  "Receding\n(real video)",    "#4DAC26"),
    ]

    # ── Extract frames ────────────────────────────────────────────────────────
    frames, labels, colours, missing = [], [], [], []

    for video_path, frame_idx, label, colour in panels:
        if not video_path.exists():
            print(f"  [SKIP] {video_path.name} — file not found")
            missing.append(label)
            continue
        frame = extract_frame(video_path, frame_idx)
        if frame is None:
            print(f"  [SKIP] {video_path.name} — could not read frame {frame_idx}")
            missing.append(label)
            continue
        frames.append(frame)
        labels.append(label)
        colours.append(colour)
        print(f"  [OK]   {video_path.name}  frame {frame_idx}")

    if not frames:
        print("\n[ERROR] No frames extracted. Check that videos/ exists.")
        return

    n = len(frames)

    # ── Compose figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 3.8))
    if n == 1:
        axes = [axes]

    fig.suptitle(
        "Stimulus Types Used in the Study\n"
        "Left: programmatically generated synthetic stimuli (64×64 px, 20 fps). "
        "Right: real-world videos from the LGMD source package.",
        fontsize=9, y=1.02, wrap=True
    )

    for ax, frame, label, colour in zip(axes, frames, labels, colours):
        ax.imshow(frame, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(label, fontsize=8.5, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])

        # Coloured border to distinguish stimulus type
        for spine in ax.spines.values():
            spine.set_edgecolor(colour)
            spine.set_linewidth(2.5)
            spine.set_visible(True)

    # Divider between synthetic and real
    if any("real" in l for l in labels):
        real_start = next(i for i, l in enumerate(labels) if "real" in l)
        # Draw a vertical separator line between synthetic and real panels
        fig.text(
            (real_start / n) + 0.005, 0.12,
            "◀ synthetic    real ▶",
            ha="center", va="bottom", fontsize=7.5, color="#888888",
            style="italic"
        )

    plt.tight_layout()

    out_pdf = OUTPUT_DIR / "fig_stimulus_overview.pdf"
    out_png = OUTPUT_DIR / "fig_stimulus_overview.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"\nSaved → {out_pdf}")
    print(f"Saved → {out_png}")

    if missing:
        print(f"\nMissing videos (not included): {', '.join(missing)}")
        print("Check that videos/synthetic/ and videos/real/ contain the expected files.")


if __name__ == "__main__":
    main()
