# label_real_videos.py
#
# Assigns ground truth labels to real-world LGMD demonstration videos
# based on filename conventions defined by Fu et al. (original authors).
#
# Label 1 — filename contains "loom" — looming stimulus, collision imminent
# Label 0 — filename contains "recession" — receding stimulus, no collision
# Excluded — all other filenames (layer visualisation videos, ambiguous)

import pandas as pd
from pathlib import Path

def label_real_videos(video_dir="real"):
    video_path = Path(video_dir)
    rows = []

    for video_file in sorted(video_path.glob("*.mp4")):
        name = video_file.name.lower()

        if "loom" in name:
            label = 1
        elif "recession" in name:
            label = 0
        else:
            continue  # exclude ambiguous files

        rows.append({
            "Filename": video_file.name,
            "TrueLabel": label,
            "StimulusType": "looming" if label == 1 else "receding"
        })

    df = pd.DataFrame(rows)
    df.to_csv("real_video_labels.csv", index=False)

    print(f"Labelled {len(df)} videos")
    print(f"  Positive (looming): {df['TrueLabel'].sum()}")
    print(f"  Negative (receding): {(df['TrueLabel']==0).sum()}")
    print(f"  Excluded: {len(list(video_path.glob('*.mp4'))) - len(df)}")
    print(f"Saved to real_video_labels.csv")

    return df

if __name__ == "__main__":
    label_real_videos("real")