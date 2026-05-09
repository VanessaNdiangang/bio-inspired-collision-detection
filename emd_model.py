# emd_model.py

import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path


class EMDModel:
    """
    Elementary Motion Detector based on the Reichardt correlator.

    Implements the shared model interface:
        predict_video(video_path)              -> DataFrame
        run_on_folder(input_dir, output_dir)   -> writes one CSV per video
        evaluate(results_dir, label_csv)       -> metrics dict

    No training required. Instantiate and run.

    Example
    -------
        emd = EMDModel()
        emd.run_on_folder("videos/synthetic", "results/emd")
        emd.run_on_folder("videos/real",      "results/emd")
        metrics = emd.evaluate("results/emd", "labels/ground_truth.csv")
    """

    SUFFIX = "_emd_output"   # appended to video stem for output CSV filenames

    def __init__(self, threshold=1000, row_idx=32):
        """
        Args:
            threshold : abs(motion_energy) above this → Prediction = 1.
                        1000 reflects any meaningful directional signal.
            row_idx   : pixel row to sample (default 32 = centre of 64px frame).
        """
        self.threshold = threshold
        self.row_idx   = row_idx

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _correlate(self, prev_frame, curr_frame):
        """
        Reichardt correlator on one frame pair.

        Compares adjacent photoreceptors across time:
            rightward = sum(prev[i]   * curr[i+1])
            leftward  = sum(prev[i+1] * curr[i])
            energy    = rightward - leftward

        Positive  → rightward motion
        Negative  → leftward motion
        Near zero → symmetric expansion (looming) — signals cancel
        """
        prev_row  = prev_frame[self.row_idx, :]
        curr_row  = curr_frame[self.row_idx, :]
        rightward = np.sum(prev_row[:-1] * curr_row[1:])
        leftward  = np.sum(prev_row[1:]  * curr_row[:-1])
        return float(rightward - leftward)

    def _read_frames(self, video_path):
        """Read all frames as float32 (64x64) numpy arrays."""
        cap, frames = cv2.VideoCapture(str(video_path)), []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(cv2.resize(gray, (64, 64)).astype(np.float32))
        cap.release()
        return frames

    def _compute_metrics(self, y_true, y_pred):
        """Return accuracy, precision, recall, F1, TP, FP, FN, TN."""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        n         = tp + fp + fn + tn
        accuracy  = (tp + tn) / n            if n > 0           else 0.0
        precision = tp / (tp + fp)           if (tp + fp) > 0   else 0.0
        recall    = tp / (tp + fn)           if (tp + fn) > 0   else 0.0
        f1        = (2 * precision * recall /
                     (precision + recall))   if (precision + recall) > 0 else 0.0
        return dict(accuracy=round(accuracy, 4), precision=round(precision, 4),
                    recall=round(recall, 4), f1=round(f1, 4),
                    tp=tp, fp=fp, fn=fn, tn=tn)

    # ── Shared interface ───────────────────────────────────────────────────────

    def predict_video(self, video_path):
        """
        Run the EMD on a single video file.

        Frame 1 has no previous frame — its MotionEnergy is 0.0.
        Output length always equals frame count so rows align with
        ground truth label CSVs without any offset adjustment.

        Args:
            video_path : path to a .mp4 file

        Returns:
            DataFrame with columns: Frame, MotionEnergy, Prediction
        """
        frames = self._read_frames(video_path)
        rows   = []

        for i, frame in enumerate(frames):
            energy = 0.0 if i == 0 else self._correlate(frames[i - 1], frame)
            rows.append({
                "Frame":        i + 1,                              # 1-indexed
                "MotionEnergy": round(energy, 4),
                "Prediction":   1 if abs(energy) > self.threshold else 0,
            })

        return pd.DataFrame(rows)

    def run_on_folder(self, input_dir, output_dir):
        """
        Run predict_video on every .mp4 in input_dir.
        Writes one CSV per video to output_dir.

        Call once for synthetic videos and once for real videos:
            emd.run_on_folder("videos/synthetic", "results/emd")
            emd.run_on_folder("videos/real",      "results/emd")

        Output CSV columns: Frame, MotionEnergy, Prediction
        """
        input_dir  = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        videos = sorted(input_dir.glob("*.mp4"))
        if not videos:
            print(f"[ERROR] No .mp4 files found in {input_dir}")
            return

        print(f"EMDModel — {len(videos)} videos → {output_dir}/\n")

        for video_path in videos:
            df       = self.predict_video(video_path)
            out_path = output_dir / f"{video_path.stem}{self.SUFFIX}.csv"
            df.to_csv(out_path, index=False)

            n_pos = int(df["Prediction"].sum())
            mean  = df["MotionEnergy"].abs().mean()
            print(f"  {video_path.name:45s}  "
                  f"frames={len(df):>3}  "
                  f"mean|energy|={mean:>10.1f}  "
                  f"predicted +ve: {n_pos}/{len(df)}")

        print(f"\nCSVs saved to {output_dir}/")

    def evaluate(self, results_dir, label_csv):
        """
        Score all output CSVs in results_dir against ground truth labels.

        Matches each *_emd_output.csv to its video in label_csv by filename,
        then computes frame-level metrics across all matched videos.

        Args:
            results_dir : folder containing *_emd_output.csv files
            label_csv   : ground truth CSV (columns: Filename, Frame, TrueLabel)

        Returns:
            dict — accuracy, precision, recall, f1, tp, fp, fn, tn
        """
        gt             = pd.read_csv(label_csv)
        all_pred       = []
        all_true       = []

        print(f"\nEMDModel evaluation\n{'─' * 65}")

        for csv_path in sorted(Path(results_dir).glob(f"*{self.SUFFIX}.csv")):
            video_name = csv_path.stem.replace(self.SUFFIX, "") + ".mp4"
            video_gt   = gt[gt["Filename"] == video_name]

            if video_gt.empty:
                print(f"  [SKIP] {video_name} — not found in label CSV")
                continue

            df     = pd.read_csv(csv_path)
            merged = df.merge(video_gt[["Frame", "TrueLabel"]], on="Frame", how="inner")

            if merged.empty:
                print(f"  [SKIP] {video_name} — no matching frames after merge")
                continue

            m = self._compute_metrics(merged["TrueLabel"].values,
                                      merged["Prediction"].values)
            all_pred.extend(merged["Prediction"].values)
            all_true.extend(merged["TrueLabel"].values)

            print(f"  {video_name:45s}  "
                  f"acc={m['accuracy']:.2f}  prec={m['precision']:.2f}  "
                  f"rec={m['recall']:.2f}  f1={m['f1']:.2f}  "
                  f"tp={m['tp']} fp={m['fp']} fn={m['fn']}")

        overall = self._compute_metrics(all_true, all_pred)
        print(f"\n{'─' * 65}")
        print(f"  Overall   acc={overall['accuracy']:.3f}  "
              f"prec={overall['precision']:.3f}  "
              f"rec={overall['recall']:.3f}  "
              f"f1={overall['f1']:.3f}")

        return overall

    # ── Self-contained tests ───────────────────────────────────────────────────

    def test(self):
        """
        Unit tests using only numpy and cv2. No external files required.
        """
        print("EMDModel unit tests\n")
        results = []

        def check(name, passed, got, expected):
            label = "PASS" if passed else "FAIL"
            results.append(passed)
            print(f"  [{label}] {name:40s}  got {got}   expected {expected}")

        # 1. Rightward motion
        prev, curr = [np.ones((64, 64), np.float32) * 255 for _ in range(2)]
        prev[self.row_idx, 20:30] = 0
        curr[self.row_idx, 22:32] = 0
        e = self._correlate(prev, curr)
        check("Rightward motion", e > 0, f"{e:.0f}", "> 0")

        # 2. Looming — symmetric expansion cancels
        prev, curr = [np.ones((64, 64), np.float32) * 255 for _ in range(2)]
        cv2.circle(prev, (32, 32),  8, 0, -1)
        cv2.circle(curr, (32, 32), 12, 0, -1)
        e = self._correlate(prev, curr)
        check("Looming near zero", abs(e) < self.threshold,
              f"{e:.0f}", f"|e| < {self.threshold}")

        # 3. Lateral disc moving right
        prev, curr = [np.ones((64, 64), np.float32) * 255 for _ in range(2)]
        cv2.circle(prev, (20, 32), 8, 0, -1)
        cv2.circle(curr, (24, 32), 8, 0, -1)
        e = self._correlate(prev, curr)
        check("Lateral rightward", e > 0, f"{e:.0f}", "> 0")

        # 4. Frame 1 always gets energy 0.0
        dummy = [np.ones((64, 64), np.float32) * 128] * 5
        rows  = self.predict_video_from_frames(dummy)
        check("Frame 1 energy = 0.0",
              rows.iloc[0]["MotionEnergy"] == 0.0,
              rows.iloc[0]["MotionEnergy"], "0.0")

        # 5. Output length equals input length
        check("Output length == input length",
              len(rows) == 5, len(rows), 5)

        # 6. Synthesised looming sequence — mean energy near zero
        looming = []
        for r in range(2, 22):
            img = np.ones((64, 64), np.float32) * 255
            cv2.circle(img, (32, 32), r, 0, -1)
            looming.append(img)
        rows   = self.predict_video_from_frames(looming)
        mean_e = rows.iloc[1:]["MotionEnergy"].abs().mean()
        check("Looming sequence mean near zero",
              mean_e < self.threshold, f"{mean_e:.1f}", f"< {self.threshold}")

        # 7. Synthesised lateral sequence — mean energy positive
        lateral = []
        for x in range(10, 50):
            img = np.ones((64, 64), np.float32) * 255
            cv2.circle(img, (x, 32), 8, 0, -1)
            lateral.append(img)
        rows   = self.predict_video_from_frames(lateral)
        mean_e = rows.iloc[1:]["MotionEnergy"].mean()
        check("Lateral sequence mean positive",
              mean_e > 0, f"{mean_e:.1f}", "> 0")

        n_pass = sum(results)
        n_fail = len(results) - n_pass
        print(f"\n{n_pass}/{len(results)} passed"
              + (f"  — {n_fail} FAILED" if n_fail else "  — all good"))

    def predict_video_from_frames(self, frames):
        """
        Same as predict_video but accepts a list of numpy arrays directly.
        Used by the test suite and any caller that already has frames in memory.
        """
        rows = []
        for i, frame in enumerate(frames):
            energy = 0.0 if i == 0 else self._correlate(frames[i - 1], frame)
            rows.append({
                "Frame":        i + 1,
                "MotionEnergy": round(energy, 4),
                "Prediction":   1 if abs(energy) > self.threshold else 0,
            })
        return pd.DataFrame(rows)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EMDModel — run on videos or test the algorithm."
    )
    parser.add_argument("--input_dir",  default=None,
                        help="Folder of .mp4 files to process")
    parser.add_argument("--output_dir", default="results/emd",
                        help="Where to save output CSVs  (default: results/emd)")
    parser.add_argument("--label_csv",  default=None,
                        help="Ground truth CSV — if provided, evaluation runs "
                             "after prediction")
    parser.add_argument("--threshold",  type=float, default=1000,
                        help="Motion energy threshold for Prediction column "
                             "(default: 1000)")
    parser.add_argument("--test",       action="store_true",
                        help="Run unit tests and exit")
    args = parser.parse_args()

    emd = EMDModel(threshold=args.threshold)

    if args.test:
        emd.test()

    elif args.input_dir:
        emd.run_on_folder(args.input_dir, args.output_dir)
        if args.label_csv:
            emd.evaluate(args.output_dir, args.label_csv)

    else:
        parser.print_help()
