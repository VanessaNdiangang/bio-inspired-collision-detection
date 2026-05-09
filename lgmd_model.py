# lgmd_model.py

import argparse
import subprocess
import tempfile
import pandas as pd
from pathlib import Path


class LGMDModel:
    """
    Python wrapper around the LGMD2 C# implementation.

    Implements the same shared interface as EMDModel:
        predict_video(video_path)              -> DataFrame
        run_on_folder(input_dir, output_dir)   -> writes one CSV per video
        evaluate(results_dir, label_csv)       -> metrics dict

    The C# binary is called via subprocess for each video.
    Call build() once before processing to avoid per-video rebuild overhead.

    Example
    -------
        lgmd = LGMDModel(project_path="models/lgmd")
        lgmd.build()
        lgmd.run_on_folder("videos/synthetic", "results/lgmd")
        lgmd.run_on_folder("videos/real",      "results/lgmd")
        metrics = lgmd.evaluate("results/lgmd", "labels/ground_truth.csv")
    """

    SUFFIX = "_lgmd_output"   # matches the CSV filenames Program.cs writes

    # Columns written by Program.cs
    RAW_COLUMNS = [
        "Frame", "MembranePotential", "Spikes", "Collision",
        "TotalSpikes", "MotionEnergy", "SFA"
    ]

    def __init__(self, project_path="models/lgmd"):
        """
        Args:
            project_path : path to the folder containing ConsoleProject.csproj
        """
        self.project_path = Path(project_path)
        self._built       = False

        if not self.project_path.exists():
            raise FileNotFoundError(
                f"LGMD project not found at: {self.project_path}\n"
                f"Expected a folder containing ConsoleProject.csproj."
            )

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self):
        """
        Compile the C# project once.

        Call this before run_on_folder to avoid dotnet rebuilding on
        every video. Safe to call multiple times — dotnet skips if
        nothing has changed.
        """
        print(f"Building LGMD project at {self.project_path}/ ...")
        result = self._run_dotnet(["dotnet", "build", str(self.project_path),
                                   "--verbosity", "quiet"])
        if result.returncode != 0:
            raise RuntimeError(
                f"dotnet build failed:\n{result.stderr}"
            )
        self._built = True
        print("Build successful.\n")

    # ── Core subprocess call ───────────────────────────────────────────────────

    def _run_dotnet(self, cmd):
        """Run a dotnet command and return the CompletedProcess."""
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            raise EnvironmentError(
                "dotnet not found on PATH.\n"
                "Install .NET 8.0 SDK from https://dotnet.microsoft.com"
            )

    def _call_lgmd(self, video_path, csv_path):
        """
        Call the compiled LGMD2 binary on one video.

        Uses --no-build if build() has already been called, otherwise
        allows dotnet to build on first run.
        """
        cmd = [
            "dotnet", "run",
            "--project", str(self.project_path),
        ]
        if self._built:
            cmd.append("--no-build")

        cmd += ["--", str(video_path), str(csv_path)]

        result = self._run_dotnet(cmd)

        if result.returncode != 0:
            raise RuntimeError(
                f"LGMD failed on {video_path.name}:\n{result.stderr}"
            )

        if not Path(csv_path).exists():
            raise RuntimeError(
                f"LGMD ran without error but no CSV was written for "
                f"{video_path.name}. Check Program.cs output path logic."
            )

        return result.stdout   # terminal output from C# (collision detections)

    # ── Shared interface ───────────────────────────────────────────────────────

    def predict_video(self, video_path):
        """
        Run the LGMD2 model on a single video file.

        Calls the C# binary, reads the CSV it writes, adds a Prediction
        column (mirrors Collision) for interface consistency with EMDModel,
        then returns the DataFrame.

        Args:
            video_path : path to a .mp4 file

        Returns:
            DataFrame with columns:
                Frame, MembranePotential, Spikes, Collision,
                TotalSpikes, MotionEnergy, SFA, Prediction
        """
        video_path = Path(video_path)

        # Write to a temp file so predict_video has no side effects on disk
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            terminal_output = self._call_lgmd(video_path, tmp_path)
            df = pd.read_csv(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        # Print the C# terminal output (collision frame detections)
        if terminal_output.strip():
            print(terminal_output.strip())

        # Add Prediction column so downstream code doesn't need to know
        # that LGMD calls its output "Collision" rather than "Prediction"
        df["Prediction"] = df["Collision"]

        return df

    def run_on_folder(self, input_dir, output_dir):
        """
        Run predict_video on every .mp4 in input_dir.
        Writes one CSV per video to output_dir.

        Call once for synthetic videos and once for real videos:
            lgmd.run_on_folder("videos/synthetic", "results/lgmd")
            lgmd.run_on_folder("videos/real",      "results/lgmd")

        Output CSV columns:
            Frame, MembranePotential, Spikes, Collision,
            TotalSpikes, MotionEnergy, SFA, Prediction
        """
        input_dir  = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        videos = sorted(input_dir.glob("*.mp4"))
        if not videos:
            print(f"[ERROR] No .mp4 files found in {input_dir}")
            return

        print(f"LGMDModel — {len(videos)} videos → {output_dir}/\n")

        for video_path in videos:
            out_path = output_dir / f"{video_path.stem}{self.SUFFIX}.csv"

            try:
                # Write directly to the output path — avoids double read/write
                terminal_output = self._call_lgmd(video_path, out_path)
                df = pd.read_csv(out_path)
                df["Prediction"] = df["Collision"]
                df.to_csv(out_path, index=False)

                n_collisions = int(df["Collision"].sum())
                peak_mp      = df["MembranePotential"].max()

                print(f"  {video_path.name:45s}  "
                      f"frames={len(df):>3}  "
                      f"collisions={n_collisions:>3}  "
                      f"peak_mp={peak_mp:>10.1f}")

                # Print per-frame collision lines from C# terminal output
                for line in terminal_output.splitlines():
                    if "Collision detected" in line:
                        print(f"    {line.strip()}")

            except RuntimeError as e:
                print(f"  [ERROR] {video_path.name}: {e}")

        print(f"\nCSVs saved to {output_dir}/")

    def evaluate(self, results_dir, label_csv):
        """
        Score all output CSVs in results_dir against ground truth labels.

        Identical logic to EMDModel.evaluate — uses the Prediction column
        which mirrors Collision in every output CSV.

        Args:
            results_dir : folder containing *_lgmd_output.csv files
            label_csv   : ground truth CSV (columns: Filename, Frame, TrueLabel)

        Returns:
            dict — accuracy, precision, recall, f1, tp, fp, fn, tn
        """
        gt       = pd.read_csv(label_csv)
        all_pred = []
        all_true = []

        print(f"\nLGMDModel evaluation\n{'─' * 65}")

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

            # Show biological outputs alongside classification metrics
            peak_mp = df["MembranePotential"].max()
            sfa_max = df["SFA"].max()
            print(f"  {video_name:45s}  "
                  f"acc={m['accuracy']:.2f}  prec={m['precision']:.2f}  "
                  f"rec={m['recall']:.2f}  f1={m['f1']:.2f}  "
                  f"peak_mp={peak_mp:.0f}  sfa_max={sfa_max:.3f}")

        overall = self._compute_metrics(all_true, all_pred)
        print(f"\n{'─' * 65}")
        print(f"  Overall   acc={overall['accuracy']:.3f}  "
              f"prec={overall['precision']:.3f}  "
              f"rec={overall['recall']:.3f}  "
              f"f1={overall['f1']:.3f}")

        return overall

    # ── Internal ───────────────────────────────────────────────────────────────

    def _compute_metrics(self, y_true, y_pred):
        """Accuracy, precision, recall, F1, TP, FP, FN, TN."""
        import numpy as np
        y_true, y_pred = map(np.array, [y_true, y_pred])
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
        return dict(accuracy=round(accuracy, 4), precision=round(precision, 4),
                    recall=round(recall, 4), f1=round(f1, 4),
                    tp=tp, fp=fp, fn=fn, tn=tn)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LGMDModel — run the LGMD2 C# model on a folder of videos."
    )
    parser.add_argument("--project",    default="models/lgmd",
                        help="Path to the C# project folder  "
                             "(default: models/lgmd)")
    parser.add_argument("--input_dir",  default=None,
                        help="Folder of .mp4 files to process")
    parser.add_argument("--output_dir", default="results/lgmd",
                        help="Where to save output CSVs  (default: results/lgmd)")
    parser.add_argument("--label_csv",  default=None,
                        help="Ground truth CSV — if provided, evaluation runs "
                             "after prediction")
    parser.add_argument("--build",      action="store_true",
                        help="Build the C# project before running")
    args = parser.parse_args()

    lgmd = LGMDModel(project_path=args.project)

    if args.build:
        lgmd.build()

    if args.input_dir:
        lgmd.run_on_folder(args.input_dir, args.output_dir)
        if args.label_csv:
            lgmd.evaluate(args.output_dir, args.label_csv)

    elif not args.build:
        parser.print_help()
