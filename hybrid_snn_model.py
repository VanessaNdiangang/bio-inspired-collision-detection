# hybrid_snn_v2.py

import argparse
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import deque
from pathlib import Path
from spikingjelly.clock_driven import neuron, functional


# ── Inner network (private) ────────────────────────────────────────────────────

class _SNNNetwork(nn.Module):
    """
    The spiking neural network architecture.
    Not used directly — accessed through HybridSNN.

    Architecture:
        ON pathway  (light increments, LGMD-inspired) ─┐
                                                         ├─ merge ─ LIF ─ pool ─ FC → logit
        OFF pathway (light decrements, LGMD-inspired) ─┘
              ↑
        The merge convolution across ON+OFF channels implements
        the spatiotemporal correlation that mirrors EMD behaviour.
    """

    def __init__(self):
        super().__init__()
        self.conv_on   = nn.Conv2d(1, 8,  kernel_size=3, padding=1)
        self.lif_on    = neuron.LIFNode(tau=2.0, v_threshold=0.2)

        self.conv_off  = nn.Conv2d(1, 8,  kernel_size=3, padding=1)
        self.lif_off   = neuron.LIFNode(tau=2.0, v_threshold=0.2)

        self.merge     = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.lif_merge = neuron.LIFNode(tau=2.0, v_threshold=0.2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(16, 1)

    def forward(self, sequence):
        """
        Args:
            sequence: tensor (time_window, 1, H, W)
                      Each position is a distinct frame — not the same
                      frame repeated. This is the fix from the original
                      where x_t = x * 5.0 used the same frame every step.

        Returns:
            logit:      scalar — raw output before sigmoid
            spike_rate: scalar — mean spike rate across all LIF layers
            mem_pot:    scalar — mean pooled membrane potential
        """
        functional.reset_net(self)
        spike_acc, mem_acc = [], []

        for t in range(sequence.size(0)):
            frame = sequence[t].unsqueeze(0) * 5.0   # (1, 1, H, W)

            on     = self.lif_on(self.conv_on(frame))
            off    = self.lif_off(self.conv_off(1.0 - frame))
            merged = self.lif_merge(self.merge(torch.cat([on, off], dim=1)))
            pooled = self.pool(merged).view(1, -1)    # (1, 16)

            spike_acc.append((merged > 0).float().mean())
            mem_acc.append(pooled)

        spike_rate = torch.stack(spike_acc).mean()
        mem_tensor = torch.stack(mem_acc).mean(dim=0)   # (1, 16)
        logit      = self.fc(mem_tensor).squeeze()       # scalar

        return logit, spike_rate, mem_tensor.mean()


# ── Public class ───────────────────────────────────────────────────────────────

class HybridSNN:
    """
    Hybrid Spiking Neural Network combining LGMD-inspired ON/OFF pathways
    with EMD-inspired spatiotemporal integration.

    Implements the shared model interface:
        predict_video(video_path)              -> DataFrame
        run_on_folder(input_dir, output_dir)   -> writes one CSV per video
        evaluate(results_dir, label_csv)       -> metrics dict

    Additionally:
        train(sequences, labels, ...)          -> trains and saves weights
        save(path) / load(path)                -> weight persistence

    Example
    -------
        from data_loader import load_all_stimuli

        snn = HybridSNN()

        # Train
        sequences, labels = load_all_stimuli("videos/synthetic", split="train")
        snn.train(sequences, labels, log_path="results/snn/training_log.csv")
        snn.save("results/snn/weights.pth")

        # Evaluate
        snn.load("results/snn/weights.pth")
        snn.run_on_folder("videos/synthetic", "results/snn")
        snn.run_on_folder("videos/real",      "results/snn")
        metrics = snn.evaluate("results/snn", "labels/ground_truth.csv")
    """

    SUFFIX      = "_snn_output"
    TIME_WINDOW = 10
    INPUT_SCALE = 5.0

    def __init__(self):
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._net    = _SNNNetwork().to(self.device)
        self._trained = False

    # ── Training ───────────────────────────────────────────────────────────────

    def train(self, sequences, labels,
              epochs=50, lr=1e-3,
              log_path="results/snn/training_log.csv"):
        """
        Train the network on sequences from data_loader.

        Args:
            sequences : list of tensors (time_window, 1, 64, 64)
                        from load_all_stimuli(..., split="train")
            labels    : list of int (0 or 1), same length as sequences
            epochs    : number of training epochs (default 50)
            lr        : learning rate (default 1e-3)
            log_path  : where to save the per-epoch training log CSV
        """
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        print(f"Training on {self.device}")
        print(f"  {len(sequences)} sequences  "
              f"({n_pos} positive, {n_neg} negative)")
        print(f"  Majority class baseline: "
              f"{max(n_pos, n_neg) / len(sequences) * 100:.1f}%\n")

        # Class imbalance weight — computed from actual data counts
        pos_weight = torch.tensor([n_neg / n_pos], device=self.device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimiser  = torch.optim.Adam(self._net.parameters(), lr=lr)

        # training_log initialised BEFORE the epoch loop
        # (original bug: it was inside, so the CSV only ever had one row)
        training_log = []

        self._net.train()

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            correct    = 0
            idx        = np.random.permutation(len(sequences))

            for i in idx:
                seq   = sequences[i].to(self.device)
                label = torch.tensor(
                    [labels[i]], dtype=torch.float32, device=self.device
                )

                optimiser.zero_grad()
                logit, _, _ = self._net(seq)
                loss        = criterion(logit.unsqueeze(0), label)
                loss.backward()
                optimiser.step()

                total_loss += loss.item()
                correct    += int((logit.item() > 0) == bool(labels[i]))

            avg_loss = total_loss / len(sequences)
            accuracy = correct   / len(sequences) * 100
            training_log.append({"Epoch": epoch, "Loss": avg_loss,
                                  "Accuracy": accuracy})

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:>3}/{epochs}   "
                      f"loss={avg_loss:.4f}   accuracy={accuracy:.1f}%")

        # Save training log
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(training_log).to_csv(log_path, index=False)
        print(f"\nTraining complete. Log saved → {log_path}")

        self._net.eval()
        self._trained = True

    # ── Weights ────────────────────────────────────────────────────────────────

    def save(self, path):
        """Save trained weights to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._net.state_dict(), path)
        print(f"Weights saved → {path}")

    def load(self, path):
        """Load weights from disk."""
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Weights file not found: {path}\n"
                f"Run train() first, then save()."
            )
        self._net.load_state_dict(
            torch.load(path, map_location=self.device)
        )
        self._net.eval()
        self._trained = True
        print(f"Weights loaded from {path}")

    # ── Shared interface ───────────────────────────────────────────────────────

    def predict_video(self, video_path):
        """
        Run the SNN on a single video file.

        Reads all frames, slides a window of TIME_WINDOW frames across them,
        and produces one prediction per window.

        The Frame column contains the last frame number of each window
        (1-indexed), so it aligns with frame-level ground truth label CSVs
        when joined on Frame.

        A 30-frame video with TIME_WINDOW=10 produces 21 rows:
            Frame 10  (window frames 1–10)
            Frame 11  (window frames 2–11)
            ...
            Frame 30  (window frames 21–30)

        Args:
            video_path : path to a .mp4 file

        Returns:
            DataFrame with columns:
                Frame, SpikeRate, MembranePotential, Logit, Prediction
        """
        if not self._trained:
            print("[WARNING] Model has not been trained or loaded. "
                  "Predictions will be random.")

        frames = self._read_frames(video_path)
        if len(frames) < self.TIME_WINDOW:
            raise ValueError(
                f"{Path(video_path).name}: only {len(frames)} frames, "
                f"need at least {self.TIME_WINDOW}."
            )

        rows   = []
        buffer = deque(maxlen=self.TIME_WINDOW)

        with torch.no_grad():
            for i, frame in enumerate(frames):
                buffer.append(frame)
                if len(buffer) < self.TIME_WINDOW:
                    continue

                seq   = torch.stack(list(buffer)).to(self.device)  # (T, 1, 64, 64)
                logit, spike_rate, mem_pot = self._net(seq)

                rows.append({
                    "Frame":             i + 1,          # last frame of window, 1-indexed
                    "SpikeRate":         round(spike_rate.item(), 6),
                    "MembranePotential": round(mem_pot.item(),    6),
                    "Logit":             round(logit.item(),      6),
                    "Prediction":        int(logit.item() > 0),
                })

        return pd.DataFrame(rows)

    def run_on_folder(self, input_dir, output_dir):
        """
        Run predict_video on every .mp4 in input_dir.
        Writes one CSV per video to output_dir.

        Call once for synthetic videos and once for real videos:
            snn.run_on_folder("videos/synthetic", "results/snn")
            snn.run_on_folder("videos/real",      "results/snn")

        Output CSV columns:
            Frame, SpikeRate, MembranePotential, Logit, Prediction
        """
        input_dir  = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        videos = sorted(input_dir.glob("*.mp4"))
        if not videos:
            print(f"[ERROR] No .mp4 files found in {input_dir}")
            return

        print(f"HybridSNN — {len(videos)} videos → {output_dir}/\n")

        for video_path in videos:
            try:
                df       = self.predict_video(video_path)
                out_path = output_dir / f"{video_path.stem}{self.SUFFIX}.csv"
                df.to_csv(out_path, index=False)

                n_pos = int(df["Prediction"].sum())
                print(f"  {video_path.name:45s}  "
                      f"sequences={len(df):>3}  "
                      f"predicted +ve: {n_pos}/{len(df)}")

            except ValueError as e:
                print(f"  [SKIP] {e}")

        print(f"\nCSVs saved to {output_dir}/")

    def evaluate(self, results_dir, label_csv):
        """
        Score all output CSVs in results_dir against ground truth labels.

        The SNN CSV has one row per 10-frame sequence, keyed on the last
        frame number. The label CSV has one row per frame. They are joined
        on Frame, so only frames with a sequence prediction are scored.

        Args:
            results_dir : folder containing *_snn_output.csv files
            label_csv   : ground truth CSV (columns: Filename, Frame, TrueLabel)

        Returns:
            dict — accuracy, precision, recall, f1, tp, fp, fn, tn
        """
        gt       = pd.read_csv(label_csv)
        all_pred = []
        all_true = []

        print(f"\nHybridSNN evaluation\n{'─' * 65}")

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

    # ── Internal ───────────────────────────────────────────────────────────────

    def _read_frames(self, video_path):
        """Read all frames as normalised float32 tensors (1, 64, 64)."""
        cap, frames = cv2.VideoCapture(str(video_path)), []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64)).astype(np.float32) / 255.0
            frames.append(
                torch.tensor(resized).unsqueeze(0)   # (1, 64, 64)
            )
        cap.release()
        return frames

    def _compute_metrics(self, y_true, y_pred):
        """Accuracy, precision, recall, F1, TP, FP, FN, TN."""
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
        description="HybridSNN — train or evaluate the spiking neural network."
    )
    parser.add_argument("--mode",        required=True, choices=["train", "eval"],
                        help="'train' to train the model, 'eval' to run inference")
    parser.add_argument("--input_dir",   required=True,
                        help="Folder of .mp4 videos")
    parser.add_argument("--label_csv",   required=True,
                        help="Ground truth CSV  e.g. labels/ground_truth.csv")
    parser.add_argument("--weights",     default="results/snn/weights.pth",
                        help="Path to save (train) or load (eval) weights  "
                             "(default: results/snn/weights.pth)")
    parser.add_argument("--output_dir",  default="results/snn",
                        help="[eval] Where to save output CSVs  "
                             "(default: results/snn)")
    parser.add_argument("--log_path",    default="results/snn/training_log.csv",
                        help="[train] Where to save training log  "
                             "(default: results/snn/training_log.csv)")
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--lr",          type=float, default=1e-3)
    args = parser.parse_args()

    snn = HybridSNN()

    if args.mode == "train":
        from data_loader import load_all_stimuli
        sequences, labels = load_all_stimuli(args.input_dir,
                                             split="train",
                                             window_size=HybridSNN.TIME_WINDOW)
        snn.train(sequences, labels,
                  epochs=args.epochs, lr=args.lr,
                  log_path=args.log_path)
        snn.save(args.weights)

    else:  # eval
        snn.load(args.weights)
        snn.run_on_folder(args.input_dir, args.output_dir)
        snn.evaluate(args.output_dir, args.label_csv)
