import argparse
import csv
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def parse_vector(text):
    text = text.strip().strip("[]")
    if not text:
        return np.array([], dtype=np.float32)
    vals = []
    for part in text.split(","):
        part = part.strip()
        vals.append(float("nan") if part.lower() == "nan" else float(part))
    return np.asarray(vals, dtype=np.float32)


def people_from_base_id(base_id):
    m = re.match(r"people(\d+)_", base_id)
    if not m:
        raise ValueError(f"base_id에서 people 수를 찾지 못했습니다: {base_id}")
    return int(m.group(1))


def strip_suffix(path, feature):
    suffix = f"_{feature}"
    stem = path.stem
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def find_feature_path(dataset_root, people, label, base_id, feature):
    root = Path(dataset_root) / f"people{people}_{feature}" / label
    exact = root / f"{base_id}_{feature}.npy"
    if exact.exists():
        return exact
    if root.exists():
        matches = sorted(root.glob(f"{base_id}*.npy"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"feature 파일을 찾지 못했습니다: {exact}")


def load_model_feature_image(path, target_size):
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"feature shape가 2D/3D가 아닙니다: {arr.shape}, path={path}")
    tensor = torch.from_numpy(arr).unsqueeze(0).float()
    tensor = F.interpolate(tensor, size=tuple(target_size), mode="bilinear", align_corners=False)
    return tensor.squeeze(0).numpy()[0]


def valid_people_points(vec, people):
    pts = vec[: people * 2].reshape(people, 2)
    return pts[~np.isnan(pts).any(axis=1)]


def mean_dist(pred_pts, target_pts):
    n = min(len(pred_pts), len(target_pts))
    if n == 0:
        return float("nan")
    return float(np.linalg.norm(pred_pts[:n] - target_pts[:n], axis=1).mean())


def draw_points(ax, pts, marker, color, label, text_prefix):
    if len(pts) == 0:
        return
    ax.scatter(pts[:, 0], pts[:, 1], s=120, marker=marker, c=color, linewidths=2.5, label=label)
    for i, (x, y) in enumerate(pts, start=1):
        ax.text(x + 0.08, y + 0.12, f"{text_prefix}{i}", color=color, fontsize=8, weight="bold")


def draw_one(row, dataset_root, target_size, out_path):
    base_id = row["base_id"]
    label = row["label"]
    true_count = int(row["count_true"])
    pred_count = int(row["count_pred"])
    people = people_from_base_id(base_id)
    if people != true_count:
        people = true_count

    mag_path = find_feature_path(dataset_root, people, label, base_id, "range_mag")
    phase_path = find_feature_path(dataset_root, people, label, base_id, "range_phase")
    mag = load_model_feature_image(mag_path, target_size)
    phase = load_model_feature_image(phase_path, target_size)

    target = parse_vector(row["target"])
    oracle = parse_vector(row["oracle_pred"])
    routed = parse_vector(row["routed_pred"])
    target_pts = valid_people_points(target, true_count)
    oracle_pts = valid_people_points(oracle, true_count)
    routed_pts = valid_people_points(routed, pred_count)
    oracle_err = mean_dist(oracle_pts, target_pts)
    routed_err = mean_dist(routed_pts, target_pts)

    fig = plt.figure(figsize=(15, 7), dpi=160)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 1.0], height_ratios=[1, 1])

    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(mag, aspect="auto", origin="lower", cmap="magma")
    ax0.set_title("Model input: range_mag")
    ax0.set_xlabel("OFDM symbol / resized")
    ax0.set_ylabel("Range bin / resized")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.03)

    ax1 = fig.add_subplot(gs[1, 0])
    im1 = ax1.imshow(phase, aspect="auto", origin="lower", cmap="twilight", vmin=-math.pi, vmax=math.pi)
    ax1.set_title("Model input: range_phase")
    ax1.set_xlabel("OFDM symbol / resized")
    ax1.set_ylabel("Range bin / resized")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.03)

    ax2 = fig.add_subplot(gs[:, 1:])
    ax2.set_title("Multitask / Multi-head Prediction")
    ax2.set_xlim(-0.5, 8.5)
    ax2.set_ylim(-0.5, 8.5)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xticks(range(9))
    ax2.set_yticks(range(9))
    ax2.grid(True, linewidth=0.7, alpha=0.35)
    ax2.invert_yaxis()

    draw_points(ax2, target_pts, "o", "#1f77b4", "target", "T")
    draw_points(ax2, oracle_pts, "^", "#ff7f0e", "oracle head", "O")
    draw_points(ax2, routed_pts, "x", "#d62728", "routed inference", "R")
    for pts, color in [(oracle_pts, "#ff7f0e"), (routed_pts, "#d62728")]:
        n = min(len(pts), len(target_pts))
        for i in range(n):
            ax2.plot([target_pts[i, 0], pts[i, 0]], [target_pts[i, 1], pts[i, 1]], color=color, linestyle="--", linewidth=1.0, alpha=0.75)
    ax2.legend(loc="upper right")

    info = (
        f"base_id: {base_id}\n"
        f"label: {label}\n"
        f"count true/pred: {true_count} / {pred_count}\n"
        f"oracle error: {oracle_err:.3f} grid = {oracle_err * 35:.1f} cm\n"
        f"routed error: {routed_err:.3f} grid = {routed_err * 35:.1f} cm\n"
        f"mag: {mag_path.name}\n"
        f"phase: {phase_path.name}"
    )
    fig.suptitle("Range Mag/Phase Inference Result", fontsize=14, weight="bold")
    fig.text(0.52, 0.04, info, ha="left", va="bottom", fontsize=9, family="monospace")
    fig.tight_layout(rect=[0.0, 0.14, 1.0, 0.94])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def read_rows(csv_path):
    with Path(csv_path).open(newline="") as f:
        return list(csv.DictReader(f))


def select_rows(rows, mode, count):
    def routed_error(row):
        true_count = int(row["count_true"])
        pred_count = int(row["count_pred"])
        target_pts = valid_people_points(parse_vector(row["target"]), true_count)
        routed_pts = valid_people_points(parse_vector(row["routed_pred"]), pred_count)
        err = mean_dist(routed_pts, target_pts)
        return float("inf") if math.isnan(err) else err

    if mode == "first":
        return rows[:count]
    if mode == "best":
        return sorted(rows, key=routed_error)[:count]
    if mode == "worst":
        return sorted(rows, key=routed_error, reverse=True)[:count]
    if mode == "mixed":
        ordered = sorted(rows, key=routed_error)
        n_best = count // 3
        n_worst = count // 3
        best = ordered[:n_best]
        worst = ordered[-n_worst:] if n_worst else []
        mid_count = count - len(best) - len(worst)
        start = max(0, len(ordered) // 2 - mid_count // 2)
        mid = ordered[start:start + mid_count]
        return best + mid + worst
    raise ValueError(mode)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="/workspace/dataset/0529")
    parser.add_argument("--run_dir", type=str, default="/workspace/khj/0614/runs/multitask_range_mag_phase")
    parser.add_argument("--pred_csv", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--target_size", nargs=2, type=int, default=[256, 256])
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--mode", choices=["first", "best", "worst", "mixed"], default="mixed")
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    pred_csv = Path(args.pred_csv) if args.pred_csv else run_dir / "test_predictions.csv"
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "inference_visuals"
    rows = read_rows(pred_csv)
    chosen = select_rows(rows, args.mode, args.count)
    print(f"prediction rows: {len(rows)}")
    print(f"selected rows: {len(chosen)} mode={args.mode}")
    print(f"out_dir: {out_dir}")
    for idx, row in enumerate(chosen, start=1):
        out_name = f"{idx:03d}_ct{row['count_true']}_cp{row['count_pred']}_{row['label']}_{row['base_id']}.png"
        draw_one(row, args.dataset_root, args.target_size, out_dir / out_name)
        if idx == 1 or idx % 10 == 0:
            print(f"saved {idx}/{len(chosen)}: {out_dir / out_name}")


if __name__ == "__main__":
    main()
