import argparse
import csv
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn

from dataset_rd import make_multitask_dataloaders
from models_rd import MAX_PEOPLE, NUM_COUNT_CLASSES, build_multitask_resnet

GRID_TO_CM = 35.0
FEATURE_CHOICES = ["range_mag", "range_phase"]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_name(text):
    text = str(text).replace("/", "_").replace("\\", "_")
    return re.sub(r"[^0-9a-zA-Z가-힣_.\-]+", "_", text)


def build_count_loss(loaders, use_weights, device):
    if not use_weights:
        return nn.CrossEntropyLoss()
    counts = np.array([loaders["count_sizes"].get(i, 0) for i in range(NUM_COUNT_CLASSES)], dtype=np.float64)
    weights = np.zeros_like(counts, dtype=np.float64)
    nonzero = counts > 0
    weights[nonzero] = counts[nonzero].sum() / (nonzero.sum() * counts[nonzero])
    weights[~nonzero] = 0.0
    print(f"count class weights: {weights.tolist()}")
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))


def location_metrics_update(pred, target, people, acc):
    if pred.numel() == 0:
        return
    diff = pred - target
    acc["abs"] += diff.abs().sum().item()
    acc["sq"] += diff.pow(2).sum().item()
    acc["coord"] += diff.numel()
    pred_xy = pred.view(-1, people, 2)
    target_xy = target.view(-1, people, 2)
    dist = torch.linalg.vector_norm(pred_xy - target_xy, dim=-1)
    acc["dist"] += dist.sum().item()
    acc["person"] += dist.numel()
    acc["sample"] += pred.shape[0]


def location_metrics_result(acc, prefix):
    coord = max(acc["coord"], 1)
    person = max(acc["person"], 1)
    return {
        f"{prefix}_mae": acc["abs"] / coord,
        f"{prefix}_rmse": (acc["sq"] / coord) ** 0.5,
        f"{prefix}_mean_dist": acc["dist"] / person,
        f"{prefix}_mean_dist_cm": acc["dist"] / person * GRID_TO_CM,
        f"{prefix}_samples": acc["sample"],
    }


def train_one_epoch(model, loader, optimizer, device, count_loss_fn, cls_w, loc_w, scaler=None, amp=False, grad_clip=None):
    model.train()
    totals = {"loss": 0.0, "count_loss": 0.0, "location_loss": 0.0, "samples": 0, "loc_samples": 0, "count_correct": 0}
    loc_acc = {"abs": 0.0, "sq": 0.0, "coord": 0, "dist": 0.0, "person": 0, "sample": 0}
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        count_target = batch["count"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            outputs = model(x)
            features = outputs["features"]
            count_logits = outputs["count_logits"]
            count_loss = count_loss_fn(count_logits, count_target)
            location_loss_num = features.sum() * 0.0
            location_sample_count = 0
            metric_groups = []
            for people in range(1, MAX_PEOPLE + 1):
                mask = count_target == people
                n_group = int(mask.sum().item())
                if n_group == 0:
                    continue
                pred = model.predict_location_head(features[mask], people)
                target = y[mask, : people * 2]
                per_sample_mse = (pred - target).pow(2).mean(dim=1)
                location_loss_num = location_loss_num + per_sample_mse.sum()
                location_sample_count += n_group
                metric_groups.append((pred.detach(), target.detach(), people))
            location_loss = location_loss_num / location_sample_count if location_sample_count else features.sum() * 0.0
            loss = cls_w * count_loss + loc_w * location_loss
        if amp:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        bs = x.shape[0]
        totals["samples"] += bs
        totals["loss"] += loss.item() * bs
        totals["count_loss"] += count_loss.item() * bs
        totals["count_correct"] += (count_logits.argmax(dim=1) == count_target).sum().item()
        if location_sample_count:
            totals["location_loss"] += location_loss.item() * location_sample_count
            totals["loc_samples"] += location_sample_count
            for pred, target, people in metric_groups:
                location_metrics_update(pred, target, people, loc_acc)
    result = {
        "loss": totals["loss"] / totals["samples"],
        "count_loss": totals["count_loss"] / totals["samples"],
        "location_loss": totals["location_loss"] / max(totals["loc_samples"], 1),
        "count_accuracy": totals["count_correct"] / totals["samples"],
    }
    result.update(location_metrics_result(loc_acc, "location"))
    return result


@torch.no_grad()
def evaluate(model, loader, device, count_loss_fn, cls_w, loc_w, return_predictions=False):
    model.eval()
    totals = {"loss": 0.0, "count_loss": 0.0, "location_loss": 0.0, "samples": 0, "loc_samples": 0, "count_correct": 0}
    confusion = np.zeros((NUM_COUNT_CLASSES, NUM_COUNT_CLASSES), dtype=np.int64)
    oracle_acc = {"abs": 0.0, "sq": 0.0, "coord": 0, "dist": 0.0, "person": 0, "sample": 0}
    routed_acc = {"abs": 0.0, "sq": 0.0, "coord": 0, "dist": 0.0, "person": 0, "sample": 0}
    collected = {"count_logits": [], "count_preds": [], "count_targets": [], "oracle_location_preds": [], "routed_location_preds": [], "location_targets": [], "coord_masks": [], "labels": [], "base_ids": []}
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        coord_mask = batch["coord_mask"].to(device, non_blocking=True)
        count_target = batch["count"].to(device, non_blocking=True)
        outputs = model(x)
        features = outputs["features"]
        count_logits = outputs["count_logits"]
        count_pred = count_logits.argmax(dim=1)
        count_loss = count_loss_fn(count_logits, count_target)
        bs = x.shape[0]
        oracle_padded = torch.full((bs, MAX_PEOPLE * 2), float("nan"), dtype=features.dtype, device=device)
        location_loss_num = features.sum() * 0.0
        location_sample_count = 0
        for people in range(1, MAX_PEOPLE + 1):
            mask = count_target == people
            n_group = int(mask.sum().item())
            if n_group == 0:
                continue
            pred = model.predict_location_head(features[mask], people)
            target = y[mask, : people * 2]
            oracle_padded[mask, : people * 2] = pred
            location_loss_num = location_loss_num + (pred - target).pow(2).mean(dim=1).sum()
            location_sample_count += n_group
            location_metrics_update(pred, target, people, oracle_acc)
        location_loss = location_loss_num / location_sample_count if location_sample_count else features.sum() * 0.0
        loss = cls_w * count_loss + loc_w * location_loss
        routed_padded = model.route_locations(features, count_pred)
        for people in range(1, MAX_PEOPLE + 1):
            mask = (count_target == people) & (count_pred == people)
            if mask.any():
                location_metrics_update(routed_padded[mask, : people * 2], y[mask, : people * 2], people, routed_acc)
        totals["samples"] += bs
        totals["loss"] += loss.item() * bs
        totals["count_loss"] += count_loss.item() * bs
        totals["count_correct"] += (count_pred == count_target).sum().item()
        if location_sample_count:
            totals["location_loss"] += location_loss.item() * location_sample_count
            totals["loc_samples"] += location_sample_count
        for t, p in zip(count_target.cpu().numpy(), count_pred.cpu().numpy()):
            confusion[int(t), int(p)] += 1
        if return_predictions:
            collected["count_logits"].append(count_logits.cpu().numpy())
            collected["count_preds"].append(count_pred.cpu().numpy())
            collected["count_targets"].append(count_target.cpu().numpy())
            collected["oracle_location_preds"].append(oracle_padded.cpu().numpy())
            collected["routed_location_preds"].append(routed_padded.cpu().numpy())
            collected["location_targets"].append(y.cpu().numpy())
            collected["coord_masks"].append(coord_mask.cpu().numpy())
            collected["labels"].extend(batch["label"])
            collected["base_ids"].extend(batch["base_id"])
    result = {
        "loss": totals["loss"] / totals["samples"],
        "count_loss": totals["count_loss"] / totals["samples"],
        "location_loss": totals["location_loss"] / max(totals["loc_samples"], 1),
        "count_accuracy": totals["count_correct"] / totals["samples"],
        "confusion_matrix": confusion,
    }
    result.update(location_metrics_result(oracle_acc, "oracle_location"))
    result.update(location_metrics_result(routed_acc, "correct_count_location"))
    if return_predictions:
        for key in ["count_logits", "count_preds", "count_targets", "oracle_location_preds", "routed_location_preds", "location_targets", "coord_masks"]:
            result[key] = np.concatenate(collected[key], axis=0)
        result["labels"] = collected["labels"]
        result["base_ids"] = collected["base_ids"]
    return result


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_loss, args, in_channels):
    torch.save({
        "model_type": "multitask_resnet",
        "features": args.features,
        "people_values": args.people_values,
        "target_size": args.target_size,
        "in_channels": in_channels,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "args": vars(args),
    }, path)


def save_predictions_csv(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["base_id", "label", "count_true", "count_pred", "target", "oracle_pred", "routed_pred"])
        for i, base_id in enumerate(result["base_ids"]):
            writer.writerow([
                base_id,
                result["labels"][i],
                int(result["count_targets"][i]),
                int(result["count_preds"][i]),
                result["location_targets"][i].tolist(),
                result["oracle_location_preds"][i].tolist(),
                result["routed_location_preds"][i].tolist(),
            ])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="/workspace/dataset/0529")
    parser.add_argument("--features", nargs="+", default=["range_mag", "range_phase"], choices=FEATURE_CHOICES)
    parser.add_argument("--people_values", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--target_size", nargs=2, type=int, default=[256, 256])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--classification_loss_weight", type=float, default=1.0)
    parser.add_argument("--localization_loss_weight", type=float, default=1.0)
    parser.add_argument("--use_count_class_weights", action="store_true")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="/workspace/khj/0614/runs")
    parser.add_argument("--normalize_input", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    run_name = safe_name(args.run_name or f"multitask_{'-'.join(args.features)}")
    run_dir = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2))
    loaders = make_multitask_dataloaders(
        root=args.dataset_root,
        features=args.features,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        normalize_input=args.normalize_input,
        target_size=args.target_size,
        people_values=args.people_values,
    )
    model = build_multitask_resnet(loaders["in_channels"], args.base_channels, args.dropout).to(args.device)
    count_loss_fn = build_count_loss(loaders, args.use_count_class_weights, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    amp_enabled = args.amp and str(args.device).startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_val_loss = float("inf")
    history_path = run_dir / "history.csv"
    with history_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_count_acc", "train_loc_dist", "val_loss", "val_count_acc", "val_oracle_dist", "elapsed_time"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            train_result = train_one_epoch(model, loaders["train_loader"], optimizer, args.device, count_loss_fn, args.classification_loss_weight, args.localization_loss_weight, scaler, amp_enabled, args.grad_clip)
            val_result = evaluate(model, loaders["val_loader"], args.device, count_loss_fn, args.classification_loss_weight, args.localization_loss_weight)
            scheduler.step(val_result["loss"])
            elapsed = time.time() - start
            writer.writerow({
                "epoch": epoch,
                "train_loss": train_result["loss"],
                "train_count_acc": train_result["count_accuracy"],
                "train_loc_dist": train_result["location_mean_dist"],
                "val_loss": val_result["loss"],
                "val_count_acc": val_result["count_accuracy"],
                "val_oracle_dist": val_result["oracle_location_mean_dist"],
                "elapsed_time": elapsed,
            })
            f.flush()
            print(
                f"epoch {epoch:03d}/{args.epochs} | train total {train_result['loss']:.5f}, "
                f"count acc {train_result['count_accuracy']:.4f}, loc dist {train_result['location_mean_dist']:.4f} | "
                f"val total {val_result['loss']:.5f}, count acc {val_result['count_accuracy']:.4f}, "
                f"oracle dist {val_result['oracle_location_mean_dist']:.4f} | elapsed {elapsed:.1f}s"
            )
            save_checkpoint(run_dir / "last.pt", model, optimizer, scheduler, epoch, best_val_loss, args, loaders["in_channels"])
            if val_result["loss"] < best_val_loss:
                best_val_loss = val_result["loss"]
                save_checkpoint(run_dir / "best.pt", model, optimizer, scheduler, epoch, best_val_loss, args, loaders["in_channels"])
                print(f"  -> best saved: {run_dir / 'best.pt'}")
    checkpoint = torch.load(run_dir / "best.pt", map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_result = evaluate(model, loaders["test_loader"], args.device, count_loss_fn, args.classification_loss_weight, args.localization_loss_weight, return_predictions=True)
    np.savez(
        run_dir / "test_predictions.npz",
        count_logits=test_result["count_logits"],
        count_preds=test_result["count_preds"],
        count_targets=test_result["count_targets"],
        oracle_location_preds=test_result["oracle_location_preds"],
        routed_location_preds=test_result["routed_location_preds"],
        location_targets=test_result["location_targets"],
        coord_masks=test_result["coord_masks"],
        labels=np.asarray(test_result["labels"]),
        base_ids=np.asarray(test_result["base_ids"]),
        confusion_matrix=test_result["confusion_matrix"],
    )
    save_predictions_csv(run_dir / "test_predictions.csv", test_result)
    summary = {k: v for k, v in test_result.items() if k not in {"count_logits", "count_preds", "count_targets", "oracle_location_preds", "routed_location_preds", "location_targets", "coord_masks", "labels", "base_ids", "confusion_matrix"}}
    summary["confusion_matrix"] = test_result["confusion_matrix"].tolist()
    summary["best_val_loss"] = best_val_loss
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("MULTITASK TEST")
    print(f"count_accuracy: {test_result['count_accuracy']:.4f}")
    print(f"oracle_location_mean_dist: {test_result['oracle_location_mean_dist']:.4f} grid ({test_result['oracle_location_mean_dist_cm']:.2f} cm)")
    print(f"correct_count_location_mean_dist: {test_result['correct_count_location_mean_dist']:.4f} grid ({test_result['correct_count_location_mean_dist_cm']:.2f} cm)")
    print("confusion matrix [true, pred]:")
    print(test_result["confusion_matrix"])
    print(f"saved: {run_dir}")


if __name__ == "__main__":
    main()
