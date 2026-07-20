from pathlib import Path
from collections import defaultdict
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset


VALID_FEATURES = {"range_mag", "range_phase"}
MAX_PEOPLE = 3
SUPPORTED_PEOPLE = (0, 1, 2, 3)


def strip_feature_suffix(path: Path, feature: str) -> str:
    suffix = f"_{feature}"
    stem = path.stem
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def coord_to_xy(coord: str):
    coord = str(coord)
    if len(coord) != 2:
        raise ValueError(f"좌표 문자열은 2자리여야 합니다: {coord}")
    return float(coord[0]), float(coord[1])


def label_to_target(label: str, people: int):
    coords = label.split("_") if label else []
    if len(coords) != people:
        raise ValueError(f"people{people}인데 label={label}, coords={coords}")
    target = []
    for coord in sorted(coords):
        x, y = coord_to_xy(coord)
        target.extend([x, y])
    return np.asarray(target, dtype=np.float32)


def label_to_padded_target(label: str, people: int, max_people: int = MAX_PEOPLE):
    target = np.zeros(max_people * 2, dtype=np.float32)
    coord_mask = np.zeros(max_people * 2, dtype=np.bool_)
    people = int(people)
    if people == 0:
        return target, coord_mask
    flat = label_to_target(label, people)
    target[: people * 2] = flat
    coord_mask[: people * 2] = True
    return target, coord_mask


def load_range_feature(path: Path):
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"range feature는 2D 또는 3D여야 합니다. got={arr.shape}, path={path}")
    return np.ascontiguousarray(arr, dtype=np.float32)


def resize_chw(arr, target_size):
    if target_size is None:
        return arr
    tensor = torch.from_numpy(arr).unsqueeze(0).float()
    tensor = F.interpolate(tensor, size=tuple(target_size), mode="bilinear", align_corners=False)
    return tensor.squeeze(0).numpy().astype(np.float32)


class SinglePeopleRangeDataset(Dataset):
    def __init__(self, root, people, features=("range_mag", "range_phase"), normalize_input=False, target_size=(256, 256)):
        self.root = Path(root)
        self.people = int(people)
        self.features = list(features)
        self.normalize_input = bool(normalize_input)
        self.target_size = tuple(target_size) if target_size is not None else None
        if self.people not in SUPPORTED_PEOPLE:
            raise ValueError(f"people은 0,1,2,3만 지원합니다: {self.people}")
        for feature in self.features:
            if feature not in VALID_FEATURES:
                raise ValueError(f"지원하지 않는 feature입니다: {feature}")
        self.samples = self._collect_samples()
        if not self.samples:
            raise RuntimeError(f"유효한 샘플이 없습니다. people={self.people}, features={self.features}")

    def _find_feature_root(self, feature):
        candidates = [self.root / f"people{self.people}_{feature}"]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"feature 폴더가 없습니다: {candidates[0]}")

    def _collect_samples(self):
        feature_maps = []
        for feature in self.features:
            feature_root = self._find_feature_root(feature)
            print(f"[RangeDataset] people{self.people} feature root for {feature}: {feature_root}")
            fmap = {}
            for path in sorted(feature_root.rglob("*.npy")):
                label = "empty" if self.people == 0 else path.parent.name
                base_id = strip_feature_suffix(path, feature)
                fmap[(label, base_id)] = path
            feature_maps.append(fmap)
        common_keys = set(feature_maps[0].keys())
        for fmap in feature_maps[1:]:
            common_keys.intersection_update(fmap.keys())
        samples = []
        for label, base_id in sorted(common_keys):
            target, coord_mask = label_to_padded_target(label, self.people)
            feature_paths = {feature: fmap[(label, base_id)] for feature, fmap in zip(self.features, feature_maps)}
            samples.append({"people": self.people, "label": label, "base_id": base_id, "feature_paths": feature_paths, "target": target, "coord_mask": coord_mask})
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        arrays = []
        for feature in self.features:
            arr = load_range_feature(item["feature_paths"][feature])
            arr = resize_chw(arr, self.target_size)
            arrays.append(arr)
        x = np.concatenate(arrays, axis=0)
        if self.normalize_input:
            x = (x - x.mean()) / (x.std() + 1e-6)
        return {
            "x": torch.from_numpy(x).float(),
            "y": torch.from_numpy(item["target"]).float(),
            "coord_mask": torch.from_numpy(item["coord_mask"]).bool(),
            "count": torch.tensor(item["people"], dtype=torch.long),
            "label": item["label"],
            "base_id": item["base_id"],
        }


class MultiPeopleRangeDataset(Dataset):
    def __init__(self, root, features=("range_mag", "range_phase"), normalize_input=False, target_size=(256, 256), people_values=(1, 2, 3), skip_missing_people=True):
        self.root = Path(root)
        self.features = list(features)
        self.normalize_input = bool(normalize_input)
        self.target_size = tuple(target_size) if target_size is not None else None
        self.people_values = tuple(int(v) for v in people_values)
        self.datasets = {}
        self.index_map = []
        self.samples = []
        self.count_sizes = {}
        for people in self.people_values:
            try:
                dataset = SinglePeopleRangeDataset(self.root, people, self.features, self.normalize_input, self.target_size)
            except (FileNotFoundError, RuntimeError) as exc:
                if skip_missing_people:
                    print(f"[RangeDataset] skip people{people}: {exc}")
                    continue
                raise
            self.datasets[people] = dataset
            self.count_sizes[people] = len(dataset)
            for local_idx, sample in enumerate(dataset.samples):
                self.index_map.append((people, local_idx))
                self.samples.append({"people": people, "label": sample["label"], "base_id": sample["base_id"]})
        if not self.index_map:
            raise RuntimeError(f"통합 Dataset에 유효한 샘플이 없습니다: {self.root}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        people, local_idx = self.index_map[idx]
        return self.datasets[people][local_idx]


def multitask_split_indices(dataset, val_ratio=0.15, test_ratio=0.15, seed=42):
    rng = random.Random(seed)
    by_group = defaultdict(list)
    for idx, sample in enumerate(dataset.samples):
        people = int(sample["people"])
        label = sample["label"] if people > 0 else f"empty_{idx}"
        by_group[(people, label)].append(idx)
    groups_by_people = defaultdict(list)
    for group in by_group:
        groups_by_people[group[0]].append(group)
    train_indices, val_indices, test_indices = [], [], []
    for people, groups in groups_by_people.items():
        groups = groups[:]
        rng.shuffle(groups)
        if people == 0:
            all_idx = [idx for group in groups for idx in by_group[group]]
            rng.shuffle(all_idx)
            n = len(all_idx)
            n_test = int(round(n * test_ratio))
            n_val = int(round(n * val_ratio))
            test_indices.extend(all_idx[:n_test])
            val_indices.extend(all_idx[n_test:n_test + n_val])
            train_indices.extend(all_idx[n_test + n_val:])
            continue
        n = len(groups)
        n_test = max(1, int(round(n * test_ratio))) if n >= 3 else 0
        n_val = max(1, int(round(n * val_ratio))) if n >= 3 else 0
        if n_test + n_val >= n:
            n_test = max(0, min(n_test, n - 2))
            n_val = max(0, min(n_val, n - n_test - 1))
        for group in groups[n_test + n_val:]:
            train_indices.extend(by_group[group])
        for group in groups[n_test:n_test + n_val]:
            val_indices.extend(by_group[group])
        for group in groups[:n_test]:
            test_indices.extend(by_group[group])
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    return train_indices, val_indices, test_indices


def collate_multitask_fn(batch):
    return {
        "x": torch.stack([b["x"] for b in batch], dim=0),
        "y": torch.stack([b["y"] for b in batch], dim=0),
        "coord_mask": torch.stack([b["coord_mask"] for b in batch], dim=0),
        "count": torch.stack([b["count"] for b in batch], dim=0),
        "label": [b["label"] for b in batch],
        "base_id": [b["base_id"] for b in batch],
    }


def make_multitask_dataloaders(root, features=("range_mag", "range_phase"), batch_size=8, val_ratio=0.15, test_ratio=0.15, seed=42, num_workers=4, normalize_input=False, target_size=(256, 256), people_values=(1, 2, 3)):
    dataset = MultiPeopleRangeDataset(root, features, normalize_input, target_size, people_values)
    train_idx, val_idx, test_idx = multitask_split_indices(dataset, val_ratio, test_ratio, seed)
    train_set, val_set, test_set = Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)
    kwargs = {"num_workers": num_workers, "pin_memory": torch.cuda.is_available(), "collate_fn": collate_multitask_fn, "persistent_workers": num_workers > 0}
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **kwargs)
    sample_x = dataset[0]["x"]
    in_channels = sample_x.shape[0]
    count_sizes = {p: dataset.count_sizes.get(p, 0) for p in SUPPORTED_PEOPLE}
    print("[RangeDataset]")
    print(f"  people_values: {people_values}")
    print(f"  features     : {features}")
    print(f"  sample x     : {tuple(sample_x.shape)}")
    print(f"  in_channels  : {in_channels}")
    print(f"  count_sizes  : {count_sizes}")
    print(f"  total        : {len(dataset)}")
    print(f"  train/val/test: {len(train_set)} / {len(val_set)} / {len(test_set)}")
    return {"dataset": dataset, "train_loader": train_loader, "val_loader": val_loader, "test_loader": test_loader, "in_channels": in_channels, "train_size": len(train_set), "val_size": len(val_set), "test_size": len(test_set), "count_sizes": count_sizes}
