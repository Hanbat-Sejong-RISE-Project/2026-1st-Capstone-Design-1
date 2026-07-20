import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_rd import make_multitask_dataloaders


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="/workspace/dataset/0529")
    parser.add_argument("--features", nargs="+", default=["range_mag", "range_phase"], choices=["range_mag", "range_phase"])
    parser.add_argument("--people_values", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--target_size", nargs=2, type=int, default=[256, 256])
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    loaders = make_multitask_dataloaders(
        root=args.dataset_root,
        features=args.features,
        people_values=args.people_values,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=args.target_size,
    )
    batch = next(iter(loaders["train_loader"]))
    print(f"batch x shape : {tuple(batch['x'].shape)}")
    print(f"batch y shape : {tuple(batch['y'].shape)}")
    print(f"coord mask    : {tuple(batch['coord_mask'].shape)}")
    print(f"count example : {batch['count'].tolist()}")
    print(f"label example : {batch['label'][:4]}")
    print(f"base_id example: {batch['base_id'][:4]}")


if __name__ == "__main__":
    main()
