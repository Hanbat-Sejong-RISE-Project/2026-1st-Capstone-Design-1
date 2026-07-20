import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ofdm_radar_features import TX_MAT_DEFAULT, process_dataset_range_mag_phase


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="/workspace/dataset/0529")
    parser.add_argument("--tx_mat", type=str, default=TX_MAT_DEFAULT)
    parser.add_argument("--out_mag_feature", type=str, default="range_mag")
    parser.add_argument("--out_phase_feature", type=str, default="range_phase")
    parser.add_argument("--h_bg_path", type=str, default=None)
    parser.add_argument("--people", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--rx_channel", type=str, default="ch1", choices=["ch0", "ch1"])
    parser.add_argument("--nfft_range", type=int, default=4096)
    parser.add_argument("--normalize", type=str, default="none", choices=["none", "minmax", "zscore"])
    parser.add_argument("--log_mag", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--background_max_samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    process_dataset_range_mag_phase(
        root=args.dataset_root,
        tx_mat=args.tx_mat,
        out_mag_feature=args.out_mag_feature,
        out_phase_feature=args.out_phase_feature,
        h_bg_path=args.h_bg_path,
        people_values=args.people,
        rx_channel=args.rx_channel,
        nfft_range=args.nfft_range,
        log_mag=args.log_mag,
        normalize=args.normalize,
        overwrite=args.overwrite,
        max_samples=args.max_samples,
        background_max_samples=args.background_max_samples,
    )


if __name__ == "__main__":
    main()
