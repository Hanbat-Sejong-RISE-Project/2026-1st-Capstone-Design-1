from pathlib import Path
from collections import defaultdict
import hashlib
import json

import numpy as np


TX_MAT_DEFAULT = "/workspace/baseline/cap_sensing_OFDM_sym64_bw40.mat"
NO_SOURCE_MESSAGE = (
    "range_mag/range_phase 생성을 위해서는 송신 MAT의 tx_sym과 각 sample별 raw IQ(.dat)가 필요합니다."
)


def parse_raw_file(path):
    tokens = path.stem.split("_")
    if len(tokens) < 3 or not tokens[0].startswith("people"):
        return None
    try:
        people = int(tokens[0].replace("people", ""))
        ch = int(tokens[-1].replace("ch", ""))
    except Exception:
        return None
    if ch not in (0, 1):
        return None

    try_pos = next((i for i, token in enumerate(tokens) if token.startswith("try")), None)
    if try_pos is None:
        coords = tokens[1:-2]
        try_token = "try1"
        sample_idx = tokens[-2]
    else:
        coords = tokens[1:try_pos]
        try_token = tokens[try_pos]
        sample_idx = tokens[try_pos + 1] if try_pos + 1 < len(tokens) - 1 else "0"

    if people == 0:
        label = "empty"
        coords = []
    else:
        if len(coords) != people:
            return None
        label = "_".join(sorted(coords))

    return {
        "people": people,
        "ch": ch,
        "label": label,
        "coords": sorted(coords),
        "try": try_token,
        "sample_idx": sample_idx,
    }


def scan_iq_pairs(root, people=None):
    root = Path(root)
    people_values = [people] if people is not None else [0, 1, 2, 3]
    pairs = []
    for p in people_values:
        people_root = root / f"people{p}"
        if not people_root.exists():
            continue
        grouped = {}
        for path in sorted(people_root.rglob("*.dat")):
            info = parse_raw_file(path)
            if info is None or info["people"] != p:
                continue
            rel_parent = str(path.parent.relative_to(people_root))
            key = (rel_parent, info["label"], info["try"], info["sample_idx"])
            grouped.setdefault(
                key,
                {
                    "people": p,
                    "label": info["label"],
                    "coords": info["coords"],
                    "try": info["try"],
                    "sample_idx": info["sample_idx"],
                    "ch0": None,
                    "ch1": None,
                },
            )[f"ch{info['ch']}"] = path
        pairs.extend(item for item in grouped.values() if item["ch0"] and item["ch1"])
    return pairs


def inspect_dataset(root="/workspace/dataset/0529"):
    root = Path(root)
    print(f"[inspect_dataset] root: {root}")
    report = {"root": str(root), "raw_iq_pairs": {}, "has_raw_iq_pairs": False}
    for people in [0, 1, 2, 3]:
        pairs = scan_iq_pairs(root, people=people)
        report["raw_iq_pairs"][f"people{people}"] = len(pairs)
        print(f"  people{people} raw ch0/ch1 .dat pairs: {len(pairs)}")
    report["has_raw_iq_pairs"] = any(v > 0 for v in report["raw_iq_pairs"].values())
    if not report["has_raw_iq_pairs"]:
        print(NO_SOURCE_MESSAGE)
    return report


def load_iq_file(path):
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 2 != 0:
        raise ValueError(f"IQ float32 길이가 홀수입니다: {path}")
    return (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)


def iq_to_ofdm_freq_matrix(iq, nfft=4096, ncp=296, num_symbols=64, start_idx=0):
    iq = np.asarray(iq, dtype=np.complex64)
    sym_len = int(nfft) + int(ncp)
    need = start_idx + num_symbols * sym_len
    if iq.size < need:
        available = max(0, (iq.size - start_idx) // sym_len)
        raise ValueError(
            f"OFDM symbol 수가 부족합니다. need={num_symbols}, available={available}, "
            f"iq_samples={iq.size}, start_idx={start_idx}"
        )
    symbols = iq[start_idx:need].reshape(num_symbols, sym_len)
    no_cp = symbols[:, ncp:]
    return np.fft.fft(no_cp, n=nfft, axis=1).T.astype(np.complex64)


def load_tx_reference_mat(mat_path=TX_MAT_DEFAULT):
    try:
        import scipy.io as sio
    except Exception as exc:
        raise RuntimeError("scipy가 필요합니다. conda env khj에서 scipy 설치 여부를 확인하세요.") from exc

    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"tx mat 파일이 없습니다: {mat_path}")
    mat = sio.loadmat(mat_path)
    required = ["tx_sym", "N_fft", "L_cp", "N_sym"]
    missing = [name for name in required if name not in mat]
    if missing:
        raise KeyError(f"tx mat에 필수 변수가 없습니다: {missing}")

    nfft = int(np.asarray(mat["N_fft"]).squeeze())
    ncp = int(np.asarray(mat["L_cp"]).squeeze())
    num_symbols = int(np.asarray(mat["N_sym"]).squeeze())
    tx_time = np.asarray(mat["tx_sym"]).reshape(-1).astype(np.complex64)
    sym_len = nfft + ncp
    expected_len = sym_len * num_symbols
    if tx_time.size != expected_len:
        raise ValueError(f"tx_sym 길이가 맞지 않습니다. tx_len={tx_time.size}, expected={expected_len}")

    X_full = iq_to_ofdm_freq_matrix(tx_time, nfft=nfft, ncp=ncp, num_symbols=num_symbols)
    if "data_idx" in mat:
        active_indices = np.asarray(mat["data_idx"]).reshape(-1).astype(np.int64) - 1
    else:
        active_indices = np.where(np.mean(np.abs(X_full), axis=1) > 1e-6)[0]
    X_active = X_full[active_indices, :].astype(np.complex64)

    info = {
        "mat_path": str(mat_path),
        "nfft": nfft,
        "ncp": ncp,
        "num_symbols": num_symbols,
        "sym_len": sym_len,
        "tx_time": tx_time,
        "X_active": X_active,
        "active_indices": active_indices,
        "fs_mhz": float(np.asarray(mat.get("Fs", [[np.nan]])).squeeze()),
        "bw_mhz": float(np.asarray(mat.get("BW", [[np.nan]])).squeeze()),
    }
    print("[tx_mat] 송신 reference 확인")
    print(f"  path       : {mat_path}")
    print(f"  N_fft/L_cp : {nfft} / {ncp}")
    print(f"  N_sym      : {num_symbols}")
    print(f"  tx_sym     : {tx_time.shape}, CP 포함 시간영역 frame")
    print(f"  X_active   : {X_active.shape}, active subcarriers={active_indices.size}")
    print(f"  Fs/BW MHz  : {info['fs_mhz']} / {info['bw_mhz']}")
    return info


def find_frame_start_by_correlation(iq, tx_time, method="fft"):
    try:
        from scipy.signal import correlate
    except Exception as exc:
        raise RuntimeError("correlation frame sync에는 scipy.signal.correlate가 필요합니다.") from exc

    iq = np.asarray(iq, dtype=np.complex64)
    tx_time = np.asarray(tx_time, dtype=np.complex64)
    if iq.size < tx_time.size:
        raise ValueError(f"수신 IQ가 tx frame보다 짧습니다: iq={iq.size}, tx={tx_time.size}")
    corr = correlate(iq, tx_time, mode="valid", method=method)
    mag = np.abs(corr)
    start = int(np.argmax(mag))
    denom = np.linalg.norm(iq[start:start + tx_time.size]) * np.linalg.norm(tx_time) + 1e-12
    return start, {
        "corr_score": float(mag[start] / denom),
        "corr_peak_to_mean": float(mag[start] / (mag.mean() + 1e-12)),
        "corr_peak": float(mag[start]),
    }


def remove_modulation_effect(Y_rx, X_ref, eps=1e-8, mag_threshold=1e-6):
    Y_rx = np.asarray(Y_rx, dtype=np.complex64)
    X_ref = np.asarray(X_ref, dtype=np.complex64)
    if Y_rx.shape != X_ref.shape:
        raise ValueError(f"Y/X shape mismatch: {Y_rx.shape} vs {X_ref.shape}")
    mask = np.abs(X_ref) > mag_threshold
    return np.divide(Y_rx, X_ref + eps, out=np.zeros_like(Y_rx), where=mask).astype(np.complex64)


def make_H_full_from_active(H_active, active_indices, nfft):
    H_full = np.zeros((int(nfft), H_active.shape[1]), dtype=np.complex64)
    H_full[np.asarray(active_indices, dtype=np.int64), :] = H_active
    return H_full


def make_base_name(pair):
    digest = hashlib.md5(str(pair["ch0"]).encode()).hexdigest()[:8]
    return f"people{pair['people']}_{pair['label']}_{pair['try']}_{pair['sample_idx']}_{digest}"


def compute_H_full_with_tx_reference(received_path, tx_ref):
    rx_iq = load_iq_file(received_path)
    start_idx, sync = find_frame_start_by_correlation(rx_iq, tx_ref["tx_time"])
    Y_full = iq_to_ofdm_freq_matrix(
        rx_iq,
        nfft=tx_ref["nfft"],
        ncp=tx_ref["ncp"],
        num_symbols=tx_ref["num_symbols"],
        start_idx=start_idx,
    )
    active_indices = tx_ref["active_indices"]
    H_active = remove_modulation_effect(Y_full[active_indices, :], tx_ref["X_active"])
    H_full = make_H_full_from_active(H_active, active_indices, tx_ref["nfft"])
    return H_full, {"frame_start": int(start_idx), **sync}


def compute_background_H_streaming(dataset_root, tx_ref, rx_channel="ch1", max_samples=None, save_path=None):
    pairs = scan_iq_pairs(dataset_root, people=0)
    if max_samples is not None:
        pairs = pairs[: int(max_samples)]
    H_sum = None
    valid = 0
    skipped = 0
    sync_scores = []
    for idx, pair in enumerate(pairs, start=1):
        try:
            H_full, meta = compute_H_full_with_tx_reference(pair[rx_channel], tx_ref)
            if H_sum is None:
                H_sum = np.zeros_like(H_full, dtype=np.complex128)
            H_sum += H_full.astype(np.complex128)
            valid += 1
            sync_scores.append(meta.get("corr_score", np.nan))
        except Exception as exc:
            skipped += 1
            if skipped <= 5:
                print(f"[H_bg] skip people0 sample: {pair.get(rx_channel)} ({exc})")
        if idx == 1 or idx % 100 == 0:
            print(f"[H_bg] {idx}/{len(pairs)} valid={valid} skipped={skipped}")
    if valid == 0:
        raise RuntimeError("people0에서 유효한 H_bg sample을 하나도 만들지 못했습니다.")
    H_bg = (H_sum / valid).astype(np.complex64)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, H_bg)
        save_path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "valid_samples": valid,
                    "skipped_samples": skipped,
                    "rx_channel": rx_channel,
                    "shape": list(H_bg.shape),
                    "corr_score_mean": float(np.nanmean(sync_scores)) if sync_scores else None,
                    "corr_score_median": float(np.nanmedian(sync_scores)) if sync_scores else None,
                    "tx_mat": tx_ref.get("mat_path"),
                    "nfft": tx_ref.get("nfft"),
                    "ncp": tx_ref.get("ncp"),
                    "num_symbols": tx_ref.get("num_symbols"),
                    "active_subcarriers": int(len(tx_ref.get("active_indices", []))),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        print(f"[H_bg] saved: {save_path}")
    return H_bg


def normalize_array(out, normalize="none"):
    out = out.astype(np.float32)
    if normalize == "zscore":
        return ((out - out.mean()) / (out.std() + 1e-6)).astype(np.float32)
    if normalize == "minmax":
        return ((out - out.min()) / (out.max() - out.min() + 1e-6)).astype(np.float32)
    if normalize in (None, "none"):
        return out
    raise ValueError(f"normalize는 none/minmax/zscore 중 하나여야 합니다: {normalize}")


def make_range_mag_phase_from_H_dyn(H_dyn, nfft_range=4096, log_mag=False, normalize="none"):
    H_dyn = np.asarray(H_dyn, dtype=np.complex64)
    if H_dyn.ndim != 2:
        raise ValueError(f"H_dyn은 (subcarrier, symbol) 2D여야 합니다. got={H_dyn.shape}")
    h_range = np.fft.ifft(H_dyn, n=int(nfft_range), axis=0)
    range_mag = np.abs(h_range).astype(np.float32)
    if log_mag:
        range_mag = np.log1p(range_mag).astype(np.float32)
    range_phase = np.angle(h_range).astype(np.float32)
    range_mag = normalize_array(range_mag, normalize=normalize)
    range_phase = normalize_array(range_phase, normalize=normalize) if normalize not in (None, "none") else range_phase
    return range_mag[None, :, :].astype(np.float32), range_phase[None, :, :].astype(np.float32)


def process_dataset_range_mag_phase(
    root="/workspace/dataset/0529",
    tx_mat=TX_MAT_DEFAULT,
    out_mag_feature="range_mag",
    out_phase_feature="range_phase",
    h_bg_path=None,
    people_values=(1, 2, 3),
    rx_channel="ch1",
    nfft_range=4096,
    log_mag=False,
    normalize="none",
    overwrite=False,
    max_samples=None,
    background_max_samples=None,
):
    root = Path(root)
    report = inspect_dataset(root)
    if not report["has_raw_iq_pairs"]:
        raise RuntimeError(NO_SOURCE_MESSAGE)
    tx_ref = load_tx_reference_mat(tx_mat)
    h_bg_path = Path(h_bg_path) if h_bg_path else root / "H_bg_tx.npy"
    if h_bg_path.exists() and not overwrite:
        H_bg = np.load(h_bg_path).astype(np.complex64)
        print(f"[H_bg] load existing: {h_bg_path}, shape={H_bg.shape}")
    else:
        H_bg = compute_background_H_streaming(
            dataset_root=root,
            tx_ref=tx_ref,
            rx_channel=rx_channel,
            max_samples=background_max_samples,
            save_path=h_bg_path,
        )

    manifest = []
    for people in people_values:
        pairs = scan_iq_pairs(root, people=people)
        if max_samples is not None:
            pairs = pairs[: int(max_samples)]
        print(f"[range_mag_phase] people{people}: {len(pairs)} raw pairs")
        for idx, pair in enumerate(pairs, start=1):
            base = make_base_name(pair)
            mag_path = root / f"people{people}_{out_mag_feature}" / pair["label"] / f"{base}_{out_mag_feature}.npy"
            phase_path = root / f"people{people}_{out_phase_feature}" / pair["label"] / f"{base}_{out_phase_feature}.npy"
            error = ""
            meta = {}
            if mag_path.exists() and phase_path.exists() and not overwrite:
                status = "skip_exists"
            else:
                try:
                    H_full, meta = compute_H_full_with_tx_reference(pair[rx_channel], tx_ref)
                    range_mag, range_phase = make_range_mag_phase_from_H_dyn(
                        H_full - H_bg,
                        nfft_range=nfft_range,
                        log_mag=log_mag,
                        normalize=normalize,
                    )
                    mag_path.parent.mkdir(parents=True, exist_ok=True)
                    phase_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(mag_path, range_mag)
                    np.save(phase_path, range_phase)
                    status = "saved"
                except Exception as exc:
                    status = "failed"
                    error = str(exc)
                    if idx <= 10 or idx % 100 == 0:
                        print(f"  people{people} {idx}/{len(pairs)} failed: {pair.get(rx_channel)} ({error})")
            manifest.append(
                {
                    "people": people,
                    "label": pair["label"],
                    "base_id": base,
                    "received_path": str(pair.get(rx_channel)),
                    "rx_channel": rx_channel,
                    "tx_mat": str(tx_mat),
                    "h_bg_path": str(h_bg_path),
                    "range_mag_path": str(mag_path),
                    "range_phase_path": str(phase_path),
                    "status": status,
                    "error": error,
                    **meta,
                }
            )
            if idx == 1 or idx % 100 == 0:
                print(f"  people{people} {idx}/{len(pairs)} {status}: {mag_path}")
    manifest_path = root / f"{out_mag_feature}_{out_phase_feature}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[range_mag_phase] manifest saved: {manifest_path}")
    return manifest
