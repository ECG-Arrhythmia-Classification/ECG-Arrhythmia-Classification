"""Export one RAW 180-sample heartbeat as a CSV accepted by ECG Studio."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_LENGTH = 180
CLASS_NAMES = {
    0: ("N", "Normal group"),
    1: ("S", "Supraventricular ectopic"),
    2: ("V", "Ventricular ectopic"),
    3: ("F", "Fusion beat"),
    4: ("Q", "Unknown / paced / other"),
}


def export_heartbeat(
    split: str, index: int, output: str | Path, *, repo_root: Path = REPO_ROOT,
    overwrite: bool = False,
) -> tuple[Path, int]:
    """Read the shared RAW split without filtering, resampling or modifying it."""
    if split not in ("train", "val", "test"):
        raise ValueError("split phải là train, val hoặc test.")
    if index < 0:
        raise ValueError("index phải lớn hơn hoặc bằng 0.")
    data_dir = repo_root / "processed_data" / "split"
    # Reject pickle/object arrays and mmap large splits instead of copying them.
    X = np.load(data_dir / f"X_{split}.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(data_dir / f"y_{split}.npy", mmap_mode="r", allow_pickle=False)
    if X.ndim != 2 or X.shape[1] != INPUT_LENGTH:
        raise ValueError(f"X_{split} phải có shape (N, {INPUT_LENGTH}); nhận {X.shape}.")
    if y.ndim != 1 or len(X) != len(y):
        raise ValueError("y phải có shape (N,) và có cùng số hàng với X.")
    if index >= len(X):
        raise ValueError(f"index {index} vượt giới hạn; split có {len(X)} heartbeat.")
    if X.dtype.kind not in "iuf" or y.dtype.kind not in "iuf":
        raise ValueError("Tín hiệu và nhãn phải là mảng số thực.")
    signal = np.asarray(X[index], dtype=np.float64)
    if not np.isfinite(signal).all():
        raise ValueError("Heartbeat được chọn chứa NaN hoặc Infinity.")
    label = float(y[index])
    if not np.isfinite(label) or label not in CLASS_NAMES:
        raise ValueError("Nhãn heartbeat phải là 0=N, 1=S, 2=V, 3=F hoặc 4=Q.")
    target = Path(output)
    if not target.is_absolute():
        target = repo_root / target
    target = target.resolve()
    if target.suffix.lower() != ".csv":
        raise ValueError("Đường dẫn output phải có đuôi .csv.")
    if target.exists() and not overwrite:
        raise FileExistsError(f"Tệp đã tồn tại: {target}; dùng --force nếu muốn ghi đè.")
    target.parent.mkdir(parents=True, exist_ok=True)
    # Export only waveform values. Do not include record IDs or patient metadata.
    np.savetxt(target, signal, delimiter=",", header="signal", comments="", fmt="%.17g")
    return target, int(label)


def main() -> None:
    # Keep Vietnamese CLI messages usable with Windows redirected output too.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="Ghi đè CSV đã tồn tại.")
    args = parser.parse_args()
    try:
        target, label = export_heartbeat(args.split, args.index, args.output, overwrite=args.force)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    code, meaning = CLASS_NAMES[label]
    print(f"Đã xuất {INPUT_LENGTH} mẫu RAW: {target}")
    print(f"Nhãn tham chiếu: {label} = {code} ({meaning}); đây không phải kết quả dự đoán.")


if __name__ == "__main__":
    main()
