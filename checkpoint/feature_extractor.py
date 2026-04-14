from __future__ import annotations

import re

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, welch
from scipy.stats import kurtosis, skew

FS = 256
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}
LBP_SCALES = [{"P": 8, "R": 1}, {"P": 8, "R": 2}]

LEFT_CH = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1"]
RIGHT_CH = ["FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8", "P8-O2"]
COMMON_CHANNELS = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
]


def preprocess_window(data: np.ndarray, fs: int = FS) -> np.ndarray:
    matrix = np.asarray(data, dtype=np.float32).copy()
    if matrix.ndim != 2:
        raise ValueError("Expected 2D EEG matrix with shape (channels, samples).")

    nyq = fs / 2.0
    low = 0.5 / nyq
    high = 40.0 / nyq
    b_band, a_band = butter(4, [low, high], btype="band")
    b_notch, a_notch = iirnotch(60.0, 30.0, fs)

    for index in range(matrix.shape[0]):
        signal = matrix[index].astype(np.float64)
        signal = filtfilt(b_band, a_band, signal)
        signal = filtfilt(b_notch, a_notch, signal)
        matrix[index] = signal.astype(np.float32)

    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    std[std < 1e-6] = 1e-6
    normalized = (matrix - mean) / std
    return normalized.astype(np.float32)


def _make_uniform_map(p: int) -> np.ndarray:
    code_count = 2**p
    mapping = np.zeros(code_count, dtype=np.int32)
    uniform_index = 0
    for code in range(code_count):
        transitions = 0
        for bit in range(p):
            current = (code >> bit) & 1
            nxt = (code >> ((bit + 1) % p)) & 1
            transitions += int(current != nxt)
        if transitions <= 2:
            mapping[code] = uniform_index
            uniform_index += 1
        else:
            mapping[code] = p * (p - 1) + 2
    return mapping


UNIFORM_MAPS = {scale["P"]: _make_uniform_map(scale["P"]) for scale in LBP_SCALES}


def compute_lbp_1d(signal: np.ndarray, p: int, r: int) -> np.ndarray:
    vector = np.asarray(signal, dtype=np.float32)
    length = vector.shape[0]
    half = p // 2
    pad = half * r
    padded = np.pad(vector, pad_width=pad, mode="reflect")
    center = padded[pad : pad + length]

    codes = np.zeros(length, dtype=np.int32)
    for bit in range(p):
        offset = -(bit + 1) * r if bit < half else (bit - half + 1) * r
        start = pad + offset
        neighbor = padded[start : start + length]
        codes += (neighbor >= center).astype(np.int32) << bit

    codes = np.clip(codes, 0, (2**p) - 1)
    uniform = UNIFORM_MAPS[p][codes]
    bins = p * (p - 1) + 3
    hist, _ = np.histogram(uniform, bins=bins, range=(0, bins))
    total = float(hist.sum())
    if total <= 0:
        return hist.astype(np.float32)
    return (hist.astype(np.float32) / total).astype(np.float32)


def bandpass_1d(signal: np.ndarray, low: float, high: float, fs: int = FS, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    lo = max(low / nyq, 1e-4)
    hi = min(high / nyq, 0.999)
    if lo >= hi:
        return np.asarray(signal, dtype=np.float32)
    try:
        b, a = butter(order, [lo, hi], btype="band")
        return filtfilt(b, a, np.asarray(signal, dtype=np.float64)).astype(np.float32)
    except Exception:
        return np.asarray(signal, dtype=np.float32)


def compute_psd(signal: np.ndarray, fs: int = FS) -> np.ndarray:
    vector = np.asarray(signal, dtype=np.float32)
    nperseg = min(int(2 * fs), vector.shape[0])
    if nperseg < 8:
        return np.zeros(len(BANDS), dtype=np.float32)
    freqs, psd = welch(vector, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, window="hann")
    total = np.trapezoid(psd, freqs)
    if total <= 1e-10:
        return np.zeros(len(BANDS), dtype=np.float32)

    values: list[float] = []
    for low, high in BANDS.values():
        index = (freqs >= low) & (freqs <= high)
        if np.any(index):
            values.append(float(np.trapezoid(psd[index], freqs[index]) / total))
        else:
            values.append(0.0)
    return np.asarray(values, dtype=np.float32)


def compute_stats(signal: np.ndarray) -> np.ndarray:
    vector = np.asarray(signal, dtype=np.float32)
    count = max(vector.shape[0], 1)
    mean = float(np.mean(vector))
    std = float(np.std(vector))
    if std <= 1e-10:
        return np.zeros(8, dtype=np.float32)

    sk = float(skew(vector))
    kt = float(kurtosis(vector))
    rms = float(np.sqrt(np.mean(vector**2)))
    zcr = float(np.sum(np.diff(np.sign(vector)) != 0) / count)

    d1 = np.diff(vector)
    d2 = np.diff(d1)
    v0 = float(np.var(vector))
    v1 = float(np.var(d1))
    v2 = float(np.var(d2))
    mobility = float(np.sqrt(v1 / v0)) if v0 > 1e-10 else 0.0
    complexity = float((np.sqrt(v2 / v1) / mobility)) if (v1 > 1e-10 and mobility > 1e-10) else 0.0
    return np.asarray([mean, std, sk, kt, rms, zcr, mobility, complexity], dtype=np.float32)


def normalize_channel(name: str) -> str:
    value = str(name).strip().upper().replace(" ", "").replace("_", "-")
    value = re.sub(r"-\d+$", "", value)
    return value


def compute_connectivity(data: np.ndarray, ch_names: list[str]) -> np.ndarray:
    matrix = np.asarray(data, dtype=np.float32)
    channel_count = matrix.shape[0]
    try:
        corr = np.corrcoef(matrix)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        idx = np.triu_indices(channel_count, k=1)
        upper = corr[idx]
        corr_features = np.asarray(
            [
                float(np.mean(upper)),
                float(np.std(upper)),
                float(np.max(np.abs(upper))),
            ],
            dtype=np.float32,
        )
    except Exception:
        corr_features = np.zeros(3, dtype=np.float32)

    normalized_names = [normalize_channel(name) for name in ch_names]
    left = {normalize_channel(name) for name in LEFT_CH}
    right = {normalize_channel(name) for name in RIGHT_CH}
    left_idx = [index for index, name in enumerate(normalized_names) if name in left]
    right_idx = [index for index, name in enumerate(normalized_names) if name in right]

    asym = np.zeros(len(BANDS), dtype=np.float32)
    if left_idx and right_idx:
        left_mean = matrix[left_idx].mean(axis=0)
        right_mean = matrix[right_idx].mean(axis=0)
        for idx, (low, high) in enumerate(BANDS.values()):
            left_band = bandpass_1d(left_mean, low, high)
            right_band = bandpass_1d(right_mean, low, high)
            left_power = float(np.mean(left_band**2))
            right_power = float(np.mean(right_band**2))
            denom = left_power + right_power
            asym[idx] = ((left_power - right_power) / denom) if denom > 1e-10 else 0.0

    return np.concatenate([corr_features, asym]).astype(np.float32)


def extract_features(data: np.ndarray, ch_names: list[str]) -> np.ndarray:
    matrix = np.asarray(data, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Expected 2D EEG matrix with shape (channels, samples).")
    if matrix.shape[0] != len(COMMON_CHANNELS):
        raise ValueError(
            f"Expected {len(COMMON_CHANNELS)} channels for checkpoint feature extraction, got {matrix.shape[0]}."
        )

    lbp_all: list[np.ndarray] = []
    psd_all: list[np.ndarray] = []
    stats_all: list[np.ndarray] = []

    for channel in range(matrix.shape[0]):
        signal = matrix[channel].astype(np.float32)
        lbp_channel: list[np.ndarray] = []
        for low, high in BANDS.values():
            band_signal = bandpass_1d(signal, low, high)
            for scale in LBP_SCALES:
                lbp_channel.append(compute_lbp_1d(band_signal, p=scale["P"], r=scale["R"]))
        lbp_all.append(np.concatenate(lbp_channel).astype(np.float32))
        psd_all.append(compute_psd(signal))
        stats_all.append(compute_stats(signal))

    lbp_arr = np.stack(lbp_all, axis=0).astype(np.float32)
    psd_arr = np.stack(psd_all, axis=0).astype(np.float32)
    stats_arr = np.stack(stats_all, axis=0).astype(np.float32)
    conn = compute_connectivity(matrix, ch_names)

    features = np.concatenate(
        [
            lbp_arr.mean(axis=0),
            lbp_arr.std(axis=0),
            psd_arr.mean(axis=0),
            psd_arr.std(axis=0),
            stats_arr.mean(axis=0),
            stats_arr.std(axis=0),
            conn,
        ]
    ).astype(np.float32)
    return features
