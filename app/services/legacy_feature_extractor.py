from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
from scipy import signal as scipy_signal

from preprocessing import PreparedEEGSegment


class ChbMitLegacyFeatureExtractor:
    def __init__(self) -> None:
        step_module = importlib.import_module("step1_extract_features")
        preprocess_module = importlib.import_module("chb_mit_preprocess")

        self._extract_glcm_lbp = getattr(step_module, "extract_glcm_lbp_separated")
        self._read_edf = getattr(preprocess_module, "read_edf")
        self._select_22ch = getattr(preprocess_module, "select_22ch")
        self._stft_22ch = getattr(preprocess_module, "stft_22ch")

        self.target_sampling_rate = int(getattr(preprocess_module, "FS", 256))
        self.segment_seconds = int(getattr(preprocess_module, "SEG_SEC", 5))
        self.expected_channels = int(getattr(preprocess_module, "N_CH", 22))
        self.target_samples = self.target_sampling_rate * self.segment_seconds

    def extract(
        self,
        *,
        prepared_segment: PreparedEEGSegment,
        feature_set: str,
        edf_path: str | Path | None = None,
    ) -> np.ndarray:
        if edf_path is None:
            raise ValueError("edf_path is required for CHB-MIT legacy feature extraction.")

        raw, ch_names, source_rate = self._read_edf(str(edf_path))
        mapped_22ch, matched = self._select_22ch(raw, ch_names)
        if matched == 0:
            raise ValueError("Could not map any channels into the 22-channel CHB-MIT bipolar montage.")

        segment = self._slice_segment(
            mapped_22ch,
            source_rate=source_rate,
            start_sec=float(prepared_segment.start_sec),
        )
        if source_rate != self.target_sampling_rate:
            segment = self._resample_channels(
                segment,
                source_rate=float(source_rate),
                target_rate=float(self.target_sampling_rate),
            )

        segment = self._fit_segment_length(segment, self.target_samples)
        if segment.shape[0] != self.expected_channels:
            raise ValueError(f"Expected {self.expected_channels} channels, got {segment.shape[0]}.")

        sample_3d = self._stft_22ch(segment)
        glcm_features: list[np.ndarray] = []
        lbp_features: list[np.ndarray] = []
        for channel_index in range(sample_3d.shape[0]):
            glcm_vec, lbp_vec = self._extract_glcm_lbp(sample_3d[channel_index])
            glcm_features.append(np.asarray(glcm_vec, dtype=np.float32))
            lbp_features.append(np.asarray(lbp_vec, dtype=np.float32))

        glcm = np.concatenate(glcm_features).astype(np.float32, copy=False)
        lbp = np.concatenate(lbp_features).astype(np.float32, copy=False)
        key = feature_set.upper()
        if key == "GLCM":
            return glcm
        if key == "LBP":
            return lbp
        if key == "COMBINED":
            return np.concatenate([glcm, lbp]).astype(np.float32, copy=False)
        raise ValueError(f"Unsupported feature_set '{feature_set}'.")

    def _slice_segment(self, data_22ch: np.ndarray, *, source_rate: float, start_sec: float) -> np.ndarray:
        n_samples = data_22ch.shape[1]
        if n_samples <= 0:
            raise ValueError("EDF has no signal samples.")

        start = int(round(max(start_sec, 0.0) * float(source_rate)))
        start = min(start, max(n_samples - 1, 0))
        end = start + int(round(self.segment_seconds * float(source_rate)))
        if end <= n_samples:
            return data_22ch[:, start:end]

        # If the requested window reaches the recording tail, shift backward to keep full segment length.
        start = max(0, n_samples - int(round(self.segment_seconds * float(source_rate))))
        end = n_samples
        return data_22ch[:, start:end]

    def _fit_segment_length(self, segment: np.ndarray, target_samples: int) -> np.ndarray:
        if segment.shape[1] == target_samples:
            return segment.astype(np.float32, copy=False)
        if segment.shape[1] > target_samples:
            return segment[:, :target_samples].astype(np.float32, copy=False)

        padded = np.zeros((segment.shape[0], target_samples), dtype=np.float32)
        padded[:, : segment.shape[1]] = segment.astype(np.float32, copy=False)
        return padded

    def _resample_channels(self, segment: np.ndarray, *, source_rate: float, target_rate: float) -> np.ndarray:
        source_rounded = max(int(round(source_rate)), 1)
        target_rounded = max(int(round(target_rate)), 1)
        gcd = np.gcd(source_rounded, target_rounded) or 1
        return scipy_signal.resample_poly(
            segment,
            up=target_rounded // gcd,
            down=source_rounded // gcd,
            axis=1,
        ).astype(np.float32, copy=False)
