import numpy as np

from app.config import runtime_config
from app.services.eeg_intake import ChannelTrace, EEGIntakeService
from app.services.preprocessing import SignalPreprocessingService


def _bipolar_trace(name: str, phase: float) -> ChannelTrace:
    signal = np.sin(np.linspace(0.0, 12.0, runtime_config.default_window_size, dtype=np.float32) + phase)
    return ChannelTrace(
        source_name=name,
        canonical_name=name,
        sampling_rate=float(runtime_config.target_sampling_rate_hz),
        signal=signal.astype(np.float32),
    )


def test_bipolar_conversion_builds_model_ready_intake():
    service = EEGIntakeService(runtime_config)
    channel_names = ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1", "Fp2-F8", "F8-T4", "T4-T6", "T6-O2"]
    trace_map = {
        name: _bipolar_trace(name, index * 0.2)
        for index, name in enumerate(channel_names)
    }

    intake = service._finalize_intake(".edf", 60.0, channel_names, trace_map)
    failure_message = service._validate_for_analysis(intake)

    assert intake.input_montage_type == "bipolar"
    assert intake.conversion_status == "converted"
    assert failure_message is None
    assert intake.validation_status == "VALIDATED"
    assert "F3" in intake.traces_by_channel
    assert "Cz" in intake.traces_by_channel
    assert len(intake.approximated_channels) >= 1

    preprocessed = SignalPreprocessingService(runtime_config).prepare(intake)
    assert preprocessed.mapped_signal.shape[0] == len(runtime_config.required_channel_order)
    assert preprocessed.segments.shape[1] == len(runtime_config.required_channel_order)


def test_bipolar_conversion_blocks_when_coverage_is_insufficient():
    service = EEGIntakeService(runtime_config)
    channel_names = ["Fp1-F7", "F7-T3", "Fp2-F8", "F8-T4"]
    trace_map = {
        name: _bipolar_trace(name, index * 0.2)
        for index, name in enumerate(channel_names)
    }

    intake = service._finalize_intake(".edf", 60.0, channel_names, trace_map)
    failure_message = service._validate_for_analysis(intake)

    assert intake.input_montage_type == "bipolar"
    assert intake.validation_status == "BLOCKED"
    assert failure_message is not None
