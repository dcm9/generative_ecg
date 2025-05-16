import neurokit2
import jax
import jax.numpy
from typing import Any, List, Tuple, Dict

def get_peaks(x_signal: jax.numpy.ndarray, sampling_rate: int = 500) -> Dict[str, Any]:
    """
    Detect ECG peaks in the input signal using neurokit2.

    Args:
        x_signal (array-like): ECG signal, shape (n_channels, n_timesteps).
        sampling_rate (int, optional): Sampling rate of the ECG signal. Default is 500.

    Returns:
        dict: Dictionary containing detected ECG peaks (R, P, Q, S, T, etc.).
    """
    _, x_peaks = neurokit2.ecg_process(x_signal[0], sampling_rate=sampling_rate)
    return x_peaks

def segment(
    x_signal: jax.numpy.ndarray, 
    x_peaks: Dict[str, Any], 
    tmax: int = 400
) -> Tuple[List[jax.numpy.ndarray], List[Tuple[int, int]]]:
    """
    Segment the ECG signal into individual beats based on detected R-peaks.

    Args:
        x_signal (array-like): ECG signal, shape (n_channels, n_timesteps).
        x_peaks (dict): Dictionary containing detected ECG peaks.
        tmax (int, optional): Length of each beat segment (in samples). Default is 400.

    Returns:
        tuple: (x_beats, x_beat_windows)
            x_beats (list): List of segmented beats, each of shape (n_channels, tmax).
            x_beat_windows (list): List of (start, end) indices for each beat segment.
    """
    l, r = 0 - tmax // 2, tmax - tmax // 2
    rpeaks = x_peaks['ECG_R_Peaks']
    
    x_beat_windows = [(rpeak + l, rpeak + r) for rpeak in rpeaks]
    x_beats = [x_signal[:, l:r] for (l, r) in x_beat_windows]

    return x_beats, x_beat_windows

def filter_beats(
    x_beats: List[jax.numpy.ndarray],
    y_signal: jax.numpy.ndarray,
    x_beat_windows: List[Tuple[int, int]],
    x_peaks: Dict[str, Any],
    drop_first: bool = True,
    drop_last: bool = True,
    range_min: float = 0.5,
    sd_min: float = 0.06,
    autocorr_min: float = 0.75
) -> Tuple[jax.numpy.ndarray, jax.numpy.ndarray]:
    """
    Filter out ECG beats based on range, standard deviation, and duplicate peaks.

    Args:
        x_beats (list): List of ECG beats, each of shape (n_channels, tmax).
        y_signal (Any): Label or target value for the beats.
        x_beat_windows (list): List of (start, end) indices for each beat segment.
        x_peaks (dict): Dictionary containing detected ECG peaks.
        drop_first (bool, optional): Whether to drop the first beat. Default is True.
        drop_last (bool, optional): Whether to drop the last beat. Default is True.
        range_min (float, optional): Minimum mean range required for a beat to be kept. Default is 0.5.
        sd_min (float, optional): Minimum mean standard deviation required for a beat to be kept. Default is 0.06.
        autocorr_min (float, optional): Minimum autocorrelation threshold (not used in current implementation). Default is 0.75.

    Returns:
        tuple: (beats_filtered, y_filtered)
            beats_filtered (jax.numpy.ndarray): Array of filtered beats.
            y_filtered (jax.numpy.ndarray): Array of corresponding labels.
    """
    # Filter out beats that are the first or last beat in the ECG signal
    if drop_first:
        x_beats = x_beats[1:]
        x_beat_windows = x_beat_windows[1:]

    if drop_last:
        x_beats = x_beats[:-1]
        x_beat_windows = x_beat_windows[:-1]

    peaks = 'PQRST'
    peak_types = [f"ECG_{peak}_Peaks" for peak in peaks]

    dup_peaks = [
        any(sum(1 for peak in x_peaks[peak_type] if l < peak < r) > 1 
            for peak_type in peak_types) 
        for (l, r) in x_beat_windows
    ]

    filtered_beats = []

    for i, beat in enumerate(x_beats):
        # Check range condition
        beat_range = jax.numpy.mean(jax.numpy.max(beat, axis=1) - jax.numpy.min(beat, axis=1)).item()
        if beat_range > range_min:
            # Check standard deviation condition
            beat_std = jax.numpy.mean(jax.numpy.std(beat, axis=1)).item()
            if beat_std > sd_min:
                if dup_peaks[i]:
                    filtered_beats.append(beat)

    if filtered_beats:
        return jax.numpy.stack(filtered_beats), jax.numpy.array([y_signal] * len(filtered_beats))

    return jax.numpy.array([]), jax.numpy.array([])