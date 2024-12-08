import neurokit2 as nk
import numpy as np
# from pytorch_forecasting.utils import autocorrelation
import torch

def _segment_signal(signal, window_lims, rpeaks):
    """
    Helper function to segment the ECG signal into individual beats.
    Args:
        signal (numpy.ndarray): ECG signal.
        window_lims (tuple): Tuple containing the left and right window limits.
        rpeaks (dict): Dictionary containing peak information.
    Returns:
        beats: List of beats.
        beat_windows: List of beat windows.
    """
    l, r = window_lims
    beat_windows = [(rpeak + l, rpeak + r) for rpeak in rpeaks['ECG_R_Peaks']]
    beats = [signal[:, l:r] for (l, r) in beat_windows]
    
    return beats, beat_windows


def _filter_beats(beats, beat_windows, rpeaks, range_cutoff=0.5, sd_cutoff=0.06, 
                 autocorr_cutoff=0.75):
    """
    Filters out beats based on range, standard deviation, and autocorrelation.
    Args:
        beats (list): List of beats.
        beat_windows (list): List of beat windows.
        rpeaks (dict): Dictionary containing peak information.
        range_cutoff (float, optional): Range cutoff value. Defaults to 0.5.
        sd_cutoff (float, optional): Standard deviation cutoff value. Defaults to 0.06.
        autocorr_cutoff (float, optional): Autocorrelation cutoff value. Defaults to 0.75.
    Returns:
        beats_filtered: np array of filtered beats.
        beat_windows_filtered: List of filtered beat windows.
    """

    # Filter out beats that are the first or last beat in the ECG signal
    beats = beats[1:-1]
    beats = torch.from_numpy(np.array(beats))
    beat_windows = beat_windows[1:-1]
    peak_types = [f"ECG_{peak}_Peaks" for peak in "PQRST"]

    assert len(beats) == len(beat_windows)

    beats_filtered = []
    dup_peaks = []

    for beat_window in beat_windows:
        l, r = beat_window

        for peak_type in peak_types:
            unique_peak_sum = 0

            for peak in rpeaks[peak_type]:
                if l < peak < r:
                    unique_peak_sum += 1

            if unique_peak_sum > 1:
                dup_peaks.append(0)
                break
        
        else:
            dup_peaks.append(1)
        

    for i, beat in enumerate(beats):
        # print(torch.max(beat, dim=1)[0] - torch.min(beat, dim=1)[0], torch.mean(torch.std(beat, dim=1)).item())
        if torch.mean(torch.max(beat, dim=1)[0] - torch.min(beat, dim=1)[0]).item() > range_cutoff:
            if torch.mean(torch.std(beat, dim=1)).item() > sd_cutoff:
                if dup_peaks[i] == 1:
                    beats_filtered.append(beat)
                # autocorrelated beats
                # autocorrs = autocorrelation(beat, dim=1)[:, :5]
                # autocorr_beat = torch.nanmean(torch.nanmean(autocorrs, dim=1), dim=0).item()
                # if autocorr_beat > autocorr_cutoff:
                #     beats_filtered.append(beat)

    return np.array(beats_filtered)


def segment_and_filter_ecg(signal, sampling_rate, window_lims=(-100, 200)):
    """
    Segments and filters out beats based on certain criteria.
    Args:
        signal (numpy.ndarray): ECG signal.
        sampling_rate (int): Sampling rate of the ECG signal.
        window_lims (tuple, optional): Tuple containing the left and right window limits.
    Returns:
        beats_filtered: Array of filtered beats.
    """
    # Identify beats
    _, rpeaks = nk.ecg_process(signal[0], sampling_rate=sampling_rate)

    # Identify beat segments
    beats, beat_windows = _segment_signal(signal, window_lims, rpeaks)
    # print(len(beats))

    # Filter beats
    beats_filtered = _filter_beats(
        beats, beat_windows, rpeaks
    )
    
    return beats_filtered