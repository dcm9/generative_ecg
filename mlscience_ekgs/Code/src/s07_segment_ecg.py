import neurokit2 as nk
import numpy as np
from pytorch_forecasting.utils import autocorrelation
import torch


WINDOW_LIMS = (-100, 200)


def segment_signal(signal, window_lims, rpeaks):
    l, r = window_lims
    beat_windows = [(rpeak + l, rpeak + r) for rpeak in rpeaks['ECG_R_Peaks']]
    beats = [signal[:, l:r] for (l, r) in beat_windows]
    
    return beats, beat_windows


def identify_duplicate_peaks(rpeaks, beat_windows):
    """
    Identifies windows that contain more than one of any kind of peak 
    (P, Q, R, S, T).
    """
    peak_types = [f"ECG_{peak}_Peaks" for peak in "PQRST"]
    dup_peaks = [
        any(sum(1 for peak in rpeaks[peak_type] if l < peak < r) > 1 
            for peak_type in peak_types) 
        for (l, r) in beat_windows
    ]
    
    return dup_peaks


def filter_beats(beats, beat_windows, rpeaks, range_cutoff=0.5, sd_cutoff=0.06, 
                 autocorr_cutoff=0.75):
    # Filter out beats that are the first or last beat in the ECG signal
    beats = beats[1:-1]
    beats = torch.from_numpy(np.array(beats))
    beat_windows = beat_windows[1:-1]
    
    # Identify duplicate peaks
    duplicate_peaks = identify_duplicate_peaks(rpeaks, beat_windows)
    
    # Identify low range beats
    max_vals = [torch.max(beat, dim=1)[0] for beat in beats]
    min_vals = [torch.min(beat, dim=1)[0] for beat in beats]
    ecg_range_per_lead = [
        max_val - min_val for max_val, min_val in zip(max_vals, min_vals)
    ]
    mean_range = [torch.mean(r).item() for r in ecg_range_per_lead]
    low_range_flag = [r <= range_cutoff for r in mean_range]
    
    # Identify low sd beats
    beat_sds = [
        round(torch.mean(torch.std(beat, dim=1)).item(), 4) 
        for beat in beats
    ]
    low_sd_flag = [sd <= sd_cutoff for sd in beat_sds]
    
    # Identify low autocorrelation beats
    beat_autocorr_per_lead = [
        autocorrelation(beat, dim=1)[:, :5] for beat in beats
    ]
    autocorr_beats = [
        torch.nanmean(torch.nanmean(autocorrs, dim=1), dim=0).item() 
        for autocorrs in beat_autocorr_per_lead
    ]
    low_ac_flag = [autocorr <= autocorr_cutoff for autocorr in autocorr_beats]
    
    all_flags = zip(duplicate_peaks, low_range_flag, low_sd_flag, low_ac_flag)
    beat_flags = [dup or low_r or low_sd or low_ac 
                  for dup, low_r, low_sd, low_ac in all_flags]
    
    beats_filtered = np.array(
        [beat for i, beat in enumerate(beats) if not beat_flags[i]]
    )
    beat_windows_filtered = [window for i, window in enumerate(beat_windows) 
                             if not beat_flags[i]]
    
    return beats_filtered, beat_windows_filtered


def segment_and_filter_ecg(signal, sampling_rate, window_lims=WINDOW_LIMS):
    # Identify beats
    _, rpeaks = nk.ecg_process(signal[0], sampling_rate=sampling_rate)
    
    # Identify beat segments
    beats, beat_windows = segment_signal(signal, window_lims, rpeaks)
    
    # Filter beats
    beats_filtered, _ = filter_beats(
        beats, beat_windows, rpeaks
    )
    
    return beats_filtered