import neurokit2
# from pytorch_forecasting.utils import autocorrelation
import torch

def _segment_signal(signal, window_lims, rpeaks):
    """
    Helper function to segment the ECG signal into individual beats.
    Args:
        signal (torch.tensor): ECG signal.
        window_lims (tuple): Tuple containing the left and right window limits.
        rpeaks (dict): Dictionary containing peak information.
    Returns:
        beats: torch tensor of beats.
        beat_windows: torch tensor of beat windows.
    """
    l, r = window_lims
    beat_windows = [(rpeak + l, rpeak + r) for rpeak in rpeaks]
    beats = [signal[:, l:r] for (l, r) in beat_windows]
    
    return torch.tensor(beats), torch.tensor(beat_windows)

def _identify_duplicate_peaks(beat_windows, rpeaks, peaks):
    """
    Filters out duplicate peaks by checking if there is more than one of each peak type
    in a given beat window.
    Args:
        beat_windows (list): List of beat windows.
        rpeaks (dict): Dictionary containing peak information.
        peak_types (list): List of peak types.
    Returns:
        dup_peaks: List of duplicate peaks.
    """
    peak_types = [f"ECG_{peak}_Peaks" for peak in peaks]

    dup_peaks = [
        any(sum(1 for peak in rpeaks[peak_type] if l < peak < r) > 1 
            for peak_type in peak_types) 
        for (l, r) in beat_windows
    ]

    return dup_peaks

def _filter_beats(beats, beat_windows, rpeaks, range_cutoff, sd_cutoff, 
                 autocorr_cutoff):
    """
    Filters out beats based on range, standard deviation, and autocorrelation.
    Args:
        beats (list): torch tensor of beats.
        beat_windows (list): torch tensor of beat windows.
        rpeaks (dict): Dictionary containing peak information.
        range_cutoff (float, optional): Range cutoff value. 
        sd_cutoff (float, optional): Standard deviation cutoff value. 
        autocorr_cutoff (float, optional): Autocorrelation cutoff value. 
    Returns:
        beats_filtered: torch tensor of filtered beats.
    """

    # Filter out beats that are the first or last beat in the ECG signal
    beats = beats[1:-1]
    beat_windows = beat_windows[1:-1]
    # beats = torch.from_numpy(np.array(beats))
    
    assert len(beats) == len(beat_windows)

    dup_peaks = _identify_duplicate_peaks(beat_windows, rpeaks, peaks='PQRST')

    beats_filtered = torch.tensor([])

    for i, beat in enumerate(beats):
        # save difference as variable 
        if torch.mean(torch.max(beat, dim=1)[0] - torch.min(beat, dim=1)[0]).item() > range_cutoff:
            # save mean as variable
            if torch.mean(torch.std(beat, dim=1)).item() > sd_cutoff:
                if not dup_peaks[i]:
                    beats_filtered  = torch.cat((beats_filtered, beat))
                # autocorrelated beats
                # autocorrs = autocorrelation(beat, dim=1)[:, :5]
                # autocorr_beat = torch.nanmean(torch.nanmean(autocorrs, dim=1), dim=0).item()
                # if autocorr_beat > autocorr_cutoff:
                #     beats_filtered.append(beat)

    return beats_filtered


def segment_and_filter_ecg(signal, sampling_rate, window_lims=(-100, 200)):
    """
    Segments and filters out beats based on certain criteria.
    Args:
        signal (numpy.ndarray): ECG signal.
        sampling_rate (int): Sampling rate of the ECG signal.
        window_lims (tuple, optional): Tuple containing the left and right window limits.
    Returns:
        beats_filtered: torch tensor of filtered beats.
    """
    # Identify beats
    _, rpeaks = neurokit2.ecg_process(signal[0], sampling_rate=sampling_rate)

    # Identify beat segments based on R-peaks
    beats, beat_windows = _segment_signal(signal, window_lims, rpeaks['ECG_R_Peaks'])

    # Filter beats
    beats_filtered = _filter_beats(
        beats, beat_windows, rpeaks, range_cutoff=0.5, sd_cutoff=0.06, 
        autocorr_cutoff=0.75
    )
    
    return beats_filtered