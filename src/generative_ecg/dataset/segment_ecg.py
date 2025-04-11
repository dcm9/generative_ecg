import neurokit2
import jax
import jax.numpy

def get_peaks(x_signal, sampling_rate=500):
    _, x_peaks = neurokit2.ecg_process(x_signal[0], sampling_rate=sampling_rate)

    return x_peaks

def segment(x_signal, x_peaks, tmax=400):
    l, r = 0 - tmax // 2, tmax - tmax // 2
    rpeaks = x_peaks['ECG_R_Peaks']
    
    x_beat_windows = [(rpeak + l, rpeak + r) for rpeak in rpeaks]
    x_beats = [x_signal[:, l:r] for (l, r) in x_beat_windows]

    return x_beats, x_beat_windows

def filter_beats(x_beats, y_signal, x_beat_windows, x_peaks, drop_first=True, drop_last=True, range_min=0.5, sd_min=0.06, autocorr_min=0.75):
    """
    Filters out beats based on range, standard deviation, and autocorrelation.
    Args:
        beats (list): JAX array of beats.
        beat_windows (list): JAX array of beat windows.
        rpeaks (dict): Dictionary containing peak information.
        range_cutoff (float, optional): Range cutoff value. 
        sd_cutoff (float, optional): Standard deviation cutoff value. 
        autocorr_cutoff (float, optional): Autocorrelation cutoff value. 
    Returns:
        beats_filtered: JAX array of filtered beats.
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