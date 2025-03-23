import neurokit2
import jax
import jax.numpy

def get_peaks(x_signal, sampling_rate=500):
    _, x_peaks = neurokit2.ecg_process(x_signal[0], sampling_rate=sampling_rate)

    return x_peaks

def process(x_signal, x_peaks, y_signal, tmax=400):
    # TODO: ENSURE WINDOW LIMS ACTUALLY MEANS TMAX
    l, r = 0 - tmax // 2, tmax - tmax // 2
    rpeaks = x_peaks['ECG_R_Peaks']
    
    x_beat_windows = jax.numpy.array([(rpeak + l, rpeak + r) for rpeak in rpeaks])
    x_beats = jax.numpy.array([x_signal[:, l:r] for (l, r) in beat_windows])

    peaks = 'PQRST'
    peak_types = [f"ECG_{peak}_Peaks" for peak in peaks]

    dup_peaks = [
        any(sum(1 for peak in x_peaks[peak_type] if l < peak < r) > 1 
            for peak_type in peak_types) 
        for (l, r) in x_beat_windows
    ]

    x_beats = x_beats[~jax.numpy.array(dup_peaks)]
    y_beats = y_signal[~jax.numpy.array(dup_peaks)]

    return x_beats, y_beats

def filter_beats(x_beats, y_beats, beat_windows, rpeaks, drop_first=True, drop_last=True, range_min=0.5, sd_min=0.06, autocorr_min=0.75):
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
        beats = beats[1:]
        beat_windows = beat_windows[1:]

    if drop_last:
        beats = beats[:-1]
        beat_windows = beat_windows[:-1]

    # dup_peaks = identify_duplicate_peaks(beat_windows, rpeaks, peaks='PQRST')

    filtered_beats = []

    for i, beat in enumerate(beats):
        # Check range condition
        beat_range = jax.numpy.mean(jax.numpy.max(beat, axis=1) - jax.numpy.min(beat, axis=1)).item()
        if beat_range > range_min:
            # Check standard deviation condition
            beat_std = jax.numpy.mean(jax.numpy.std(beat, axis=1)).item()
            if beat_std > sd_min:
                filtered_beats.append(beat)

    if filtered_beats:
        return jax.numpy.stack(filtered_beats)

    return jax.numpy.array([])


def segment_and_filter_ecg(x_signal, sampling_rate, window_lims=(-100, 200)):
    """
    Segments and filters out beats based on certain criteria.
    Args:
        signal (numpy.ndarray): ECG signal.
        sampling_rate (int): Sampling rate of the ECG signal.
        window_lims (tuple, optional): Tuple containing the left and right window limits.
    Returns:
        beats_filtered: JAX array of filtered beats.
    """
    rpeaks = get_peaks(x_signal, sampling_rate=sampling_rate)

    # Identify beat segments based on R-peaks
    beats, beat_windows = segment_signal(signal, window_lims, rpeaks['ECG_R_Peaks'])

    # Filter beats
    beats_filtered = filter_beats(
        beats, beat_windows, rpeaks, range_cutoff=0.5, sd_cutoff=0.06, 
        autocorr_cutoff=0.75
    )
    
    return beats_filtered