import jax.numpy
import pathlib

def save_beat_dataset(x_beats: jax.numpy.ndarray, y_beats: jax.numpy.ndarray, filepath: str) -> None:
    """
    Save ECG beat data and corresponding labels to .npy files in a 'processed' subdirectory.

    Args:
        x_beats (Any): Array of ECG beats to save.
        y_beats (Any): Array of labels or targets to save.
        filepath (str): Directory path where the data should be saved.
    """
    proc_path = pathlib.Path(filepath + "/processed")
    if not proc_path.exists():
        proc_path.mkdir()
        
    x_path = proc_path / "x_beats.npy"
    y_path = proc_path / "y_beats.npy"
        
    jax.numpy.save(x_path, x_beats)
    jax.numpy.save(y_path, y_beats)

def load_beat_dataset(filepath: str) -> tuple[jax.numpy.ndarray, jax.numpy.ndarray]:
    """
    Load ECG beat data and corresponding labels from .npy files in a 'processed' subdirectory.
    Args:
        filepath (str): Directory path where the data is saved.
    Returns:
        x_beats (jax.numpy.ndarray): Array of ECG beats.
        y_beats (jax.numpy.ndarray): Array of labels or targets.

    Raises:
        FileNotFoundError: If the .npy files do not exist in the specified directory.
    """
    x_path = pathlib.Path(filepath + "/processed/x_beats.npy")
    y_path = pathlib.Path(filepath + "/processed/y_beats.npy")

    if x_path.exists() and y_path.exists():
        x_beats = jax.numpy.load(x_path)
        y_beats = jax.numpy.load(y_path)
        return x_beats, y_beats
    
    raise FileNotFoundError("x_beat/y_beat dataset not found.")

