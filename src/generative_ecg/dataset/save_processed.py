import jax.numpy
import pathlib

def save_beat_dataset(x_beats, y_beats, filepath):
    proc_path = pathlib.Path(filepath + "/processed")
    if not proc_path.exists():
        proc_path.mkdir()
        
    x_path = proc_path / "x_beats.npy"
    y_path = proc_path / "y_beats.npy"
        
    jax.numpy.save(x_path, x_beats)
    jax.numpy.save(y_path, y_beats)

def load_beat_dataset(filepath):
    x_path = pathlib.Path(filepath + "/processed/x_beats.npy")
    y_path = pathlib.Path(filepath + "/processed/y_beats.npy")

    if x_path.exists() and y_path.exists():
        x_beats = jax.numpy.load(x_path)
        y_beats = jax.numpy.load(y_path)
        return x_beats, y_beats
    
    raise FileNotFoundError("x_beat/y_beat dataset not found.")

