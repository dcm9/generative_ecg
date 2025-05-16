import jax.numpy
import jax
import matplotlib.gridspec 
import matplotlib.pyplot
from typing import Optional 

from ..models.math_utils import OMAT

def plot_ecg(channel_data: jax.numpy.ndarray, channels: list[str], n_channels: int, 
             figsize: tuple[int,int], std: Optional[list[float]], title: str, 
             ylim: Optional[tuple[float, float]]) -> tuple[matplotlib.figure.Figure, list[matplotlib.axes.Axes]]:
    """
    Plot ECG data for multiple channels.
    Args:
        channel_data (jax.numpy.ndarray): List of ECG data for each channel.
        channels (list): List of channel names.
        n_channels (int): Number of channels.
        figsize (tuple): Figure size.
        std (list): List of standard deviations for each channel.
        title (str): Title of the plot.
        ylim (tuple): Y-axis limits.
    Returns:
        fig (matplotlib.figure.Figure): Figure object.
        axes (list): List of axes objects for each channel.
    """
    fig, ax = matplotlib.pyplot.subplots(figsize=figsize)
    ax.set_xticks([])
    ax.set_yticks([])
    gs = matplotlib.gridspec.GridSpec(n_channels, 1, height_ratios=[1,] * n_channels)

    # Create each subplot
    axes = []
    for i in range(n_channels):
        if i == 0:
            ax = matplotlib.pyplot.subplot(gs[i])
        else:
            ax = matplotlib.pyplot.subplot(gs[i], sharex=axes[0])
        axes.append(ax)

    for i, (ax, data) in enumerate(zip(axes, channel_data)):
        ax.plot(data)
        if std is not None:
            ax.fill_between(
                range(len(data)), data - 2*std[i], data + 2*std[i], 
                color='orange', alpha=0.5
            )
        if ylim:
            ax.set_ylim(ylim)
        
        ax.set_ylabel(f"{channels[i]} (mV)")
    
    if title is not None:
        # if args:
        #     for k, v in args.items():
        #         title += f" | {k}: {v(channel_data):.2f}"
        fig.suptitle(title, fontsize=16)
        
    return fig, axes

# def find_closest_real_ecg(real_ecgs, ecg, processed=False):
#     """
#     Find the closest real ECG to the generated ECG using RMSE.
#     Args:
#         real_ecgs (jax.numpy.ndarray): Array of real ECGs.
#         ecg (jax.numpy.ndarray): Generated ECG.
#         processed (bool): Whether the ECGs are processed or not.
#     Returns:
#         closest_ecg (jax.numpy.ndarray): The closest real ECG.
#         min_rmse (float): The minimum RMSE value.
#     """
#     def _compute_rmse(carry, real_ecg):
#         if processed:
#             real_ecg = OMAT @ real_ecg
#         rmse = jax.numpy.sqrt(jax.numpy.mean((real_ecg - ecg)**2))
#         return rmse, rmse
    
#     _, rmses = jax.lax.scan(_compute_rmse, 0.0, real_ecgs)
#     closest_ecg = real_ecgs[jax.numpy.argmin(rmses)]
#     if processed:
#         closest_ecg = OMAT @ closest_ecg
#     min_rmse = jax.numpy.min(rmses)
    
#     return closest_ecg, min_rmse

# def plot_gaussian_noise(mu, sigma, key, shape):
#     return mu + sigma * jax.random.normal(key, shape=shape)