import jax.numpy
import jax
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from ..models.math_utils import OMAT

def plot_ecg(channel_data, channels, n_channels=12, figsize=(16, 8), std=None,
             title=None, ylim=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xticks([])
    ax.set_yticks([])
    gs = gridspec.GridSpec(n_channels, 1, height_ratios=[1,] * n_channels)

    # Create each subplot
    axes = []
    for i in range(n_channels):
        if i == 0:
            ax = plt.subplot(gs[i])
        else:
            ax = plt.subplot(gs[i], sharex=axes[0])
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

def find_closest_real_ecg(real_ecgs, ecg, processed=False):
    def _compute_rmse(carry, real_ecg):
        if processed:
            real_ecg = OMAT @ real_ecg
        rmse = jax.numpy.sqrt(jax.numpy.mean((real_ecg - ecg)**2))
        return rmse, rmse
    
    _, rmses = jax.lax.scan(_compute_rmse, 0.0, real_ecgs)
    closest_ecg = real_ecgs[jax.numpy.argmin(rmses)]
    if processed:
        closest_ecg = OMAT @ closest_ecg
    min_rmse = jax.numpy.min(rmses)
    
    return closest_ecg, min_rmse