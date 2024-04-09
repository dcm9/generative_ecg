import argparse
from functools import partial
from itertools import islice
import json
from pathlib import Path
import pickle

import jax.numpy as jnp
import jax.random as jr
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
from mpl_toolkits.mplot3d import art3d
import numpy as np
import pandas as pd
import tqdm
import seaborn as sns

from mlscience_ekgs.settings import mimic_ecg_path, result_path, ptb_path
from mlscience_ekgs.Code.experiments import s02_train_and_generate_ecgs as train
from mlscience_ekgs.Code.src import s01_data_loader as data_loader
from mlscience_ekgs.Code.src import s02_dipole_model as model
import jax.random as jr


json.encoder.FLOAT_REPR = lambda o: format(o, '.8f')


def _check_positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    
    return ivalue


def _check_nonneg_int(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    
    return ivalue


def _diff_norm(x):
    diffs = np.mean(np.linalg.norm(np.diff(x, axis=0), axis=2), axis=1)
    
    return diffs


def mask_channels(channel_data, n_masked_channels, n_masked_steps):
    """Mask channels in the channel data.

    Args:
        channel_data: 12-lead ECG data.
        n_masked_channels: number of channels to mask.
        n_masked_steps: number of time steps to mask.
    
    Returns:
        masked_channel_data: channel data with masked channels.
    """
    assert n_masked_channels <= channel_data.shape[0]
    channel_mask = jnp.arange(channel_data.shape[0]) < n_masked_channels
    time_mask = jnp.arange(channel_data.shape[1]) < n_masked_steps
    combined_mask = channel_mask[:, None] * time_mask
    masked_channel_data = jnp.where(combined_mask, jnp.nan, channel_data)
    
    return masked_channel_data


def fit_n_dipole_rmse(n_dipoles, channel_data, n_masked_channels, 
                      n_masked_steps, s_smooth, p_smooth,
                      lr_peak, lr_end, n_epochs, key=0,
                      n_electrodes=9, fix_electrodes=False, fix_dipoles=False,
                      constrain_dipoles_to_cuboid=False,
                      s_hard=-1, p_hard=-1, pgd=False):
    """Fit the dipole model and compute the RMSE.

    Args:
        n_dipoles (int): number of dipoles to fit.
        channel_data (int): 12-lead ECG data.
        n_masked_channels (int): number of channels to mask.
        n_masked_steps (int): number of time steps to mask.
        lr_peak (float): peak learning rate.
        lr_end (float): end learning rate.
        n_epochs (int): number of training epochs.
        key (int, optional): random seed. Defaults to 0.
        n_electrodes (int, optional): number of electrodes. Defaults to 9.
        fix_electrodes (bool, optional): whether to fix the electrode
        fix_dipoles (bool, optional): whether to fix the dipole locations
        constrain_dipoles_to_cuboid (bool, optional): whether to constrain
            dipoles within the cuboid. Defaults to False.
        s_hard (float, optional): hard constraint on s. Defaults to -1.
        p_hard (float, optional): hard constraint on p. Defaults to -1.
        pgd (bool, optional): whether to use projected gradient descent.

    Returns:
        rmses: RMSEs of the model on the first n_masked_steps steps of each 
            channel.
        state_post: the final state of the model.
    """    
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    keys = jr.split(key, 3)
    n_steps = channel_data.shape[1]
    
    # Mask the channels
    masked_channel_data = \
        mask_channels(channel_data, n_masked_channels, n_masked_steps)
    if n_masked_channels == 0:
        n_masked_steps = 1e6 # Evaluate on all steps if no masked channels
    
    # Initialize parameters
    r_init = 0.125 * jr.normal(keys[0], (n_electrodes, 3))
    s_init = 0.001 * jr.normal(keys[1], (n_steps, n_dipoles, 3))
    p_init = 0.001 * jr.normal(keys[2], (n_steps, n_dipoles, 3))
    params = {
        "r": r_init,
        "s": s_init,
        "p": p_init,
    }
    
    # Train the model
    if pgd:
        train_fn = model.train_proj_grad_rmse
    else:
        train_fn = model.train_rmse
    state_post = train_fn(
        params, masked_channel_data, s_smooth, p_smooth,
        lr_peak, lr_end, n_epochs, fix_electrodes, fix_dipoles, 
        constrain_dipoles_to_cuboid, s_hard, p_hard
    )
    
    # Evaluate on the first n_masked_steps steps of each channel
    rmses = []
    stds = []
    n_channels = channel_data.shape[0]
    pred_channel_data = model.predict_lead_obs(state_post.params)
    for i in range(n_channels):
        channel_mask = jnp.arange(n_channels) == i
        time_mask = jnp.arange(channel_data.shape[1]) < n_masked_steps
        combined_mask = channel_mask[:, None] * time_mask
        
        pred_masked = pred_channel_data * combined_mask
        data_masked = channel_data * combined_mask
        
        # Compute standard deviation of data_masked
        # data_std = jnp.std(data_masked[jnp.nonzero(data_masked)])
        # stds.append(data_std.item())
        curr_rmses = jnp.sqrt(
            (pred_masked - data_masked)**2
        )
        curr_means = jnp.mean(curr_rmses[jnp.nonzero(data_masked)])
        curr_stds = jnp.std(curr_rmses[jnp.nonzero(data_masked)])
        # rmse = jnp.sqrt(
        #     jnp.sum(
        #         (pred_masked - data_masked)**2
        #     ) / combined_mask.sum()
        # )
        rmses.append(curr_means.item())
        stds.append(curr_stds.item())
    
    return rmses, stds, state_post


def fit_n_dipole_summary_stats(n_dipoles, channel_data, n_masked_channels,
                               n_masked_steps, n_epochs,
                               s_smooth, p_smooth,
                               n_electrodes=9, n_iter=20,
                               channel_idx=0, fix_electrodes=False,
                               fix_dipoles=False,
                               constrain_dipoles_to_cuboid=False,
                               s_hard=-1, p_hard=-1, pgd=False):
    """Fit the dipole model and compute the summary statistics.

    Args:
        n_dipoles (int): number of dipoles to fit.
        channel_data (Array): 12-lead ECG data.
        n_masked_channels (int): number of channels to mask.
        n_masked_steps (int): number of time steps to mask.
        n_epochs (int): number of training epochs.
        n_electrodes (int, optional): number of electrodes. Defaults to 9.
        n_iter (int, optional): number of iterations. Defaults to 20.
        channel_idx (int, optional): channel index. Defaults to 0.
        fix_electrodes (bool, optional): whether to fix the electrode
        fix_dipoles (bool, optional): whether to fix the dipole locations
        constrain_dipoles_to_cuboid (bool, optional): whether to constrain
            dipoles within the cuboid. Defaults to False.
        s_hard (float, optional): hard constraint on s. Defaults to -1.
        p_hard (float, optional): hard constraint on p. Defaults to -1.            
        pgd (bool, optional): whether to use projected gradient descent.

    Returns:
        summary_stats: summary statistics of the model.
    """    
    summary_stats = {
        "r_mean": [],
        "r_std": [],
        "s_mean": [],
        "s_std": [],
        "p_mean": [],
        "p_std": [],
        "rmse": [],
    }
    lr_peaks = jnp.arange(1, n_iter+1) * 1e-2
    lr_ends = jnp.arange(1, n_iter+1) * 1e-8
    for i in range(n_iter):
        rmses, stds, state_post = fit_n_dipole_rmse(
            n_dipoles, channel_data, n_masked_channels, n_masked_steps,
            s_smooth, p_smooth, lr_peaks[i], lr_ends[i], n_epochs, i, 
            n_electrodes, fix_electrodes, fix_dipoles, 
            constrain_dipoles_to_cuboid, s_hard, p_hard, pgd
        )
        rmse = rmses[channel_idx]
        summary_stats["r_mean"].append(state_post.params["r"].mean().item())
        summary_stats["r_std"].append(state_post.params["r"].std().item())
        summary_stats["s_mean"].append(state_post.params["s"].mean().item())
        summary_stats["s_std"].append(state_post.params["s"].std().item())
        summary_stats["p_mean"].append(state_post.params["p"].mean().item())
        summary_stats["p_std"].append(state_post.params["p"].std().item())
        summary_stats["rmse"].append(rmse)

    return summary_stats


def make_and_save_video(result, ecg, file_path, fps):
    """Make and save a video of the dipole model imputation.

    Args:
        result (dict): result of the dipole model imputation.
        ecg (Array): 12-lead ECG data.
        file_path (str): path to save the video.
        fps (int): frames per second.
    """    
    def make_frame(t, result, ecg, ecg_pred, rmse, fps):
        fig = plt.figure(figsize=(20, 6), dpi=100)
        gs = gridspec.GridSpec(3, 3, figure=fig)

        # Plot ECG
        ecg_axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
        idx = int(t * fps)
        time = np.arange(ecg.shape[1]) / fps  # Convert index to time using fps
        for i, ax in enumerate(ecg_axes):
            ax.clear()
            ax.plot(time, ecg[i], '--', label=f"Channel {i+1}", alpha=0.6)
            ax.plot(time[:idx], ecg_pred[i, :idx], 
                    label=f"Channel {i+1} (pred)", alpha=0.6, color='red')
            ax.set_xlim(time[0], time[-1])
            ax.set_ylim(np.min(ecg[i], axis=0), np.max(ecg[i], axis=0))
            ax.set_xticks([])
            ax.set_yticks([])
        # ecg_axes[0].set_title(f"RMSE: {rmse[idx]*1e5:.3f}e-5")
        
        # Plot arrows
        locs, moments = result["s"], result["p"]
        ax1 = fig.add_subplot(gs[:, 1], projection='3d')
        colors = ['red', 'blue', 'green', 'orange']
        for i in range(locs.shape[1]):
            ax1.quiver(locs[idx, i, 0], locs[idx, i, 1], locs[idx, i, 2],
                       10*moments[idx, i, 0], 10*moments[idx, i, 1], 
                       10*moments[idx, i, 2], linewidths=3, 
                       color=colors[i % len(colors)])
            ax1.scatter(locs[idx, i, 0], locs[idx, i, 1], locs[idx, i, 2], 
                        color=colors[i % len(colors)], s=40)
            
            # Trace trajectory of locs
            ax1.plot(locs[:idx+1, i, 0], locs[:idx+1, i, 1], locs[:idx+1, i, 2], 
                     '--', color=colors[i % len(colors)], alpha=0.5)
        
        # Plot elliptic cylinder
        theta = np.linspace(0, 2*np.pi, 100)
        z = np.linspace(-0.125, 0.125, 100)
        theta, z = np.meshgrid(theta, z)

        major_axis_length = 0.125
        minor_axis_length = major_axis_length / 2.75
        x = major_axis_length * np.cos(theta)
        y = minor_axis_length * np.sin(theta)

        ax1.plot_surface(x, y, z, alpha=0.2, color='grey')
        ax1.plot(x[0], y[0], z[0], color='grey', alpha=0.2)
        ax1.plot(x[-1], y[-1], z[-1], color='grey', alpha=0.2)
        ax1.set_xlim([-0.13, 0.13])
        ax1.set_ylim([-0.13, 0.13])
        ax1.set_zlim([-0.13, 0.13])
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        n_ticks = 5
        ax1.set_xticks(np.linspace(-0.1, 0.1, n_ticks))
        ax1.set_yticks(np.linspace(-0.1, 0.1, n_ticks))
        ax1.set_zticks(np.linspace(-0.1, 0.1, n_ticks))
        # ax.set_title(f"Average RMSE: {rmse.mean()*1e5:.3f}e-5")
        
        # Plot cuboid
        x_len, y_len, z_len = 0.085, 0.06, 0.12
        vertices = np.array([[-x_len/2, -y_len/2, -z_len/2],
                            [x_len/2, -y_len/2, -z_len/2],
                            [x_len/2, y_len/2, -z_len/2],
                            [-x_len/2, y_len/2, -z_len/2],
                            [-x_len/2, -y_len/2, z_len/2],
                            [x_len/2, -y_len/2, z_len/2],
                            [x_len/2, y_len/2, z_len/2],
                            [-x_len/2, y_len/2, z_len/2]])
        sides = [[vertices[0], vertices[1], vertices[5], vertices[4]],
                 [vertices[7], vertices[6], vertices[2], vertices[3]],
                 [vertices[0], vertices[3], vertices[7], vertices[4]],
                 [vertices[1], vertices[2], vertices[6], vertices[5]],
                 [vertices[0], vertices[1], vertices[2], vertices[3]],
                 [vertices[4], vertices[5], vertices[6], vertices[7]]]

        # Create and add polygons for each side of the cuboid
        for side in sides:
            poly = art3d.Poly3DCollection([side], alpha=0.1, color='orange')
            ax1.add_collection3d(poly)
        
        ax2 = fig.add_subplot(gs[:, 2])
        ax2.plot(time[:idx], rmse[:idx]*1e5, label="RMSE")
        ax2.set_xlim(time[0], time[-1])
        ax2.set_ylim(0, rmse.max()*1e5*1.1)
        ax2.set_title(f"RMSE: {rmse[:idx].mean()*1e5:.4f}e-5")
        ax2.set_xticks([])
        ax2.set_ylabel("RMSE (1e-5 mV)")

        # Convert matplotlib figure to RGB array
        fig.subplots_adjust(wspace=0.3);
        frame = mplfig_to_npimage(fig)
        plt.close(fig)
        
        return frame
    duration = result["s"].shape[0] / fps
    ecg_pred = model.predict_lead_obs(result)
    rmse = jnp.sqrt(jnp.mean((ecg_pred - ecg)**2, axis=0))
    clip = VideoClip(
        partial(make_frame, result=result, ecg=ecg, 
                ecg_pred=ecg_pred, rmse=rmse, fps=fps),
        duration=duration
    )
    clip.write_videofile(str(file_path), fps=fps)


def main(args):
    # Load the data
    dataset_name = args.dataset
    if args.processed:
        dataset_name += "_processed"
    sp_name = f"s_{args.s_smooth}_p_{args.p_smooth}"
    if args.s_hard >= 0 or args.p_hard >= 0:
        sp_name = f"s_hard_{args.s_hard}_p_hard_{args.p_hard}"
    
    if args.fix_electrodes:
        sp_name += "_fixed_electrodes"
    if args.fix_dipoles:
        sp_name += "_fixed_dipoles"
    elif args.constrain_dipoles_to_cuboid:
        sp_name += "_cuboid_constrained_dipoles"
    ecg_result_path = Path(result_path, "s01_dipole_model_imputation_rmse",
                           dataset_name, args.gen_model)
    if args.pgd:
        ecg_result_path = Path(ecg_result_path, "pgd")
    else:
        ecg_result_path = Path(ecg_result_path, "adam")
    ecg_result_path = Path(ecg_result_path, sp_name)
    
    if args.dataset == "ptb":
        data_loader_fn = data_loader.load_ptb_data
        with open(Path(ptb_path, "RECORDS")) as fp:
            data_list = fp.readlines()
        data_list = [Path(ptb_path, data.strip()) for data in data_list]
        path_iterator = iter(data_list)
    elif args.dataset == "mimic-iv":
        data_loader_fn = data_loader.load_mimic_data
        path_iterator = Path(mimic_ecg_path, "files").rglob("*.dat")
    elif args.dataset == "ptb-xl":
        X, *_ = train.load_dataset(args.dataset, args.beat_segment,
                                   args.processed, args.n_channels)
        if args.processed:
            dataset_name = f"{args.dataset}_processed"
        else:
            dataset_name = args.dataset
        if args.gen_model == "dr-vae":
            gen_result_path = Path(
                result_path, "s03_generative_models", 
                dataset_name, args.gen_model
            )
            if args.beat_segment:
                gen_result_path = Path(
                    gen_result_path, f"bs_ch{args.n_channels}"
                )
            else:
                gen_result_path = Path(gen_result_path, f"ch{args.n_channels}")
            gen_result_path = Path(
                gen_result_path, f"beta1_{args.beta1_scheduler}",
                f"beta1_{args.beta1}"
            )
            if args.beat_segment:
                gen_ckpt_dir = Path(
                    gen_result_path, f"bs_{args.n_channels}_ckpt"
                )
            else:
                gen_ckpt_dir = Path(gen_result_path, f"{args.n_channels}_ckpt")
            result = train.load_model(X, gen_ckpt_dir, args)
            path_iterator = iter(jnp.arange(args.n_examples))
            data_loader_fn = lambda ecg_filepath, length: \
                (train.generate_vae_ecg(result, args.n_channels, ecg_filepath),
                None)
        elif args.gen_model == "real":
            path_iterator = iter(jnp.arange(args.n_examples))
            data_loader_fn = lambda ecg_filepath, length: \
                (X[ecg_filepath], None)
            if args.processed:
                data_loader_fn = lambda ecg_filepath, length: \
                    (model.OMAT @ X[ecg_filepath], None)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    
    if args.compute_rmse_using_stored_results:
        indices = [str(i) for i in range(1, 13)]
        columns = [str(i) for i in range(1, 7)] + ["std"]
        rmse_df = pd.DataFrame(0.0, columns=columns, index=indices)
        result = {ind: {col: [] for col in columns} for ind in indices}
        counts = 0
        if not ecg_result_path.is_dir():
            raise ValueError(f"Invalid path: {ecg_result_path}")
        for filepath in ecg_result_path.iterdir():
            if filepath.is_dir():
                counts += 1
                curr_path = Path(filepath, "dipole_imputation_rmse.json")
                with open(curr_path, "r") as f:
                    rmse = json.load(f)
                for key, value in rmse.items():
                    for k, v in value.items():
                        result[key][k].append(v['0'])
                std_path = Path(filepath, "dipole_imputation_std.json")
                with open(std_path, "r") as f:
                    std = json.load(f)
                for key, value in std.items():
                    # rmse_df["std"][key] += value
                    result[key]["std"].append(value)
        # Save result as json
        with open(Path(ecg_result_path, "dipole_imputation_rmse.json"), "w") as f:
            json.dump(result, f, indent=4)
        for key, val in result.items():
            for k, v in val.items():
                rmse_df[k][key] = jnp.nanmean(jnp.array(v))
        print(f"RMSE for {counts} examples:\n")
        print(rmse_df.to_latex())
        return
    
    if args.verbose:
        iterator = enumerate(islice(path_iterator, args.n_examples))
    else:
        iterator = enumerate(tqdm.tqdm(islice(path_iterator, args.n_examples),
                                       total=args.n_examples))
    for i, ecg_path in iterator:
        if isinstance(ecg_path, Path):
            ecg_path = ecg_path.parent / ecg_path.stem
            curr_result_path = Path(ecg_result_path, ecg_path.parts[-1])
        else:
            ecg_path = int(ecg_path)
            curr_result_path = Path(ecg_result_path, str(ecg_path))
        if curr_result_path.is_dir():
            continue
        if args.verbose:
            print(f"Example {i+1}:")
        channel_data, *_ = data_loader_fn(
            ecg_filepath=ecg_path, length=args.n_steps
        )
        n_channels = channel_data.shape[0]
        
        # Fit the model and compute the RMSE
        rmse_result = {i: {} for i in range(1, n_channels+1)}
        std_result = {i: {} for i in range(1, n_channels+1)}
        params_result = {}
        for n_dipole in args.n_dipoles:
            for n_masked_channel in args.n_masked_channels:
                if args.verbose:
                    print(f"\tn_dipole: {n_dipole},"
                        f" n_masked_channel: {n_masked_channel}")
                if args.summary_stats:
                    summary_stats = fit_n_dipole_summary_stats(
                        n_dipole, channel_data, n_masked_channel, 
                        args.n_masked_steps, args.s_smooth, args.p_smooth,
                        9, args.n_iter, 0, args.fix_electrodes,
                        args.fix_dipoles, args.constrain_dipoles_to_cuboid,
                        args.s_hard, args.p_hard, args.pgd
                    )
                    print(json.dumps(summary_stats, indent=4))
                else:
                    rmses, stds, state_post = fit_n_dipole_rmse(
                        n_dipole, channel_data, n_masked_channel,
                        args.n_masked_steps, args.s_smooth, args.p_smooth,
                        args.lr_peak, args.lr_end, args.n_epochs, args.seed, 9,
                        args.fix_electrodes, args.fix_dipoles,
                        args.constrain_dipoles_to_cuboid,
                        args.s_hard, args.p_hard, args.pgd
                    )
                    for i, rmse in enumerate(rmses):
                        if n_dipole not in rmse_result[i+1]:
                            rmse_result[i+1][n_dipole] = {}
                        rmse_result[i+1][n_dipole][n_masked_channel] = rmse
                        std_result[i+1] = stds[i]
                    if args.verbose:
                        print(f"\t\tCh 1 - RMSE: {rmses[0]*1e3:.9f}1e-3"
                            f" std: {stds[0]*1e3:.9f}1e-3")
                    params_result[n_dipole] = state_post.params
        if not args.summary_stats:
            curr_result_path.mkdir(parents=True, exist_ok=True)
            rmse_path = Path(curr_result_path, "dipole_imputation_rmse.json")
            std_path = Path(curr_result_path, "dipole_imputation_std.json")
            ecg_path = Path(curr_result_path, "ecg.npy")
            with open(rmse_path, "w") as fp:
                json.dump(rmse_result, fp, indent=4)
            with open(std_path, "w") as fp:
                json.dump(std_result, fp, indent=4)
            jnp.save(ecg_path, channel_data)
            for n_dipole, state_post in params_result.items():
                state_post_path = Path(
                    curr_result_path, 
                    f"dipole_imputation_params_{n_dipole}_dipole.pkl"
                )
                with open(state_post_path, "wb") as fp:
                    pickle.dump(state_post, fp)
            
            for n_dipole, state_post in params_result.items():
                s, p = state_post["s"], state_post["p"]
                
                # Plot consecutive differences of s and p
                s_diffs = _diff_norm(s)
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.hist(s_diffs, label="s", bins=100)
                ax.set_title("Distribution of consecutive differences of s "
                             f"({n_dipole} dipoles)")
                ax.set_xlabel("Sum of norm of differences")
                ax.set_ylabel("Frequency");
                fig.savefig(
                    Path(curr_result_path, 
                        f"dipole_imputation_s_diff_{n_dipole}_dipole.png"),
                    bbox_inches="tight", dpi=300
                )
                plt.close('all')
                
                p_diffs = _diff_norm(p)
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.hist(p_diffs, label="p", bins=100)
                ax.set_title("Distribution of consecutive differences of p "
                             f"({n_dipole} dipoles)")
                ax.set_xlabel("Sum of norm of differences")
                ax.set_ylabel("Frequency");
                fig.savefig(
                    Path(curr_result_path, 
                        f"dipole_imputation_p_diff_{n_dipole}_dipole.png"),
                    bbox_inches="tight", dpi=300
                )
                plt.close('all')
                
                if args.generate_video:
                    make_and_save_video(
                        state_post, channel_data,
                        Path(curr_result_path, 
                            f"dipole_imputation_{n_dipole}_dipole.mp4"), 20
                    )
            if args.save_plots:
                for i in range(n_channels):
                    curr_result = rmse_result[i]
                    # Plot heatmap
                    df = pd.DataFrame(curr_result).T.astype(float)
                    df = df*1e3
                    fig, ax = plt.subplots(figsize=(10, 8))
                    ax = sns.heatmap(df, annot=True, fmt=".6f", 
                                    ax=ax, cmap="viridis")
                    ax.set_title(f"RMSE of dipole model imputation of Ch {i+1}" 
                                " (in 1e-3 mV)")
                    ax.set_xlabel("Number of masked channels")
                    ax.set_ylabel("Number of dipoles")
                    fig.savefig(
                        Path(curr_result_path, 
                            f"dipole_imputation_rmse_ch{i+1}.pdf"),
                        bbox_inches="tight", dpi=300
                    )
                    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Compute average RMSE statistics given previous results
    parser.add_argument("--compute_rmse_using_stored_results", 
                        action="store_true")
    
    # Specify dataset
    parser.add_argument("--dataset", type=str, default="ptb",
                        choices=["ptb", "mimic-iv", "ptb-xl"])
    parser.add_argument("--n_channels", type=int, default=12)
    parser.add_argument("--beat_segment", action="store_true") # use beat segmentations
    parser.add_argument("--processed", action="store_true") # use processed dataset
    
    # Generate video?
    parser.add_argument("--generate_video", action="store_true")
    
    # Use generative model
    parser.add_argument("--gen_model", type=str, default="real",
                        choices=["real", "dr-vae"])
    
    # Specify model parameters for VAE
    parser.add_argument("--z_dim", type=int, default=512) # latent dim
    parser.add_argument("--hidden_width", type=int, default=100) # hidden layer width
    parser.add_argument("--hidden_depth", type=int, default=4) # hidden layer depth
    parser.add_argument("--lr_init", type=float, default=1e-7) # initial learning rate
    parser.add_argument("--beta1", type=float, default=0.01) # KL-div reg. weight
    parser.add_argument("--beta2", type=float, default=0.0) # disc. reg. weight
    parser.add_argument("--target", type=str, default="sex") # target for discriminator
    parser.add_argument("--beta1_scheduler", type=str, default="constant",
                        choices=["constant", "linear", "cosine",
                                 "warmup_cosine", "cyclical"],)
    
    # Specify numbers of dipoles to simulate
    parser.add_argument("--n_dipoles", type=_check_positive_int, nargs="+",
                        default=[1, 2, 3, 4, 5, 6])
    
    # Specify smoothness regularization
    parser.add_argument("--s_smooth", type=float, default=0.0)
    parser.add_argument("--p_smooth", type=float, default=0.0)
    
    # Specify smoothness hard constraints
    parser.add_argument("--s_hard", type=float, default=-1.0)
    parser.add_argument("--p_hard", type=float, default=-1.0)
    
    # Whether to fix electrode placements
    parser.add_argument("--fix_electrodes", action="store_true")
    
    # Whether to fix dipole locations
    parser.add_argument("--fix_dipoles", action="store_true")
    
    # Whether to constrain dipoles within the cuboid
    parser.add_argument("--constrain_dipoles_to_cuboid", action="store_true")
    
    # Specify number of ECG examples to fit
    parser.add_argument("--n_examples", type=_check_positive_int, default=1)
    
    # Number of time steps to load
    parser.add_argument("--n_steps", type=float, default=5_000)
    
    # Specify number of timesteps to mask
    parser.add_argument("--n_masked_steps", type=int, default=1_000)
    
    # Specify numbers of simultaneous masked channels
    parser.add_argument("--n_masked_channels", type=_check_nonneg_int, nargs="+",
                        default=[0, 1, 2, 3, 4, 5])
    
    # Number of training epochs
    parser.add_argument("--n_epochs", type=int, default=10_000)
    
    # Peak learning rate for training
    parser.add_argument("--lr_peak", type=float, default=1e-1)
    
    # End learning rate for training
    parser.add_argument("--lr_end", type=float, default=1e-7)
    
    # Random seed
    parser.add_argument("--seed", type=int, default=0)
    
    # Save the results as heat maps
    parser.add_argument("--save_plots", action="store_true")
    
    # Summary stats experiment
    parser.add_argument("--summary_stats", action="store_true")
    
    # Summary stats experiment n_iter
    parser.add_argument("--n_iter", type=int, default=5)
    
    # Use projected gradient descent
    parser.add_argument("--pgd", action="store_true")
    
    # Verbose
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    main(args)
