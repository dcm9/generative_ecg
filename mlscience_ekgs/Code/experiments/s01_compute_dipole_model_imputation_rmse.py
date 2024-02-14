import argparse
from itertools import islice
import json
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from mlscience_ekgs.settings import mimic_ecg_path, result_path, ptb_path
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
                      n_masked_steps, lr_peak, lr_end, n_epochs, 
                      key=0, n_electrodes=9):
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
    r_init = jr.normal(keys[0], (n_electrodes, 3))
    s_init = jr.normal(keys[1], (n_steps, n_dipoles, 3))
    p_init = jr.normal(keys[2], (n_steps, n_dipoles, 3))
    params = {
        "r": r_init,
        "s": s_init,
        "p": p_init,
    }
    
    # Train the model
    state_post = model.train_rmse(
        params, masked_channel_data, lr_peak, lr_end, n_epochs
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
        data_std = jnp.std(data_masked[jnp.nonzero(data_masked)])
        stds.append(data_std.item())
        
        rmse = jnp.sqrt(
            jnp.sum(
                (pred_masked - data_masked)**2
            ) / combined_mask.sum()
        )
        rmses.append(rmse.item())
    
    return rmses, stds, state_post


def fit_n_dipole_summary_stats(n_dipoles, channel_data, n_masked_channels,
                               n_masked_steps, n_epochs,
                               n_electrodes=9, n_iter=20,
                               channel_idx=0):
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
        rmses, state_post = fit_n_dipole_rmse(
            n_dipoles, channel_data, n_masked_channels, n_masked_steps,
            lr_peaks[i], lr_ends[i], n_epochs, i, n_electrodes
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


def main(args):
    ecg_result_path = Path(result_path, "s01_dipole_model_imputation_rmse")
    if args.compute_rmse_using_stored_results:
        ecg_result_path = Path(result_path, "s01_dipole_model_imputation_rmse")
        ecg_result_path = Path(ecg_result_path, "mimic-iv")
        indices = [str(i) for i in range(1, 13)]
        columns = [str(i) for i in range(1, 7)] + ["std"]
        rmse_df = pd.DataFrame(0.0, columns=columns, index=indices)
        result = {ind: {col: [] for col in columns} for ind in indices}
        counts = 0
        if not ecg_result_path.is_dir():
            raise ValueError(f"Invalid path: {ecg_result_path}")
        for filepath in ecg_result_path.iterdir():
            counts += 1
            if filepath.is_dir():
                curr_path = Path(filepath, "x01_dipole_imputation_rmse.json")
                with open(curr_path, "r") as f:
                    rmse = json.load(f)
                for key, value in rmse.items():
                    for k, v in value.items():
                        result[key][k].append(v['0'])
                std_path = Path(filepath, "x01_dipole_imputation_std.json")
                with open(std_path, "r") as f:
                    std = json.load(f)
                for key, value in std.items():
                    # rmse_df["std"][key] += value
                    result[key]["std"].append(value)
        for key, val in result.items():
            for k, v in val.items():
                rmse_df[k][key] = jnp.nanmean(jnp.array(v))
        print(f"RMSE for {counts} examples:\n")
        print(rmse_df.to_latex())
        return
    
    # Load the data
    if args.dataset == "ptb":
        data_loader_fn = data_loader.load_ptb_data
        with open(Path(ptb_path, "RECORDS")) as fp:
            data_list = fp.readlines()
        data_list = [Path(ptb_path, data.strip()) for data in data_list]
        path_iterator = iter(data_list)
        ecg_result_path = Path(ecg_result_path, "ptb")
    elif args.dataset == "mimic-iv":
        data_loader_fn = data_loader.load_mimic_data
        path_iterator = Path(mimic_ecg_path, "files").rglob("*.dat")
        ecg_result_path = Path(ecg_result_path, "mimic-iv")
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    
    for i, ecg_path in enumerate(islice(path_iterator, args.n_examples)):
        ecg_path = ecg_path.parent / ecg_path.stem
        curr_result_path = Path(ecg_result_path, ecg_path.parts[-1])
        if curr_result_path.is_dir():
            continue
        print(f"Example {i+1}:")
        channel_data, _ = data_loader_fn(
            ecg_filepath=ecg_path, length=args.n_steps
        )
        n_channels = channel_data.shape[0]
        
        # Fit the model and compute the RMSE
        rmse_result = {i: {} for i in range(1, n_channels+1)}
        std_result = {i: {} for i in range(1, n_channels+1)}
        for n_dipole in args.n_dipoles:
            for n_masked_channel in args.n_masked_channels:
                print(f"\tn_dipole: {n_dipole},"
                      f" n_masked_channel: {n_masked_channel}")
                if args.summary_stats:
                    summary_stats = fit_n_dipole_summary_stats(
                        n_dipole, channel_data, n_masked_channel, 
                        args.n_masked_steps, args.n_epochs, n_iter=args.n_iter 
                    )
                    print(json.dumps(summary_stats, indent=4))
                else:
                    rmses, stds, *_ = fit_n_dipole_rmse(
                        n_dipole, channel_data, n_masked_channel, 
                        args.n_masked_steps, args.lr_peak, 
                        args.lr_end, args.n_epochs, args.seed
                    )
                    for i, rmse in enumerate(rmses):
                        if n_dipole not in rmse_result[i+1]:
                            rmse_result[i+1][n_dipole] = {}
                        rmse_result[i+1][n_dipole][n_masked_channel] = rmse
                        std_result[i+1] = stds[i]
                    print(f"\t\tRMSE: {rmses[0]*1e3:.9f}1e-3"
                          f" std: {stds[0]*1e3:.9f}1e-3")
        if not args.summary_stats:
            curr_result_path.mkdir(parents=True, exist_ok=True)
            rmse_path = Path(curr_result_path, "x01_dipole_imputation_rmse.json")
            std_path = Path(curr_result_path, "x01_dipole_imputation_std.json")
            with open(rmse_path, "w") as fp:
                json.dump(rmse_result, fp, indent=4)
            with open(std_path, "w") as fp:
                json.dump(std_result, fp, indent=4)
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
                            f"x01_dipole_imputation_rmse_ch{i+1}.pdf"),
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
                        choices=["ptb", "mimic-iv"])
    
    # Specify numbers of dipoles to simulate
    parser.add_argument("--n_dipoles", type=_check_positive_int, nargs="+",
                        default=[1, 2, 3, 4, 5, 6])
    
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
    
    args = parser.parse_args()
    
    main(args)
