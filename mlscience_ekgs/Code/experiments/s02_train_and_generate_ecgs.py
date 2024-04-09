import argparse
from functools import partial
from pathlib import Path

from flax.training import orbax_utils
import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import orbax.checkpoint as orbax_ckpt
import tqdm

from mlscience_ekgs.settings import result_path
from mlscience_ekgs.Code.src.s01_data_loader import load_ptb_xl_dataset
from mlscience_ekgs.Code.src.s02_dipole_model import OMAT
from mlscience_ekgs.Code.src.s03_dr_vae import (
    CNNEncoder, 
    Decoder, 
    gaussian_sample,
    train_dr_vae
)
import mlscience_ekgs.Code.src.s04_models as models
from mlscience_ekgs.Code.src.s06_utils import plot_ecg
from mlscience_ekgs.Code.src.s08_dsm import (
    create_train_state, 
    sample_annealed_langevin,
    train_dsm
)


CHANNELS = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 
            'V4', 'V5', 'V6']


def load_dataset(dataset, beat_segment, processed, n_channels, verbose=True, 
                 target="age", x_len=400):
    if dataset == "ptb-xl":
        if beat_segment:
            sampling_rate = 500
        else:
            sampling_rate = 100
        X_tr, y_tr, X_te, y_te = load_ptb_xl_dataset(
            segmentation=beat_segment, 
            sampling_rate=sampling_rate,
            target=target,
            processed=processed,
        )
    else:
        raise ValueError(f"Dataset {dataset} not supported.")
    # Take only the first n channels
    if not processed:
        X_tr = X_tr[:, :n_channels, :]
        X_te = X_te[:, :n_channels, :]
        if not beat_segment:
            X_tr = X_tr[:, :, :x_len]
            X_te = X_te[:, :, :x_len]
    
    if verbose:
        print(
            f"Loaded dataset {dataset} with shape X_tr: {X_tr.shape}, "
            f"y_tr: {y_tr.shape}, X_te: {X_te.shape}, y_te: {y_te.shape}"
        )
    
    return X_tr, y_tr, X_te, y_te, target
    

def train_and_save_model(X, gen_ckpt_dir, dataset_name, configs):
    # Load discriminative model
    ckpt_dir = Path(result_path, "s02_discriminative_models", 
                    dataset_name, configs.target)
    if configs.beat_segment:
        ckpt_dir = Path(ckpt_dir, f"cnn_bs_{configs.n_channels}_ckpt")
    else:
        ckpt_dir = Path(ckpt_dir, f"cnn_{configs.n_channels}_ckpt")
    state_disc = models.create_cnn_train_state(X)
    ckptr = orbax_ckpt.Checkpointer(orbax_ckpt.PyTreeCheckpointHandler())
    state_disc = ckptr.restore(
        ckpt_dir, item=state_disc
    )
    model = models.CNN(output_dim=1)
    vae_pred_fn = lambda x: model.apply(state_disc.params, x)
    
    gen_ckpt_dir.mkdir(parents=True, exist_ok=True)
    if configs.gen_model == "dr-vae":
        result = train_dr_vae(vae_pred_fn, X, configs.beta1, configs.beta2,
                              configs.z_dim, configs.seed, configs.n_epochs,
                              configs.batch_size, configs.hidden_width,
                              configs.hidden_depth, configs.lr_init,
                              configs.lr_peak, configs.lr_end,
                              encoder_type="cnn", use_bias=False,
                              beta1_scheduler_type=configs.beta1_scheduler,)
        # drvae_result_path = Path(result_path, "s03_ptbxl", "s01_dr_vae")
        # gen_ckpt_dir = Path(drvae_result_path, "dr_vae_ckpt")
        with open(Path(gen_ckpt_dir, "params_enc.npy"), "wb") as f:
            jnp.save(f, result["params_enc"])
        with open(Path(gen_ckpt_dir, "params_dec.npy"), "wb") as f:
            jnp.save(f, result["params_dec"])
        with open(Path(gen_ckpt_dir, "mu_mean.npy"), "wb") as f:
            jnp.save(f, result["mu_mean"])
        with open(Path(gen_ckpt_dir, "mu_std.npy"), "wb") as f:
            jnp.save(f, result["mu_std"])
        with open(Path(gen_ckpt_dir, "sigmasq_mean.npy"), "wb") as f:
            jnp.save(f, result["sigmasq_mean"])
        with open(Path(gen_ckpt_dir, "sigmasq_std.npy"), "wb") as f:
            jnp.save(f, result["sigmasq_std"])
        with open(Path(gen_ckpt_dir, "losses.npy"), "wb") as f:
            jnp.save(f, result["losses"])
        with open(Path(gen_ckpt_dir, "losses_rec.npy"), "wb") as f:
            jnp.save(f, result["losses_rec"])
        with open(Path(gen_ckpt_dir, "losses_kl.npy"), "wb") as f:
            jnp.save(f, result["losses_kl"])
        with open(Path(gen_ckpt_dir, "losses_dr.npy"), "wb") as f:
            jnp.save(f, result["losses_dr"])
    elif configs.gen_model == "dsm":
        X_tr = jnp.swapaxes(X, 1, 2)
        result, loss_history = train_dsm(
            X_tr, key=configs.seed, n_epochs=configs.n_epochs,
        )
        ckptr = orbax_ckpt.Checkpointer(orbax_ckpt.PyTreeCheckpointHandler())
        save_args = orbax_utils.save_args_from_target(result)
        ckptr.save(gen_ckpt_dir, result, force=True, save_args=save_args)
    else:
        raise ValueError(f"Model {configs.gen_model} not supported.")
    
    return result


def find_closest_real_ecg(real_ecgs, ecg, processed=False):
    def _compute_rmse(carry, real_ecg):
        if processed:
            real_ecg = OMAT @ real_ecg
        rmse = jnp.sqrt(jnp.mean((real_ecg - ecg)**2))
        return rmse, rmse
    
    _, rmses = jax.lax.scan(_compute_rmse, 0.0, real_ecgs)
    closest_ecg = real_ecgs[jnp.argmin(rmses)]
    if processed:
        closest_ecg = OMAT @ closest_ecg
    min_rmse = jnp.min(rmses)
    
    return closest_ecg, min_rmse


def generate_vae_ecg(result_dict, n_channels, key=0):
    if isinstance(key, int):
        key = jr.PRNGKey(key)
    keys = jr.split(key, 3)
    fn_dec, params_dec = result_dict["apply_fn_dec"], result_dict["params_dec"]
    mu_mean, mu_std = result_dict["mu_mean"], result_dict["mu_std"]
    sigmasq_mean, sigmasq_std = \
        result_dict["sigmasq_mean"], result_dict["sigmasq_std"]
    
    mu_curr = mu_mean + mu_std*jr.normal(keys[0], shape=mu_mean.shape)
    sigmasq_curr = sigmasq_mean + \
        sigmasq_std*jr.normal(keys[1], shape=sigmasq_mean.shape)
    z = mu_curr + jnp.sqrt(sigmasq_curr)*jr.normal(keys[2], shape=mu_curr.shape)
    ecg = fn_dec(params_dec, z).reshape(n_channels, -1)
    
    return ecg

    
def generate_and_save_ecgs(X, result, gen_result_path, configs):
    gen_result_path = Path(gen_result_path, "generated_ecgs")
    gen_result_path.mkdir(parents=True, exist_ok=True)
    key = jr.PRNGKey(configs.seed)
    key, subkey = jr.split(key)
    if configs.gen_model == "dr-vae":
        fn_dec, params_dec = result["apply_fn_dec"], result["params_dec"]
        mu_mean, mu_std = result["mu_mean"], result["mu_std"]
        sigmasq_mean, sigmasq_std = \
            result["sigmasq_mean"], result["sigmasq_std"]
        ecgs, rmses = [], []
        for i in tqdm.trange(configs.n_ecgs):
            key1, key2, key3, key  = jr.split(key, 4)
            mu_curr = mu_mean + mu_std*jr.normal(key1, shape=(configs.z_dim,))
            sigmasq_curr = sigmasq_mean + \
                sigmasq_std*jr.normal(key2, shape=(configs.z_dim,))
            z = mu_curr + \
                jnp.sqrt(sigmasq_curr)*jr.normal(key3, shape=(configs.z_dim,))
            x = fn_dec(params_dec, z).reshape(X.shape[1], -1)
            if configs.processed:
                x = OMAT @ x
            ecgs.append(x)
            fig, _ = plot_ecg(x, CHANNELS, configs.n_channels, 
                              (6, configs.n_channels+1), 
                              title=f"ECG {i+1}")
            fig.savefig(Path(gen_result_path, f"ecg_{i+1}.png"))
            plt.close("all")
            
            if configs.find_closest_real:
                ecg_c, dist = find_closest_real_ecg(X, x, configs.processed)
                rmses.append(dist)
                fig, _ = plot_ecg(
                    ecg_c, CHANNELS, configs.n_channels, 
                    (6, configs.n_channels+1),
                    title=f"Closest real ECG {i+1}, dist: {dist:.3f}"
                )
                fig.savefig(Path(gen_result_path, f"ecg_{i+1}_closest.png"))
            plt.close("all")
        if rmses:
            rmses = jnp.array(rmses)
            print(f"\nClosest real ECG:")
            print(f"\tmean: {jnp.mean(rmses):.3f}")
            print(f"\tstd: {jnp.std(rmses):.3f}")
            print(f"\tmax: {jnp.max(rmses):.3f}")
            print(f"\tmin: {jnp.min(rmses):.3f}")
    elif configs.gen_model == "dsm":
        _, *x_dim = X.shape
        x_dim = jnp.insert(jnp.array(x_dim)[::-1], 0, 1)
        ecgs = []
        for i in range(configs.n_ecgs):
            key1, key2, subkey = jr.split(subkey, 3)
            x_init = jr.uniform(key1, x_dim)
            xs = sample_annealed_langevin(
                result.apply_fn, x_init, result.params, key2
            )
            x = jnp.swapaxes(xs[-1].squeeze(), 0, 1)
            print(x.shape)
            if configs.processed:
                x = OMAT @ x
            ecgs.append(x)
            fig, _ = plot_ecg(x, CHANNELS, configs.n_channels, 
                              (6, configs.n_channels+1))
            fig.savefig(Path(gen_result_path, f"ecg_{i+1}.png"))
            if configs.find_closest_real:
                ecg_c, dist = find_closest_real_ecg(X, x)
                fig, _ = plot_ecg(
                    ecg_c, CHANNELS, configs.n_channels, 
                    (6, configs.n_channels+1),
                    title=f"Closest real ECG {i+1}, dist: {dist:.3f}"
                )
                fig.savefig(Path(gen_result_path, f"ecg_{i+1}_closest.png"))
            plt.close("all")
    else:
        raise ValueError(f"Dataset {configs.dataset} not supported.")
    ecgs = jnp.array(ecgs)
    # Visualize distribution
    if not configs.find_closest_real:
        ecg_mean, ecg_std = jnp.nanmean(ecgs, axis=0), jnp.nanstd(ecgs, axis=0)
        fig, _ = plot_ecg(ecg_mean, CHANNELS, configs.n_channels, 
                          (6, args.n_channels+1), ecg_std,
                          title=f"std: {jnp.mean(ecg_std):.3f}")
        fig.savefig(Path(gen_result_path, "ecg_dist.png"))
        
        plt.close("all")
    

def load_model(X, gen_ckpt_dir, configs):
    key = jr.PRNGKey(configs.seed)
    if configs.gen_model == "dr-vae":
        result = {}
        with open(Path(gen_ckpt_dir, "params_enc.npy"), "rb") as f:
            result["params_enc"] = jnp.load(f)
        with open(Path(gen_ckpt_dir, "params_dec.npy"), "rb") as f:
            result["params_dec"] = jnp.load(f)
        with open(Path(gen_ckpt_dir, "mu_mean.npy"), "rb") as f:
            result["mu_mean"] = jnp.load(f)
        with open(Path(gen_ckpt_dir, "mu_std.npy"), "rb") as f:
            result["mu_std"] = jnp.load(f)
        with open(Path(gen_ckpt_dir, "sigmasq_mean.npy"), "rb") as f:
            result["sigmasq_mean"] = jnp.load(f)
        with open(Path(gen_ckpt_dir, "sigmasq_std.npy"), "rb") as f:
            result["sigmasq_std"] = jnp.load(f)
        _, *x_dim = X.shape
        x_dim = jnp.array(x_dim)

        hidden_feats = [configs.hidden_width] * configs.hidden_depth
        decoder_feats = [*hidden_feats, jnp.prod(x_dim)]

        key_enc, key_dec = jr.split(key)

        # Encoder
        encoder = CNNEncoder(configs.z_dim)
        params_enc = encoder.init(key_enc, jnp.ones(x_dim,))['params']
        params_enc, unflatten_fn_enc = ravel_pytree(params_enc)
        apply_fn_enc = lambda params, x: encoder.apply(
            {'params': unflatten_fn_enc(params)}, x
        )

        # Decoder
        decoder = Decoder(decoder_feats, use_bias=False)
        params_dec = decoder.init(key_dec, jnp.ones(configs.z_dim,))['params']
        params_dec, unflatten_fn_dec = ravel_pytree(params_dec)
        apply_fn_dec = lambda params, x: decoder.apply(
            {'params': unflatten_fn_dec(params)}, x
        )
        result["apply_fn_enc"] = apply_fn_enc
        result["apply_fn_dec"] = apply_fn_dec
    elif configs.gen_model == "dsm":
        X_tr = jnp.swapaxes(X, 1, 2)
        model = models.NCSN(num_features=16)
        params = model.init(key, X_tr[0:1], jnp.array([0]))
        flat_params, unflatten_fn = ravel_pytree(params)
        print(f"Number of parameters: {len(flat_params):,}")
        apply_fn = lambda flat_params, x, y: model.apply(
            unflatten_fn(flat_params), x, y
        )
        state = create_train_state(flat_params, apply_fn, configs.lr_init)
        ckptr = orbax_ckpt.Checkpointer(orbax_ckpt.PyTreeCheckpointHandler())
        result = ckptr.restore(gen_ckpt_dir, item=state)
    else:
        raise ValueError(f"Model {configs.gen_model} not supported.")
    
    return result
    

def main(args):
    if args.reconstruct_real:
        assert args.gen_model == "dr-vae" # only supported for dr-vae
        
    X_tr, _, X_te, _, _ = load_dataset(args.dataset, args.beat_segment,
                                       args.processed, args.n_channels)
    dataset_name = args.dataset
    if args.processed:
        dataset_name += "_processed"
    gen_result_path = Path(
        result_path, "s03_generative_models", dataset_name, args.gen_model
    )
    if args.beat_segment:
        gen_result_path = Path(gen_result_path, f"bs_ch{args.n_channels}")
    else:
        gen_result_path = Path(gen_result_path, f"ch{args.n_channels}")
    if args.gen_model == "dr-vae":
        gen_result_path = Path(gen_result_path, f"beta1_{args.beta1_scheduler}",
                               f"beta1_{args.beta1}")
    if args.beat_segment:
        gen_ckpt_dir = Path(gen_result_path, f"bs_{args.n_channels}_ckpt")
    else:
        gen_ckpt_dir = Path(gen_result_path, f"{args.n_channels}_ckpt")
        
    if args.load_model:
        result = load_model(X_tr, gen_ckpt_dir, args)
    elif args.gen_model != "real":
        result = train_and_save_model(X_tr, gen_ckpt_dir, dataset_name, args)
    
    if args.gen_model == "real" or args.reconstruct_real:
        key, subkey = jr.split(jr.PRNGKey(args.seed))
        ecgs = X_tr
        if args.n_ecgs > 0:
            idx = jr.choice(key, len(ecgs), shape=(args.n_ecgs,), replace=False)
            ecgs = ecgs[idx]
        gen_ecg_path = Path(gen_result_path, "generated_ecgs")
        gen_ecg_path.mkdir(parents=True, exist_ok=True)
        rmses = []
        if args.gen_model == "real":
            for i, x in enumerate(tqdm.tqdm(ecgs)):
                if args.processed:
                    x = OMAT @ x
                fig, _ = plot_ecg(x, CHANNELS, args.n_channels, 
                                  (6, args.n_channels+1),
                                  title=f"Real ECG {i+1}")
                fig.savefig(Path(gen_ecg_path, f"ecg_{i+1}.png"))
            ecg_mean, ecg_std = \
                jnp.nanmean(ecgs, axis=0), jnp.nanstd(ecgs, axis=0)
            if args.processed:
                ecg_mean = OMAT @ ecg_mean
            fig, _ = plot_ecg(ecg_mean, CHANNELS, args.n_channels, 
                              (6, args.n_channels+1), ecg_std,
                            title=f"std: {jnp.mean(ecg_std):.3f}")
            fig.savefig(Path(gen_ecg_path, "ecg_dist.png"))
            plt.close("all")
        else:
            fn_enc, params_enc = result["apply_fn_enc"], result["params_enc"]
            fn_dec, params_dec = result["apply_fn_dec"], result["params_dec"]
            
            def _reconstruct(x, key):
                mu, sigmasq = fn_enc(params_enc, x)
                z = gaussian_sample(key, mu, sigmasq)
                x_rec = fn_dec(params_dec, z).reshape(x.shape)
                if args.processed:
                    x = OMAT @ x
                    x_rec = OMAT @ x_rec
                rmse = jnp.sqrt(jnp.mean((x - x_rec)**2))
                
                return x_rec, rmse
            
            # Compute RMSEs
            for XX, X_name in [(X_tr, "tr"), (X_te, "te")]:
                print(f"\nX_{X_name} RMSEs:")
                keys = jr.split(subkey, len(XX)+1)
                keys, subkey = keys[:-1], keys[-1]
                _, rmses = jax.vmap(_reconstruct)(XX, keys)
                print(f"\tmean: {jnp.mean(rmses):.3f}")
                print(f"\tstd: {jnp.std(rmses):.3f}")
                print(f"\tmax: {jnp.max(rmses):.3f}")
                print(f"\tmin: {jnp.min(rmses):.3f}")
                print(f"\targmax: {jnp.argmax(rmses)}")
                save_path = Path(gen_result_path, f"rec_rmses_{X_name}.npy")
                with open(save_path, "wb") as f:
                    jnp.save(f, rmses)
                # Plot reconstructed ECG for the argmax RMSE
                key, subkey = jr.split(subkey)
                x_ma = XX[jnp.argmax(rmses)]
                x_rec, rmse = _reconstruct(x_ma, key)
                if args.processed:
                    x_ma = OMAT @ x_ma
                fig, _ = plot_ecg(x_ma, CHANNELS, args.n_channels,
                                (6, args.n_channels+1),
                                title=f"Real ECG with max RMSE")
                fig.savefig(Path(gen_ecg_path, f"ecg_{X_name}_rmse_argmax.png"))
                
                fig, _ = plot_ecg(x_rec, CHANNELS, args.n_channels,
                                (6, args.n_channels+1),
                                title=f"Reconstructed ECG with max RMSE, " 
                                f"RMSE: {rmse:.3f}")
                fig.savefig(
                    Path(gen_ecg_path, f"ecg_{X_name}_rmse_argmax_rec.png")
                )
                plt.close("all")
            # Plot reconstructed ECGs
            for i, x in enumerate(ecgs):
                key, subkey = jr.split(subkey)
                x_rec, rmse = _reconstruct(x, key)
                fig, _ = plot_ecg(x_rec, CHANNELS, args.n_channels,
                                (6, args.n_channels+1),
                                title=f"Reconstructed ECG {i+1}, " 
                                f"RMSE: {rmse:.3f}")
                fig.savefig(Path(gen_ecg_path, f"ecg_{i+1}_rec.png"))
                plt.close("all")
    else:
        ecgs = generate_and_save_ecgs(X_tr, result, gen_result_path, args)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and generate ECGs")
    
    parser.add_argument("--load_model", action="store_true",)
    # Find the closest real ECG   
    parser.add_argument("--find_closest_real", action="store_true",)
    # Reconstruct real ECGs using model
    parser.add_argument("--reconstruct_real", action="store_true",)
    
    # Specify the generative model to train
    parser.add_argument("--gen_model", type=str, default="dr-vae",
                        choices=["dr-vae", "baseline", "dsm", "real"],)
    
    # Specify model parameters for VAE
    parser.add_argument("--z_dim", type=int, default=512) # latent dim
    parser.add_argument("--beta1", type=float, default=1.0) # KL-div reg. weight
    parser.add_argument("--beta2", type=float, default=0.0) # disc. reg. weight
    parser.add_argument("--hidden_width", type=int, default=100) # hidden layer width
    parser.add_argument("--hidden_depth", type=int, default=4) # hidden layer depth
    parser.add_argument("--lr_init", type=float, default=1e-7) # initial learning rate
    parser.add_argument("--lr_peak", type=float, default=1e-4) # peak learning rate
    parser.add_argument("--lr_end", type=float, default=1e-7) # end learning rate
    parser.add_argument("--target", type=str, default="age") # target for discriminator
    
    parser.add_argument("--beta1_scheduler", type=str, default="constant",
                        choices=["constant", "linear", "cosine",
                                 "warmup_cosine", "cyclical"],)
                        
    
    # Specify dataset
    parser.add_argument("--dataset", type=str, default="ptb-xl",
                        choices=["mimic-iv", "ptb-xl"],)
    parser.add_argument("--n_channels", type=int, default=12) # number of channels to use
    parser.add_argument("--beat_segment", action="store_true") # use beat segmentations
    parser.add_argument("--processed", action="store_true") # use processed dataset
    
    # Specify training parameters
    parser.add_argument("--seed", type=int, default=0) # random seed
    parser.add_argument("--n_epochs", type=int, default=100) # number of epochs to train
    parser.add_argument("--batch_size", type=int, default=512) # batch size
    
    # Specify number of generated ECGs
    parser.add_argument("--n_ecgs", type=int, default=100) # number of ECGs to generate
    
    args = parser.parse_args()
    
    main(args)