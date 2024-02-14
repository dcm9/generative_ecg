import argparse
from pathlib import Path

from flax.training import orbax_utils
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import jax.random as jr
import orbax.checkpoint as orbax_ckpt

from mlscience_ekgs.settings import (
    mimic_ecg_path, mimic_ecg_parsed_path, ptb_path, result_path
)
from mlscience_ekgs.Code.src.s01_data_loader import load_ptb_xl_dataset
from mlscience_ekgs.Code.src.s03_dr_vae import CNNEncoder, Decoder, train_dr_vae
import mlscience_ekgs.Code.src.s04_models as models
from mlscience_ekgs.Code.src.s06_utils import plot_ecg
from mlscience_ekgs.Code.src.s08_dsm import (
    create_train_state, 
    sample_annealed_langevin,
    train_dsm
)


CHANNELS = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 
            'V4', 'V5', 'V6']


def load_dataset(dataset, beat_segment, brady_tachy_subset, 
                 n_channels, verbose=True, target="age"):
    if dataset == "ptb-xl":
        X, y = load_ptb_xl_dataset(segmentation=beat_segment,
                                   target=target)
    else:
        raise ValueError(f"Dataset {dataset} not supported.")
    # Take only the first n channels
    X = X[:, :n_channels, :]
    if verbose:
        print(
            f"Loaded dataset {dataset} with shape X: {X.shape}, y: {y.shape}."
        )
    
    return X, y
    

def train_and_save_model(X, gen_ckpt_dir, configs):
    # Load discriminative model
    ckpt_dir = Path(result_path, "s02_discriminative_models", 
                        configs.dataset, configs.target)
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
    
    if configs.gen_model == "dr-vae":
        result = train_dr_vae(vae_pred_fn, X, configs.alpha, configs.beta,
                              configs.z_dim, configs.seed, configs.n_epochs,
                              configs.batch_size, configs.hidden_width,
                              configs.hidden_depth, configs.lr_init,
                              configs.lr_peak, configs.lr_end,
                              encoder_type="cnn", use_bias=False)
        # drvae_result_path = Path(result_path, "s03_ptbxl", "s01_dr_vae")
        # gen_ckpt_dir = Path(drvae_result_path, "dr_vae_ckpt")
        gen_ckpt_dir.mkdir(parents=True, exist_ok=True)
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

    
def generate_and_save_ecgs(X, result, gen_result_path, configs):
    gen_result_path = Path(gen_result_path, "generated_ecgs")
    gen_result_path.mkdir(parents=True, exist_ok=True)
    key = jr.PRNGKey(configs.seed)
    key, subkey = jr.split(key)
    if configs.gen_model == "dr-vae":
        fn_dec, params_dec = result["apply_fn_dec"], result["params_dec"]
        mu_mean, mu_std = result["mu_mean"], result["mu_std"]
        sigmasq_mean, sigmasq_std = result["sigmasq_mean"], result["sigmasq_std"]
        ecgs = []
        for i in range(configs.n_ecgs):
            key1, key2, key3, key  = jr.split(key, 4)
            mu_curr = mu_mean + mu_std*jr.normal(key1, shape=(configs.z_dim,))
            sigmasq_curr = sigmasq_mean + \
                sigmasq_std*jr.normal(key2, shape=(configs.z_dim,))
            z = mu_curr + \
                jnp.sqrt(sigmasq_curr)*jr.normal(key3, shape=(configs.z_dim,))
            x = fn_dec(params_dec, z)
            x = fn_dec(params_dec, z).reshape(configs.n_channels, -1)
            ecgs.append(x)
            fig, _ = plot_ecg(x, CHANNELS, configs.n_channels, 
                              (2*configs.n_channels, 6))
            fig.savefig(Path(gen_result_path, f"ecg_{i+1}.png"))
    elif configs.gen_model == "dsm":
        _, *x_dim = X.shape
        x_dim = jnp.insert(jnp.swapaxes(jnp.array(x_dim), 0, 1), 0, 1)
        ecgs = []
        for i in range(configs.n_ecgs):
            key1, key2, subkey = jr.split(subkey, 3)
            x_init = jr.uniform(key1, x_dim)
            xs = sample_annealed_langevin(
                result.apply_fn, x_init, result.params, key2
            )
            x = jnp.swapaxes(xs[-1].squeeze(), 0, 1)
            print(x.shape)
            ecgs.append(x)
            fig, _ = plot_ecg(x, CHANNELS, configs.n_channels, 
                              (2*configs.n_channels, 6))
            fig.savefig(Path(gen_result_path, f"ecg_{i+1}.png"))
    else:
        raise ValueError(f"Dataset {configs.dataset} not supported.")
    ecgs = jnp.array(ecgs)
    # Visualize distribution
    ecg_mean, ecg_std = jnp.nanmean(ecgs, axis=0), jnp.nanstd(ecgs, axis=0)
    # print(ecg_mean[:,50:150], '\n')
    # print(ecg_std[:,50:150])
    fig, _ = plot_ecg(ecg_mean, CHANNELS, 3, (4, 4), ecg_std)
    fig.savefig(Path(gen_result_path, "ecg_dist.png"))
    

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
        model = models.NCSN(num_features=16)
        params = model.init(key, X[0:1], jnp.array([0]))
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
    X_tr, _ = load_dataset(args.dataset, args.beat_segment,
                           args.brady_tachy_subset,
                           args.n_channels, verbose=True)
    gen_result_path = Path(
        result_path, "s03_generative_models", args.dataset, args.gen_model
    )
    if args.beat_segment:
        gen_ckpt_dir = Path(gen_result_path, f"bs_{args.n_channels}_ckpt")
    else:
        gen_ckpt_dir = Path(gen_result_path, f"{args.n_channels}_ckpt")
        
    if args.load_model:
        result = load_model(X_tr, gen_ckpt_dir, args)
    elif args.gen_model != "real":
        result = train_and_save_model(X_tr, gen_ckpt_dir, args)
    if args.gen_model == "real":
        key = jr.PRNGKey(args.seed)
        idx = jr.choice(key, len(X_tr), shape=(args.n_ecgs,), replace=False)
        ecgs = X_tr[idx]
        for i, x in enumerate(ecgs):
            fig, _ = plot_ecg(x, CHANNELS, args.n_channels, 
                              (2*args.n_channels, 6))
            drvae_result_path = Path(result_path, "s03_ptbxl", "s01_dr_vae", 
                                     "real_ecgs")
            drvae_result_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(Path(drvae_result_path, f"ecg_{i+1}.png"))
            
        ecg_mean, ecg_std = jnp.nanmean(ecgs, axis=0), jnp.nanstd(ecgs, axis=0)
        
        fig, _ = plot_ecg(ecg_mean, CHANNELS, 3, (4, 4), ecg_std)
        fig.savefig(Path(drvae_result_path, "ecg_dist.png"))
    else:
        ecgs = generate_and_save_ecgs(X_tr, result, gen_result_path, args)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and generate ECGs")
    
    parser.add_argument("--load_model", action="store_true",)
    
    # Specify the generative model to train
    parser.add_argument("--gen_model", type=str, default="dr-vae",
                        choices=["dr_vae", "baseline", "dsm", "real"],)
    
    # Specify model parameters for VAE
    parser.add_argument("--z_dim", type=int, default=512) # latent dim
    parser.add_argument("--alpha", type=float, default=1.0) # disc. reg. weight
    parser.add_argument("--beta", type=float, default=0.0) # disc. reg. weight
    parser.add_argument("--hidden_width", type=int, default=100) # hidden layer width
    parser.add_argument("--hidden_depth", type=int, default=4) # hidden layer depth
    parser.add_argument("--lr_init", type=float, default=1e-7) # initial learning rate
    parser.add_argument("--lr_peak", type=float, default=1e-4) # peak learning rate
    parser.add_argument("--lr_end", type=float, default=1e-7) # end learning rate
    parser.add_argument("--target", type=str, default="sex") # target for discriminator
    
    # Specify dataset
    parser.add_argument("--dataset", type=str, default="ptb-xl",
                        choices=["mimic-iv", "ptb-xl"],)
    parser.add_argument("--n_channels", type=int, default=3) # number of channels to use
    parser.add_argument("--beat_segment", action="store_true") # use beat segmentations
    parser.add_argument("--brady_tachy_subset", action="store_true") # use brady/tachy subset
    
    # Specify training parameters
    parser.add_argument("--seed", type=int, default=0) # random seed
    parser.add_argument("--n_epochs", type=int, default=100) # number of epochs to train
    parser.add_argument("--batch_size", type=int, default=512) # batch size
    
    # Specify number of generated ECGs
    parser.add_argument("--n_ecgs", type=int, default=100) # number of ECGs to generate
    
    args = parser.parse_args()
    
    main(args)