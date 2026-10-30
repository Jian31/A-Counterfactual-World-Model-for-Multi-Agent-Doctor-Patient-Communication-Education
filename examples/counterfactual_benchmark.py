import argparse

import numpy as np
import torch
from scipy.stats import kendalltau

from apwm import patient_scm as scm
from apwm.world_models import MODELS, abduct_eps, abduct_u, rollout, train

T, T0, K, N_RESAMPLE = 12, 6, 5, 20


def tensor(x):
    return torch.as_tensor(x, dtype=torch.float32)


def ground_truth(te, opts):
    finals = []
    for j in range(opts.shape[1]):
        acts = te["a"][:, T0:].copy()
        acts[:, 0] = opts[:, j]
        traj = scm.rollout(te["s"][:, T0], te["z"], te["u"], acts, te["eps"][:, T0:])
        finals.append(scm.utility(traj[:, -1]))
    return np.stack(finals, 1)


def predict(model, te, opts, procedure, seed=0):
    S, Z, A = tensor(te["s"]), tensor(te["z"]), tensor(te["a"])
    n = S.shape[0]
    u = abduct_u(model, S, Z, A) if model.latent and procedure == "abduction" else None
    eps = abduct_eps(model, S, Z, A, T0, u) if procedure == "abduction" else None
    gen = torch.Generator().manual_seed(seed)
    finals = []
    for j in range(opts.shape[1]):
        acts = A[:, T0:].clone()
        acts[:, 0] = tensor(opts[:, j])
        if procedure == "abduction":
            traj = rollout(model, S[:, T0], Z, acts, u, eps=eps)
            finals.append(scm.utility(traj[:, -1].numpy()))
        else:
            reps = []
            for _ in range(N_RESAMPLE):
                ur = torch.randn(n, 3, generator=gen) if model.latent else None
                traj = rollout(model, S[:, T0], Z, acts, ur, sample=True, gen=gen)
                reps.append(scm.utility(traj[:, -1].numpy()))
            finals.append(np.mean(reps, 0))
    return np.stack(finals, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    tr = scm.simulate(args.n_train, T, rng)
    te = scm.simulate(args.n_test, T, rng)
    opts = np.concatenate([te["a"][:, None, T0], rng.uniform(0, 1, (args.n_test, K, scm.NA))], 1)
    gt = ground_truth(te, opts)

    print(f"{'model':<28}{'procedure':<14}{'Kendall tau':>12}{'top-1':>8}")
    for name in ["resid", "gru", "aux_resid"]:
        model = train(MODELS[name](), tr, epochs=args.epochs, seed=args.seed)
        for procedure in ["resampling", "abduction"]:
            pred = predict(model, te, opts, procedure, args.seed)
            tau = np.nanmean([kendalltau(gt[i], pred[i])[0] for i in range(len(gt))])
            top1 = (pred.argmax(1) == gt.argmax(1)).mean()
            print(f"{name:<28}{procedure:<14}{tau:>12.3f}{top1:>8.3f}")


if __name__ == "__main__":
    main()
