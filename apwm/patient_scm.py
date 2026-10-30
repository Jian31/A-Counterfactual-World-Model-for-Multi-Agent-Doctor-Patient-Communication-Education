import numpy as np

S = ["anxiety", "sadness", "anger", "trust", "understanding", "coping"]
Z = ["baseline_anxiety", "health_literacy", "numeric_preference", "prior_distrust", "news_severity"]
A = [
    "empathy",
    "info_amount",
    "jargon",
    "numeric_risk",
    "teach_back",
    "invite_pause",
    "show_image",
    "premature_reassurance",
]
AP = ["relevance", "goal_congruence", "certainty", "controllability", "coping_potential"]
NS, NZ, NA, NU, NAP = len(S), len(Z), len(A), 3, len(AP)
SIGMA = 0.03
DIRECT = 0.0
sig = lambda x: 1 / (1 + np.exp(-x))


def appraisal(s, z, u, a):
    anx, sad, ang, tru, und, cop = s.T
    base, lit, nump, dist, sev = z.T
    ue, uj, ur = u.T
    emp, info, jar, num, tb, pause, img, pre = a.T
    jar_eff = jar * (1 - lit) * (1 + 0.5 * np.tanh(uj))
    overload = info * (0.5 + anx) * (1 - 0.5 * pause)
    relevance = sig(2.5 * sev + 1.5 * info - 1.0)
    congruence = sig(-2.0 * sev + 1.2 * emp * (1 + 0.6 * np.tanh(ue)) + 0.8 * pre - 0.5)
    certainty = sig(
        2.0 * info * (1 - jar_eff)
        + 1.2 * tb
        + 0.6 * img
        + 1.0 * num * (nump - 0.4)
        - 1.5 * overload
        - 0.6 * pre * und
        - 0.3
    )
    control = sig(1.2 * num * nump + 0.8 * tb + 0.7 * pause - 0.8 * jar_eff - 0.4)
    coping_p = sig(1.5 * emp * (1 + 0.6 * np.tanh(ue)) + 1.0 * pause + 0.6 * tb - 1.2 * overload + 0.8 * tru - 0.8)
    return np.stack([relevance, congruence, certainty, control, coping_p], 1)


def transition(s, z, u, a, eps):
    ap = appraisal(s, z, u, a)
    rel, con, cer, ctl, cpp = ap.T
    anx, sad, ang, tru, und, cop = s.T
    base, lit, nump, dist, sev = z.T
    react = 0.6 + 0.4 * sig(1.5 * u[:, 2])
    pre = a[:, 7]
    d = np.zeros_like(s)
    d[:, 0] = react * (0.22 * rel * (1 - con) - 0.16 * cer * cpp - 0.08 * ctl) + 0.04 * (base - anx)
    d[:, 1] = react * (0.15 * rel * (1 - con) * sev - 0.08 * cpp)
    d[:, 2] = react * (0.12 * (1 - cer) * (1 - con) * dist + 0.10 * pre * und - 0.06 * cpp)
    d[:, 3] = 0.10 * cpp + 0.06 * cer - 0.10 * (1 - cer) * dist - 0.12 * pre * und - 0.03 * ang - 0.03
    d[:, 4] = 0.16 * cer * (1 - 0.5 * anx) + 0.05 * ctl - 0.075
    d[:, 5] = 0.12 * cpp + 0.08 * ctl - 0.06 * anx - 0.09
    if DIRECT:
        jar, num, img = a[:, 2], a[:, 3], a[:, 6]
        d[:, 2] += DIRECT * 0.06 * (jar - 0.4)
        d[:, 0] += DIRECT * 0.06 * (img - 0.5) * (1 - z[:, 1])
        d[:, 3] += DIRECT * 0.05 * (num - 0.5) * (z[:, 2] - 0.5) * 2
    return np.clip(s + d + eps, 0, 1), d, ap


def sample_patients(n, rng):
    z = np.c_[rng.beta(2, 3, n), rng.beta(3, 2, n), rng.uniform(0, 1, n), rng.beta(2, 4, n), rng.uniform(0.3, 1, n)]
    u = rng.normal(0, 1, (n, NU))
    s0 = np.c_[
        np.clip(z[:, 0] + 0.2 * z[:, 4] + rng.normal(0, 0.05, n), 0, 1),
        np.clip(0.3 * z[:, 4] + rng.normal(0, 0.05, n), 0, 1),
        np.clip(0.2 * z[:, 3] + rng.normal(0, 0.05, n), 0, 1),
        np.clip(0.6 - 0.4 * z[:, 3] + rng.normal(0, 0.05, n), 0, 1),
        np.clip(0.2 + 0.2 * z[:, 1] + rng.normal(0, 0.05, n), 0, 1),
        np.clip(0.5 - 0.2 * z[:, 0] + rng.normal(0, 0.05, n), 0, 1),
    ]
    return z, u, s0


def policy(s, z, skill, rng, randomized=False):
    n = len(s)
    if randomized:
        return rng.uniform(0, 1, (n, NA))
    anx, sad, ang, tru, und, cop = s.T
    p = np.c_[
        sig(2 * skill + 2.0 * anx - 1.5),
        sig(1.5 - 1.5 * anx + 0.5 * skill),
        sig(0.5 - 2 * skill),
        sig(-0.5 + skill),
        sig(-1.5 + 2 * skill),
        sig(-1 + 1.5 * skill + anx),
        sig(-1 + skill),
        sig(0.5 + 2 * anx - 2.5 * skill),
    ]
    return np.clip(p + rng.normal(0, 0.15, p.shape), 0, 1)


def simulate(n, T, rng, randomized=False):
    z, u, s = sample_patients(n, rng)
    skill = rng.normal(0, 1, n)[:, None].repeat(1, 1)[:, 0]
    S_, A_, E_, D_, AP_ = [s], [], [], [], []
    for t in range(T):
        a = policy(s, z, skill, rng, randomized)
        eps = rng.normal(0, SIGMA, (n, NS))
        s, d, ap = transition(s, z, u, a, eps)
        S_.append(s)
        A_.append(a)
        E_.append(eps)
        D_.append(d)
        AP_.append(ap)
    return dict(z=z, u=u, s=np.stack(S_, 1), a=np.stack(A_, 1), eps=np.stack(E_, 1), app=np.stack(AP_, 1), skill=skill)


def rollout(s0, z, u, actions, eps):
    out = [s0]
    s = s0
    for k in range(actions.shape[1]):
        s, _, _ = transition(s, z, u, actions[:, k], eps[:, k])
        out.append(s)
    return np.stack(out, 1)


def utility(s):
    return s[..., 3] + s[..., 4] + s[..., 5] - s[..., 0] - s[..., 1] - s[..., 2]
