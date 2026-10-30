import torch, torch.nn as nn
from .patient_scm import NS, NZ, NA, NAP

DU = 3


def mlp(i, o, h=64):
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, o))


class Base(nn.Module):
    latent = False
    residual = True
    recurrent = False

    def __init__(self):
        super().__init__()
        self.logsig = nn.Parameter(torch.full((NS,), -3.0))

    def mean_next(self, s, z, a, u=None, h=None):
        d, h = self.delta(s, z, a, u, h)
        return (s + d if self.residual else d), h


class FlatMLP(Base):
    residual = False
    name = "Flat MLP (predicts s_t+1)"

    def __init__(self):
        super().__init__()
        self.f = mlp(NS + NZ + NA, NS)

    def delta(self, s, z, a, u, h):
        return torch.sigmoid(self.f(torch.cat([s, z, a], -1))), None


class ResidMLP(Base):
    name = "Residual MLP"

    def __init__(self):
        super().__init__()
        self.f = mlp(NS + NZ + NA, NS)

    def delta(self, s, z, a, u, h):
        return 0.3 * torch.tanh(self.f(torch.cat([s, z, a], -1))), None


class GRUWM(Base):
    recurrent = True
    name = "Recurrent WM (end-to-end)"

    def __init__(self, H=64):
        super().__init__()
        self.H = H
        self.cell = nn.GRUCell(NS + NZ + NA, H)
        self.f = mlp(H + NS + NZ + NA, NS)

    def delta(self, s, z, a, u, h):
        x = torch.cat([s, z, a], -1)
        if h is None:
            h = torch.zeros(s.shape[0], self.H)
        out = 0.3 * torch.tanh(self.f(torch.cat([h, x], -1)))
        return out, self.cell(x, h)


class LatentMLP(Base):
    latent = True
    name = "APWM w/o appraisal layer"

    def __init__(self):
        super().__init__()
        self.f = mlp(NS + NZ + NA + DU, NS)

    def delta(self, s, z, a, u, h):
        return 0.3 * torch.tanh(self.f(torch.cat([s, z, a, u], -1))), None


class APWM(Base):
    latent = True
    name = "APWM"

    def __init__(self, app_sup=False):
        super().__init__()
        self.g = mlp(NS + NZ + NA + DU, NAP)
        self.h = mlp(NAP + NS + NZ + DU, NS)
        self.app_sup = app_sup
        if app_sup:
            self.name = "APWM + weak appraisal labels"

    def appraise(self, s, z, a, u):
        return torch.sigmoid(self.g(torch.cat([s, z, a, u], -1)))

    def delta(self, s, z, a, u, h):
        ap = self.appraise(s, z, a, u)
        return 0.3 * torch.tanh(self.h(torch.cat([ap, s, z, u], -1))), None


class APWMHybrid(APWM):
    name = "APWM-hybrid (soft bottleneck)"

    def __init__(self, app_sup=True, lam=0.1):
        super().__init__(app_sup=app_sup)
        self.d = mlp(NS + NZ + NA + DU, NS)
        self.lam = lam
        self.name = "APWM-hybrid (soft bottleneck)"

    def delta(self, s, z, a, u, h):
        ap = self.appraise(s, z, a, u)
        main = 0.3 * torch.tanh(self.h(torch.cat([ap, s, z, u], -1)))
        direct = 0.3 * torch.tanh(self.d(torch.cat([s, z, a, u], -1)))
        self._pen = ((direct / self.logsig.exp()) ** 2).sum(-1)
        return main + direct, None


class AuxResid(Base):
    name = "Residual MLP + appraisal auxiliary head"
    app_sup = True

    def __init__(self, H=64):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(NS + NZ + NA, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU())
        self.head, self.aux = nn.Linear(H, NS), nn.Linear(H, NAP)

    def appraise(self, s, z, a, u):
        return torch.sigmoid(self.aux(self.trunk(torch.cat([s, z, a], -1))))

    def delta(self, s, z, a, u, h):
        return 0.3 * torch.tanh(self.head(self.trunk(torch.cat([s, z, a], -1)))), None


MODELS = {
    "flat": FlatMLP,
    "resid": ResidMLP,
    "gru": GRUWM,
    "latent_noapp": LatentMLP,
    "apwm": APWM,
    "apwm_sup": lambda: APWM(app_sup=True),
    "hybrid": lambda: APWMHybrid(app_sup=True),
    "aux_resid": AuxResid,
}


def nll(model, S, Z, A, U, APP=None, w_app=1.0):
    B, T = A.shape[:2]
    h, loss = None, 0.0
    sig = model.logsig.exp()
    app_loss = 0.0
    for t in range(T):
        mu, h = model.mean_next(S[:, t], Z, A[:, t], U, h)
        loss = loss + (((S[:, t + 1] - mu) / sig) ** 2 / 2 + model.logsig).sum(-1)
        if getattr(model, "_pen", None) is not None and model.training:
            loss = loss + model.lam * model._pen
        if APP is not None and getattr(model, "app_sup", False):
            app_loss = app_loss + ((model.appraise(S[:, t], Z, A[:, t], U) - APP[:, t]) ** 2).sum(-1)
    return loss, app_loss


def train(model, data, epochs=300, lr=3e-3, bs=256, seed=0, app_noise=0.1, log=None):
    torch.manual_seed(seed)
    S, Z, A = [torch.tensor(data[k], dtype=torch.float32) for k in ("s", "z", "a")]
    APP = torch.tensor(data["app"], dtype=torch.float32)
    APP = (APP + app_noise * torch.randn_like(APP)).clamp(0, 1)
    n = S.shape[0]
    U = nn.Parameter(torch.zeros(n, DU)) if model.latent else None
    params = list(model.parameters()) + ([U] if U is not None else [])
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            u = U[idx] if U is not None else None
            l, la = nll(model, S[idx], Z[idx], A[idx], u, APP[idx])
            loss = l.mean()
            if u is not None:
                loss = loss + 0.5 * (u**2).sum(-1).mean()
            if getattr(model, "app_sup", False):
                loss = loss + 10.0 * la.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        if log is not None and (ep % 50 == 0 or ep == epochs - 1):
            log(f"  ep {ep:3d} loss {tot / n:.3f}")
    model.eval()
    return model


def abduct_u(model, S, Z, A, steps=200, lr=0.05):
    if not model.latent:
        return None
    S, Z, A = [torch.as_tensor(x, dtype=torch.float32) for x in (S, Z, A)]
    u = torch.zeros(S.shape[0], DU, requires_grad=True)
    opt = torch.optim.Adam([u], lr=lr)
    for p in model.parameters():
        p.requires_grad_(False)
    for _ in range(steps):
        l, _ = nll(model, S, Z, A, u)
        loss = l.sum() + 0.5 * (u**2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    for p in model.parameters():
        p.requires_grad_(True)
    return u.detach()


@torch.no_grad()
def history_state(model, S, Z, A):
    h = None
    for t in range(A.shape[1]):
        _, h = model.mean_next(S[:, t], Z, A[:, t], None, h)
    return h


@torch.no_grad()
def rollout(model, s0, Z, actions, u=None, h=None, eps=None, sample=False, gen=None):
    s, out = s0, [s0]
    sig = model.logsig.exp()
    for k in range(actions.shape[1]):
        mu, h = model.mean_next(s, Z, actions[:, k], u, h)
        if eps is not None:
            mu = mu + eps[:, k]
        elif sample:
            mu = mu + sig * torch.randn(mu.shape, generator=gen)
        s = mu.clamp(0, 1)
        out.append(s)
    return torch.stack(out, 1)


@torch.no_grad()
def abduct_eps(model, S, Z, A, t0, u=None):
    h = history_state(model, S, Z, A[:, :t0]) if model.recurrent else None
    eps = []
    for k in range(t0, A.shape[1]):
        mu, h = model.mean_next(S[:, k], Z, A[:, k], u, h)
        eps.append(S[:, k + 1] - mu)
    return torch.stack(eps, 1)
