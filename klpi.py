"""Simulations of Section VI of "KL-Regularized Policy Iteration for Adaptive LQR: Gain Updates and Probing from One Policy Improvement" [1].

Dependencies: numpy, scipy (third-party).

  python klpi.py --quick    24 seeds per plant (6 per random plant), about 8 min on 20 cores,
                            written to results/quick/
  python klpi.py            the paper's 60960 runs, about 2 h 10 min on 20 cores, written to
                            results/

Each run writes the per-run files of its setting; ``write_tables`` turns them into the tables
that the figure sources in figures/ read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
from scipy.linalg import lapack
from scipy.linalg import solve_discrete_are as dare

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")


# ================================================================================== plants
@dataclass(frozen=True)
class Plant:
    """Plant ``x' = A x + B u + w``, ``w ~ N(0, Sigma_W)``, stage cost ``x'Qx + u'Ru``."""

    name: str
    A: np.ndarray
    B: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    Sigma_W: np.ndarray

    @property
    def n(self) -> int:
        return self.B.shape[0]

    @property
    def m(self) -> int:
        return self.B.shape[1]

    def matrices(self) -> tuple[np.ndarray, ...]:
        """``(A, B, Q, R, Sigma_W)``."""
        return self.A, self.B, self.Q, self.R, self.Sigma_W


def make(name: str, A, B, Q=None, R=None, Sigma_W=None) -> Plant:
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    n, m = B.shape
    Q = np.eye(n) if Q is None else np.atleast_2d(np.asarray(Q, dtype=float))
    R = np.eye(m) if R is None else np.atleast_2d(np.asarray(R, dtype=float))
    Sigma_W = np.eye(n) if Sigma_W is None else np.atleast_2d(np.asarray(Sigma_W, float))
    return Plant(name, A, B, Q, R, Sigma_W)


A_P = np.array([[0.9, 0.3], [0.0, 0.95]])
A_P1 = np.array([[-0.13, 0.14, -0.29, 0.28], [0.48, 0.09, 0.41, 0.30],
                 [-0.01, 0.04, 0.17, 0.43], [0.14, 0.31, -0.29, -0.10]])
B_P1 = np.array([[1.63, 0.93], [0.26, 1.79], [1.46, 1.18], [0.77, 0.11]])
B_P3 = 9.927459762687201   # gives P3 the input spread of P2


def paper_plants() -> dict[str, Plant]:
    """P1-P4 of Table II: P1 the DeePO benchmark, P2 spread from R, P3 spread from B, P4
    diagonal with the cheap input driving the slow state."""
    return {"P1": make("P1", A_P1, B_P1),
            "P2": make("P2", A_P, np.eye(2), R=np.diag([1.0, 100.0])),
            "P3": make("P3", A_P, np.diag([1.0, B_P3])),
            "P4": make("P4", np.diag([0.0, 0.9]), np.diag([2.0, 1.0]), R=np.diag([100.0, 1.0]))}


def random_plants() -> dict[str, Plant]:
    """The forty random plants of Section VI (n = 3, m = 2), stored in data/random_plants.json;
    ids 1 to 40 of Fig. 1(b) in this order."""
    with open(os.path.join(DATA, "random_plants.json")) as fh:
        specs = json.load(fh)["plants"]
    return {d["name"]: make(d["name"], d["A"], d["B"], R=np.diag([1.0, d["r"]])) for d in specs}


def spectral_radius(M: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(M))))


# =========================================================================== adaptive loop
# The engine below performs, sample by sample, exactly the floating-point operations of the
# code that produced the paper's runs, so for the same seed it reproduces them bit for bit
# (with the paper's linear-algebra libraries; with others, to about 1e-11 relative).
#
# Random numbers of one seed (common to all variants): offline inputs and noise
# default_rng([seed, 0]); online process noise default_rng([seed, 1]); the standard-normal
# vectors xi_t of the probing perturbation Omega_t = Psi_t^{1/2} xi_t default_rng([seed, 2]).
# Starts: t0 = 100 offline samples from X_0 = 0 with inputs N(0, I_m); 'standard' uses the
# certainty-equivalent gain of the fitted plant estimate, 'poor' that of the estimate fitted to
# the first eight offline samples (the loop still uses all t0 samples).
# Variants (Table I): every variant starts from H_t0 = S_off / (1 - alpha) and updates
# K_{t+1} = K_t - H_{t+1}^{-1} E_t;
#   'KLPI'         H_{t+1} = alpha H_t + S_t, probing Psi_t = (tau / 2) H_t^{-1}
#   'KLPI-iso'     KLPI's preconditioner, isotropic probing
#   'PGAC-iso'     H_{t+1} = S_t / (1 - alpha), isotropic probing
#   'PGAC-shaped'  PGAC's preconditioner, probing shaped like its own H_t^{-1}
# KLPI uses tau = 2 c / (m (1 - alpha)); the others match KLPI's probing cost Tr(S* Psi_t) at
# every sample (19). A K_t that does not stabilize the plant estimate skips the update, and a
# state above 1e4 stops the run; neither happens in the paper's runs.
BLOW = 1e4
T0 = 100
POOR_GAIN_SAMPLES = 8
VARIANTS = ("PGAC-iso", "PGAC-shaped", "KLPI-iso", "KLPI")
# variant -> (preconditioner 'kl' (averaged) or 'gn' (current), probing shape)
SPEC = {"KLPI": ("kl", "own"), "KLPI-iso": ("kl", "iso"), "PGAC-iso": ("gn", "iso"),
        "PGAC-shaped": ("gn", "est")}


# --------------------------------------------------------------------------- plant objects
def lyap(a: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Solve ``a X a' - X + q = 0`` by the Kronecker method (as scipy for n < 10)."""
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(q))):
        raise ValueError("non-finite input to lyap")
    n = a.shape[0]
    lhs = np.eye(n * n) - (a[:, None, :, None] * a[None, :, None, :]).reshape(n * n, n * n)
    _, _, x, info = lapack.dgesv(lhs, q.flatten())
    if info != 0:
        raise np.linalg.LinAlgError("singular Lyapunov system")
    return x.reshape(q.shape)


class SimPlant:
    """A plant with the objects of its optimal gain, as used by the simulations:
    ``P_star``, ``S_star = R + B' P* B``, ``K_star``, ``C_star = Tr(P* Sigma_W)``."""

    def __init__(self, plant: "Plant"):
        self.plant = plant
        self.name = plant.name
        self.A, self.B, self.Q, self.R, self.SW = (np.asarray(x, dtype=float)
                                                    for x in plant.matrices())
        self.n, self.m = self.B.shape
        self.sw_id = bool(np.array_equal(self.SW, np.eye(self.n)))
        self.LW = np.linalg.cholesky(self.SW)
        self.P_star = dare(self.A, self.B, self.Q, self.R)
        self.S_star = self.R + self.B.T @ self.P_star @ self.B
        self.K_star = -np.linalg.solve(self.S_star, self.B.T @ self.P_star @ self.A)
        self.C_star = (float(np.trace(self.P_star)) if self.sw_id
                       else float(np.trace(self.P_star @ self.SW)))
        self.S_star_inv = np.linalg.inv(self.S_star)

    def cost_matrix(self, K, Ah=None, Bh=None) -> np.ndarray | None:
        """``P_K`` on the model ``(Ah, Bh)`` (default: the true plant); None if not Schur."""
        Ah = self.A if Ah is None else Ah
        Bh = self.B if Bh is None else Bh
        M = Ah + Bh @ K
        if max(abs(np.linalg.eigvals(M))) >= 1:
            return None
        return lyap(M.T, self.Q + K.T @ self.R @ K)

    def cost(self, K) -> float:
        """Average cost ``C(K) = Tr(P_K Sigma_W)``, ``inf`` if ``K`` does not stabilize."""
        P = self.cost_matrix(K)
        if P is None:
            return np.inf
        return float(np.trace(P)) if self.sw_id else float(np.trace(P @ self.SW))

    def ce_gain(self, Ah, Bh) -> np.ndarray:
        """Certainty-equivalent gain: the LQR gain of the model ``(Ah, Bh)``."""
        P = dare(Ah, Bh, self.Q, self.R)
        return -np.linalg.solve(self.R + Bh.T @ P @ Bh, Bh.T @ P @ Ah)

    def spread(self) -> float:
        """``lambda_max(S*) / lambda_min(S*)``."""
        ev = np.linalg.eigvalsh((self.S_star + self.S_star.T) / 2)
        return float(ev[-1] / ev[0])


_CACHE: dict[int, SimPlant] = {}


def sim_plant(plant) -> SimPlant:
    """``SimPlant`` of a plant (cached per plant object)."""
    if isinstance(plant, SimPlant):
        return plant
    k = id(plant)
    if k not in _CACHE or _CACHE[k].plant is not plant:
        _CACHE[k] = SimPlant(plant)
    return _CACHE[k]


def tau_for_budget(c: float, m: int, alpha: float) -> float:
    """Constant regularization weight ``tau = 2 c / (m (1 - alpha))``, for which KLPI's probing
    cost tends to the budget ``c`` on a known plant (Theorem 1(a), Proposition 2); rounded to
    12 significant digits as in the paper's runs."""
    return float(f"{2 * c / (m * (1 - alpha)):.12g}")


# --------------------------------------------------------------------------- offline phase
def offline_phase(sp: SimPlant, seed: int, t0: int):
    """``t0`` offline samples from ``X_0 = 0`` with inputs ``N(0, I_m)``. Returns
    ``(x, Phi, Xc)`` with the unnormalized sums ``Phi = sum z z'`` and ``Xc = sum x_next z'``,
    ``z = [u; x]``."""
    rng = np.random.default_rng([seed, 0])
    n, m = sp.n, sp.m
    x = np.zeros(n)
    Phi = np.zeros((n + m, n + m))
    Xc = np.zeros((n, n + m))
    for _ in range(t0):
        u = rng.normal(size=m)
        z = np.r_[u, x]
        w = rng.normal(size=n) if sp.sw_id else sp.LW @ rng.normal(size=n)
        xn = sp.A @ x + sp.B @ u + w
        Phi += np.outer(z, z)
        Xc += np.outer(xn, z)
        x = xn
    return x, Phi, Xc


def estimate(sp: SimPlant, Phi, Xc):
    """Least-squares estimate ``[B_hat A_hat] = Xc Phi^{-1}``; returns ``(A_hat, B_hat)``."""
    Th = Xc @ np.linalg.inv(Phi)
    return Th[:, sp.m:], Th[:, :sp.m]


def initial_condition(plant, seed: int, start: str = "standard", t0: int = T0):
    """Offline phase and initial gain of one seed (see the module docstring).

    Returns ``(x, Phi, Xc, A_hat, B_hat, K)`` at the first online sample, or None if the seed
    is not admissible."""
    sp = sim_plant(plant)
    x, Phi, Xc = offline_phase(sp, seed, t0)
    if np.linalg.matrix_rank(Phi) < Phi.shape[0]:
        return None
    try:
        Ah, Bh = estimate(sp, Phi, Xc)
        if start == "poor":
            _, Pg, Xg = offline_phase(sp, seed, POOR_GAIN_SAMPLES)  # first 8 of the t0 samples
            if np.linalg.matrix_rank(Pg) < Pg.shape[0]:
                return None
            Ag, Bg = estimate(sp, Pg, Xg)
            K = sp.ce_gain(Ag, Bg)
            if not np.isfinite(sp.cost(K)) or sp.cost_matrix(K, Ah, Bh) is None:
                return None
        elif start == "standard":
            K = sp.ce_gain(Ah, Bh)
            if not np.isfinite(sp.cost(K)):
                return None
        else:
            raise ValueError(f"unknown start {start!r}")
    except Exception:
        return None
    return x, Phi, Xc, Ah, Bh, K


# ----------------------------------------------------------------------- array kernels
def _kron_lyap(a, q):
    """Solve ``a X a' - X + q = 0`` for a batch ``(Bt, n, n)``."""
    Bt, n, _ = a.shape
    kr = (a[:, :, None, :, None] * a[:, None, :, None, :]).reshape(Bt, n * n, n * n)
    lhs = np.eye(n * n, dtype=a.dtype)[None] - kr
    x = _solve(lhs, q.reshape(Bt, n * n, 1))
    return x.reshape(Bt, n, n)


def _solve(a, b):
    """Batched solve; a singular member gives nan instead of failing the whole batch (this
    fallback is never reached in the paper's runs)."""
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        out = np.full(np.broadcast_shapes(a.shape[:-2], b.shape[:-2]) + b.shape[-2:], np.nan)
        for i in range(out.shape[0]):
            try:
                out[i] = np.linalg.solve(a[i], b[i])
            except np.linalg.LinAlgError:
                pass
        return out


def _inv(a):
    try:
        return np.linalg.inv(a)
    except np.linalg.LinAlgError:
        return _solve(a, np.broadcast_to(np.eye(a.shape[-1]), a.shape).copy())


def _sym(x):
    return 0.5 * (x + x.swapaxes(-1, -2))


def _tr(x):
    return x.diagonal(axis1=-2, axis2=-1).sum(-1)


def _sqrtm(P):
    """Symmetric square root of a batch of positive semidefinite matrices (closed form for
    2 x 2)."""
    m = P.shape[-1]
    if m == 2:
        s = np.sqrt(np.maximum(P[:, 0, 0] * P[:, 1, 1] - P[:, 0, 1] * P[:, 1, 0], 0.0))
        t = np.sqrt(P[:, 0, 0] + P[:, 1, 1] + 2 * s)
        return (P + s[:, None, None] * np.eye(2, dtype=P.dtype)[None]) / t[:, None, None]
    w, U = np.linalg.eigh(P)
    return (U * np.sqrt(np.maximum(w, 0.0))[:, None, :]) @ U.swapaxes(-1, -2)


def _posdef(P):
    ok = np.all(np.isfinite(P.reshape(P.shape[0], -1)), axis=1)
    Pc = np.where(ok[:, None, None], P, np.eye(P.shape[-1], dtype=P.dtype)[None])
    return ok & (np.linalg.eigvalsh(_sym(Pc))[:, 0] > 0)


def _step(k: int, c: dict, p: dict, Wb, Xb) -> dict:
    """One sample for all runs of a batch. c: state, p: parameters. Returns the new state."""
    A, Bm, Q, R, Ss = p["A"], p["B"], p["Q"], p["R"], p["Ss"]
    K, x, H, Shat = c["K"], c["x"], c["H"], c["Shat"]
    act = c["alive"]
    # gain-error cost of the applied gain on the true plant (cost-difference lemma)
    M = A + Bm @ K
    Sig = _kron_lyap(M, p["SW"])
    okT = _posdef(Sig)
    d = K - p["Ks"]
    g = _tr(Sig @ d.swapaxes(-1, -2) @ Ss @ d)
    gain = np.where(okT, g, np.inf)
    # probing covariance: KLPI's own, or matched in cost to the KLPI run of the same seed
    tau = p["tau"]
    m = Bm.shape[-1]
    Hs = np.where(p["kl"][:, None, None], H, np.eye(m, dtype=H.dtype)[None])
    psi_own = _sym(0.5 * tau[:, None, None] * _inv(Hs))
    pr_own = _tr(Ss @ psi_own)
    src = p["src"]
    pr = np.where(act[src], pr_own[src], p["budget"])
    Zest = _sym(_inv(Shat))
    Z = np.where(p["est"][:, None, None], Zest, np.eye(m)[None])
    psi_oth = Z * (pr / _tr(Ss @ Z))[:, None, None]
    Psi = np.where(p["own"][:, None, None], psi_own, psi_oth)
    price = _tr(Ss @ Psi)
    L = _sqrtm(Psi)
    slot = p["slot"]
    xi = Xb[slot, k]
    w = Wb[slot, k]
    u = (K @ x[:, :, None])[:, :, 0] + (L @ xi[:, :, None])[:, :, 0]
    dx = (d @ x[:, :, None])[:, :, 0]
    ge = np.einsum("bi,bij,bj->b", dx, Ss, dx)
    xn = (A @ x[:, :, None])[:, :, 0] + (Bm @ u[:, :, None])[:, :, 0] \
        + (p["LW"] @ w[:, :, None])[:, :, 0]
    # recursive least squares (as the running sums Phi, Xc)
    z = np.concatenate([u, x], axis=1)
    Phi = c["Phi"] + z[:, :, None] * z[:, None, :]
    Xc = c["Xc"] + xn[:, :, None] * z[:, None, :]
    Th = _solve(Phi, Xc.swapaxes(-1, -2)).swapaxes(-1, -2)
    Bh, Ah = Th[:, :, :m], Th[:, :, m:]
    blow = ~np.all(np.isfinite(xn), axis=1) | (np.sqrt((xn * xn).sum(1)) > BLOW)
    # gain update on the new estimate (skip rule)
    Me = Ah + Bh @ K
    Kt = K.swapaxes(-1, -2)
    P = _kron_lyap(Me.swapaxes(-1, -2), Q + Kt @ R @ K)
    okE = _posdef(P)
    Pc = np.where(okE[:, None, None], P, np.zeros_like(P))
    S = R + Bh.swapaxes(-1, -2) @ Pc @ Bh
    E = S @ K + Bh.swapaxes(-1, -2) @ Pc @ Ah
    Hn = p["alpha"][:, None, None] * H + S
    Mk = np.where(p["kl"][:, None, None], Hn, S)
    stp = _solve(Mk, E)
    fac = np.where(p["kl"], 1.0, 2 * p["eta"])
    Kn = K - fac[:, None, None] * stp
    okK = okE & np.all(np.isfinite(Kn.reshape(Kn.shape[0], -1)), axis=1)
    upd = act & ~blow & okK
    u3 = upd[:, None, None]
    a2 = act[:, None]
    a3 = act[:, None, None]
    return dict(
        K=np.where(u3, Kn, K),
        H=np.where(u3 & p["kl"][:, None, None], Hn, H),
        Shat=np.where(u3, S, Shat),
        x=np.where(a2, xn, x),
        Phi=np.where(a3, Phi, c["Phi"]),
        Xc=np.where(a3, Xc, c["Xc"]),
        alive=act & ~blow,
        skips=c["skips"] + (act & ~blow & ~okK),
        unstable=c["unstable"] + (act & ~okT),
        E_cum=c["E_cum"] + np.where(act, ge, 0.0),
        P_cum=c["P_cum"] + np.where(act, price, 0.0),
        gain=np.where(act, gain, c["gain"]),
        # per-sample costs of this step (nan once the run has stopped)
        _E=np.where(act, ge, np.nan), _P=np.where(act, price, np.nan),
    )


def checkpoints(T: int, npts: int = 70) -> np.ndarray:
    """0-based indices k of the log-spaced checkpoints at which ``simulate`` records the run
    (online sample number k + 1, from 1 to T)."""
    return np.unique(np.r_[np.round(np.logspace(0, np.log10(T), npts)).astype(int), T]) - 1


def dense_checkpoints(T: int) -> np.ndarray:
    """0-based indices of the dense grid on which the poor-gain start records the relative gap
    (Fig. 2(b)): every sample up to 200, 400 log-spaced samples from 1 to T, and the
    checkpoints of :func:`checkpoints`."""
    lg = np.round(np.logspace(0, np.log10(T), 400)).astype(int) - 1
    return np.unique(np.r_[np.arange(min(200, T)), lg, checkpoints(T)])


# ------------------------------------------------------------------------- many runs at once
def simulate(plant, seeds, c: float, variants=VARIANTS, start: str = "standard",
             T: int = 100_000, alpha: float = 0.9, t0: int = T0, record: str = "checkpoints",
             chk=None) -> dict:
    """Run every (variant, seed) combination together, one sample at a time.

    plant     a :class:`klpi.plants.Plant`; seeds: admissible seeds (klpi.experiments);
    c         probing budget per sample (the paper uses c = f C* with f in {1/64, ..., 2});
    record    'checkpoints': record at the samples ``chk`` (default ``checkpoints(T)``);
              'all': record every sample (memory T x runs; for short runs).
    Returns a dict with arrays indexed [variant, seed, ...]:
      E_sum, P_sum   summed gain-error and probing costs over the T online samples
      E_cum, P_cum   their running sums at the recorded samples, (nv, ns, nk)
      delta          relative gap delta_t (Section VI-D) at the recorded samples, (nv, ns, nk)
      status (0 ok, 1 blown up), skips, unstable (samples whose applied gain did not stabilize
      the plant); and k (the recorded 0-based sample indices), seeds, variants, C_star, tau.
    With record='all' it also carries E, P (per-sample costs) and delta of every sample.
    """
    sp = sim_plant(plant)
    variants = list(variants)
    run_vars = variants if "KLPI" in variants else variants + ["KLPI"]
    seeds = [int(s) for s in seeds]
    n, m = sp.n, sp.m
    nv, ns = len(run_vars), len(seeds)
    Bt = nv * ns
    tau = tau_for_budget(c, m, alpha)
    var_of = [run_vars[i // ns] for i in range(Bt)]
    slot = np.array([i % ns for i in range(Bt)], dtype=np.int32)
    jk = run_vars.index("KLPI")
    src = np.array([jk * ns + i % ns for i in range(Bt)], dtype=np.int32)
    inits = []
    for s in seeds:
        ic = initial_condition(sp, s, start, t0)
        if ic is None:
            raise RuntimeError(f"seed {s} is not admissible for {sp.name} ({start} start)")
        inits.append(ic)
    Wb = np.stack([np.random.default_rng([s, 1]).normal(size=(T, n)) for s in seeds])
    Xb = np.stack([np.random.default_rng([s, 2]).normal(size=(T, m)) for s in seeds])
    C: dict = {k_: [] for k_ in ("K", "x", "H", "Shat", "Phi", "Xc")}
    for i in range(Bt):
        x, Phi, Xc, Ah, Bh, K = inits[slot[i]]
        P0 = sp.cost_matrix(K, Ah, Bh)
        Sh = sp.R + Bh.T @ P0 @ Bh
        C["K"].append(K)
        C["x"].append(x)
        C["Phi"].append(Phi.copy())
        C["Xc"].append(Xc.copy())
        C["Shat"].append(Sh)
        C["H"].append(Sh / (1 - alpha) if SPEC[var_of[i]][0] == "kl" else np.eye(m))
    def rep(a):
        return np.array([a] * Bt)

    p = dict(A=rep(sp.A), B=rep(sp.B), Q=rep(sp.Q), R=rep(sp.R), SW=rep(sp.SW), LW=rep(sp.LW),
             Ss=rep(sp.S_star), Ks=rep(sp.K_star), budget=np.full(Bt, float(c)),
             tau=np.full(Bt, tau), alpha=np.full(Bt, float(alpha)),
             eta=np.full(Bt, (1 - alpha) / 2),
             kl=np.array([SPEC[v][0] == "kl" for v in var_of]),
             own=np.array([SPEC[v][1] == "own" for v in var_of]),
             est=np.array([SPEC[v][1] == "est" for v in var_of]), src=src, slot=slot)
    state = {k_: np.array(v, dtype=float) for k_, v in C.items()}
    z = np.zeros(Bt)
    state.update(alive=np.ones(Bt, bool), skips=np.zeros(Bt, np.int32),
                 unstable=np.zeros(Bt, np.int32), E_cum=z.copy(), P_cum=z.copy(), gain=z.copy())
    if record == "all":
        ks = np.arange(T)
    else:
        ks = checkpoints(T) if chk is None else np.asarray(chk)
    pos = {int(k_): j for j, k_ in enumerate(ks)}
    rec = {k_: np.full((Bt, len(ks)), np.nan) for k_ in ("E_cum", "P_cum", "delta")}
    if record == "all":
        rec.update(E=np.full((Bt, T), np.nan), P=np.full((Bt, T), np.nan))
    with np.errstate(all="ignore"):
        for k in range(T):
            state = _step(k, state, p, Wb, Xb)
            j = pos.get(k)
            if j is not None:
                rec["delta"][:, j] = state["gain"] / sp.C_star
                rec["E_cum"][:, j] = state["E_cum"]
                rec["P_cum"][:, j] = state["P_cum"]
            if record == "all":
                rec["E"][:, k] = state["_E"]
                rec["P"][:, k] = state["_P"]
    keep = [run_vars.index(v) for v in variants]

    def shaped(a):
        a = np.asarray(a)
        return a.reshape((nv, ns) + a.shape[1:])[keep]

    out = dict(variants=variants, seeds=seeds, k=ks, C_star=sp.C_star, tau=tau, c=float(c),
               T=T, alpha=alpha, start=start,
               E_sum=shaped(state["E_cum"]), P_sum=shaped(state["P_cum"]),
               status=shaped((~state["alive"]).astype(int)), skips=shaped(state["skips"]),
               unstable=shaped(state["unstable"]))
    for k_, v in rec.items():
        out[k_] = shaped(v)
    return out



# ========================================================================= simulation grid
# Three settings (Section VI-B), all with alpha = 0.9 and T = 1e5 online samples:
#   standard  P1-P4, standard start, budgets c = f C*, f in {1/64, ..., 2}, four variants,
#             the first 480 admissible seeds counted from 10000;
#   poor      P1-P4, poor-gain start, c = C*, four variants, the first 480 seeds counted from
#             2000000 that pass the poor-gain rule (C(K_t0) >= 2 C*, rho(A + B K_t0) <= 0.9,
#             rho(A_hat + B_hat K_t0) <= 0.85);
#   random    the forty random plants, standard start, c = C*/2, KLPI, PGAC-iso and
#             PGAC-shaped, 60 seeds run and the first 50 without an event analysed.
ALPHA = 0.9
T_RUN = 100_000
FRACTIONS = (1 / 64, 1 / 16, 1 / 4, 1 / 2, 1.0, 2.0)
N_SEEDS, N_SEEDS_RANDOM, N_SEEDS_RANDOM_RUN = 480, 50, 60
STANDARD_SEED0, POOR_SEED0 = 10_000, 2_000_000
POOR_RULE = dict(cost_ratio_min=2.0, rho_true_max=0.9, rho_est_max=0.85)
RANDOM_FRACTION, POOR_FRACTION, WINDOW_FRACTIONS = 0.5, 1.0, (1.0, 2.0)
LEGACY_NAME = {"KLPI": "KLPI", "PGAC-iso": "PGAC", "PGAC-shaped": "PGACest",
               "KLPI-iso": "KLPIiso"}   # variant names in the bootstrap labels of the paper

with open(os.path.join(DATA, "c_star.json")) as _fh:
    C_STAR_REF = json.load(_fh)["C_star"]   # C* as computed in the paper's runs


@dataclass
class Cell:
    """One configuration of the grid: a plant, a start and a budget, run for several seeds."""

    setting: str
    plant: Plant
    f: float          # budget as a multiple of C*: c = f C*
    seeds: list
    variants: tuple
    start: str
    T: int = T_RUN

    @property
    def c(self) -> float:
        return budget(self.plant, self.f)


def budget(plant: Plant, f: float) -> float:
    """Probing budget ``c = f C*``, with C* as in the paper's runs (bit-identical budgets)."""
    cs = C_STAR_REF.get(plant.name)
    return float(f * (cs if cs is not None else sim_plant(plant).C_star))


def standard_seeds(plant: Plant, n: int, start: int = STANDARD_SEED0) -> list[int]:
    """The first ``n`` admissible seeds counted from ``start`` for the standard start."""
    out, s = [], start
    while len(out) < n and s - start < 20000:
        if initial_condition(plant, s, "standard") is not None:
            out.append(s)
        s += 1
    return out


def poor_features(plant: Plant, seed: int) -> dict | None:
    """C(K_t0)/C*, rho(A + B K_t0) and rho(A_hat + B_hat K_t0) of a poor-gain start."""
    sp = sim_plant(plant)
    ic = initial_condition(sp, seed, "poor")
    if ic is None:
        return None
    _, _, _, Ah, Bh, K = ic
    return dict(cost_ratio=sp.cost(K) / sp.C_star, rho_true=spectral_radius(sp.A + sp.B @ K),
                rho_est=spectral_radius(Ah + Bh @ K))


def passes_poor_rule(feat: dict | None) -> bool:
    return (feat is not None and feat["cost_ratio"] >= POOR_RULE["cost_ratio_min"]
            and feat["rho_true"] <= POOR_RULE["rho_true_max"]
            and feat["rho_est"] <= POOR_RULE["rho_est_max"])


def _poor_check(args):
    plant, s = args
    return s, passes_poor_rule(poor_features(plant, s))


def poor_seeds(plant: Plant, n: int, start: int = POOR_SEED0, workers=None) -> list[int]:
    """The first ``n`` seeds counted from ``start`` that pass the poor-gain rule."""
    out, s = [], start
    with Pool(workers) as pool:
        while len(out) < n and s - start < 450_000:
            blk = [(plant, x) for x in range(s, s + 4000)]
            for sd, ok in pool.imap(_poor_check, blk, chunksize=100):
                if ok and len(out) < n:
                    out.append(sd)
            s = blk[-1][1] + 1
    return out


def cells(setting: str, n_seeds=None, T: int = T_RUN, workers=None) -> list[Cell]:
    """The cells of one setting (``n_seeds``: only the first n seeds, e.g. for a quick run)."""
    if setting == "random":
        ns = N_SEEDS_RANDOM_RUN if n_seeds is None else n_seeds
        return [Cell("random", pl, RANDOM_FRACTION, standard_seeds(pl, ns),
                     ("KLPI", "PGAC-iso", "PGAC-shaped"), "standard", T)
                for pl in random_plants().values()]
    ns = N_SEEDS if n_seeds is None else n_seeds
    out = []
    for pl in paper_plants().values():
        if setting == "standard":
            sd = standard_seeds(pl, ns)
            out += [Cell("standard", pl, f, sd, VARIANTS, "standard", T) for f in FRACTIONS]
        else:
            out.append(Cell("poor", pl, POOR_FRACTION, poor_seeds(pl, ns, workers=workers),
                            VARIANTS, "poor", T))
    return out


def bootstrap_label(cell: Cell, a: str, b: str) -> str:
    """Label whose hash seeds the bootstrap of the ratio of ``a`` to ``b`` in ``cell``; the
    format is that of the paper's code, so the paper's intervals are reproduced exactly."""
    mats = [tuple(tuple(float(v) for v in row) for row in np.asarray(M, dtype=float))
            for M in cell.plant.matrices()]
    pkey = hashlib.sha1(json.dumps(mats).encode()).hexdigest()[:12]
    d = dict(v="2", plant=pkey, t0=T0, koff=None,
             kinit="ce@8" if cell.start == "poor" else "ce", wkind="const", wval=cell.c,
             alpha=ALPHA, eta=(1 - ALPHA) / 2, T=cell.T, conv="new", extra={})
    key = hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]
    kind, tag = {"standard": ("g1std", "g1"), "poor": ("g1poor", "g1poor"),
                 "random": ("g1ens", "g1ens")}[cell.setting]
    return f"C7|C8|{kind}|{key}|{LEGACY_NAME[a]}|{LEGACY_NAME[b]}|ce|{tag}|None"


def rng_for(label: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha1(label.encode()).hexdigest()[:12], 16))


# ============================================================================== statistics
def ratio(x, y, rng: np.random.Generator, n_boot: int = 5000) -> dict:
    """Ratio ``r = sum_s x_s / sum_s y_s`` of summed gain-error costs over paired seeds (18),
    with the 95% paired-bootstrap interval (resample the seeds ``n_boot`` times)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    lo, hi = np.percentile(x[idx].mean(1) / y[idx].mean(1), [2.5, 97.5])
    return dict(r=float(x.mean() / y.mean()), lo=float(lo), hi=float(hi), n=len(x))


def windowed_ratio(ca, cb, k_edges, rng=None, n_boot: int = 2000) -> list[dict]:
    """Ratio of the gain-error costs of two variants summed over seeds and over the online
    samples of each window (Fig. 1(c)); ``ca``, ``cb`` are running sums at the window edges
    ``k_edges`` (seeds x edges)."""
    rng = np.random.default_rng(0) if rng is None else rng
    rows, prev = [], -1
    for j in range(ca.shape[1]):
        wa = ca[:, j] - (0 if prev < 0 else ca[:, prev])
        wb = cb[:, j] - (0 if prev < 0 else cb[:, prev])
        b = rng.integers(0, len(wa), (n_boot, len(wa)))
        lo, hi = np.quantile(wa[b].sum(1) / wb[b].sum(1), [0.025, 0.975])
        k0 = 1 if prev < 0 else int(k_edges[prev]) + 1
        rows.append(dict(k=float(np.sqrt(k0 * k_edges[j])), r=wa.sum() / wb.sum(), lo=lo,
                         hi=hi, rc=ca[:, j].sum() / cb[:, j].sum()))
        prev = j
    return rows


def log_window_ratio(k, num, den, edges=np.logspace(0, 5, 101)) -> list[tuple]:
    """Ratio ``num / den`` of two curves at the samples ``k``, averaged over logarithmic windows
    (20 per decade, Fig. 2(b)); one (geometric-mean sample, mean ratio) pair per window."""
    k = np.asarray(k, float)
    r = np.asarray(num, float) / np.asarray(den, float)
    rows = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        msk = (k >= lo) & ((k <= hi) if i == len(edges) - 2 else (k < hi))
        if msk.any():
            rows.append((float(np.exp(np.log(k[msk]).mean())), float(r[msk].mean())))
    return rows


# ========================================================================= per-run results
def _task(args) -> tuple:
    cid, setting, plant, f, seeds, variants, start, T = args
    chk = dense_checkpoints(T) if setting == "poor" else None
    r = simulate(plant, seeds, budget(plant, f), variants, start=start, T=T, alpha=ALPHA,
                 chk=chk)
    return cid, seeds, {k: r[k] for k in ("E_sum", "P_sum", "status", "skips", "unstable",
                                          "E_cum", "delta", "k")}


def run_setting(setting: str, workers=None, n_seeds=None, T: int = T_RUN) -> list[dict]:
    """Simulate every cell of a setting in parallel; one dict per cell, arrays [variant, seed]."""
    t_start = time.time()
    cs = cells(setting, n_seeds=n_seeds, T=T, workers=workers)
    tasks = [(i, setting, c.plant, c.f, c.seeds[j:j + 10], c.variants, c.start, T)
             for i, c in enumerate(cs) for j in range(0, len(c.seeds), 10)]
    print(f"{setting}: {len(cs)} cells, {len(tasks)} tasks")
    parts: dict = {}
    with Pool(workers) as pool:
        for n_done, (cid, seeds, res) in enumerate(pool.imap_unordered(_task, tasks), 1):
            parts.setdefault(cid, []).append((seeds, res))
            if n_done % max(1, len(tasks) // 10) == 0:
                print(f"  {n_done}/{len(tasks)} tasks, {time.time() - t_start:.0f} s")
    out = []
    for i, c in enumerate(cs):
        order = {s: j for j, s in enumerate(c.seeds)}
        got = sorted(parts[i], key=lambda q: order[q[0][0]])
        d = {k: np.concatenate([g[1][k] for g in got], axis=1)
             for k in ("E_sum", "P_sum", "status", "skips", "unstable", "E_cum", "delta")}
        d.update(plant=c.plant.name, f=c.f, seeds=list(c.seeds), variants=list(c.variants),
                 k=got[0][1]["k"], T=T)
        out.append(d)
    return out


def write_setting(setting: str, cell_results: list[dict], outdir: str = RESULTS) -> None:
    """Write the per-run file ``runs_<setting>.csv`` (one row per seed: plant, f, seed, the
    initial relative gap, the summed gain-error cost of each variant, KLPI's summed probing
    cost, the largest deviation of another variant's probing cost from it, and event counts),
    and the windowed sums (standard start) and relative gaps (poor-gain start) of the figures."""
    os.makedirs(outdir, exist_ok=True)
    variants = cell_results[0]["variants"]
    hdr = ["plant", "f", "seed", "delta0"] + [f"E_{v}" for v in variants] + \
        ["P_KLPI", "P_dev", "skips", "blowups", "unstable"]
    with open(os.path.join(outdir, f"runs_{setting}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        for d in cell_results:
            jk = d["variants"].index("KLPI")
            pk = d["P_sum"][jk]
            dev = np.max(np.abs(d["P_sum"] / pk[None] - 1), axis=0)
            for j, s in enumerate(d["seeds"]):
                w.writerow([d["plant"], f"{d['f']:g}", s, repr(float(d["delta"][jk, j, 0]))]
                           + [repr(float(d["E_sum"][i, j])) for i in range(len(variants))]
                           + [repr(float(pk[j])), f"{dev[j]:.3g}", int(d["skips"][:, j].sum()),
                              int(d["status"][:, j].sum()), int(d["unstable"][:, j].sum())])
    if setting == "standard":
        win = {}
        for d in cell_results:
            if d["f"] in WINDOW_FRACTIONS:
                rows = [d["variants"].index(v) for v in ("KLPI", "PGAC-iso")]
                win[f"{d['plant']}_f{d['f']:g}"] = d["E_cum"][rows]
                win["k_edges"] = np.asarray(d["k"]) + 1
        np.savez_compressed(os.path.join(outdir, "windows_standard.npz"), **win)
    if setting == "poor":
        gaps = {"k": np.asarray(cell_results[0]["k"])}
        for d in cell_results:
            rows = [d["variants"].index(v) for v in ("KLPI", "PGAC-iso", "PGAC-shaped",
                                                     "KLPI-iso")]
            gaps[d["plant"]] = d["delta"][rows].astype(np.float32)
        np.savez_compressed(os.path.join(outdir, "gaps_poor.npz"), **gaps)
    meta_f = os.path.join(outdir, "meta.json")
    meta = json.load(open(meta_f)) if os.path.exists(meta_f) else {}
    meta[setting] = dict(T=int(cell_results[0]["T"]), alpha=ALPHA, variants=list(variants),
                         seeds={d["plant"]: len(d["seeds"]) for d in cell_results})
    with open(meta_f, "w") as fh:
        json.dump(meta, fh, indent=1)


def read_setting(setting: str, outdir: str = RESULTS) -> dict:
    """Per-run results of one setting: ``{(plant, f): dict(seeds, delta0, E={variant: array},
    P_KLPI, P_dev, events)}`` plus ``"_T"``. For the random plants only the first 50 seeds
    without an event are kept (in the paper's runs this skips seed 10001 of EN25)."""
    out: dict = {}
    with open(os.path.join(outdir, f"runs_{setting}.csv")) as fh:
        rd = csv.reader(fh)
        hdr = next(rd)
        ev = [h for h in hdr if h.startswith("E_")]
        for row in rd:
            r = dict(zip(hdr, row))
            d = out.setdefault((r["plant"], float(r["f"])),
                               dict(seeds=[], delta0=[], E={h[2:]: [] for h in ev}, P_KLPI=[],
                                    P_dev=[], events=[]))
            d["seeds"].append(int(r["seed"]))
            d["delta0"].append(float(r["delta0"]))
            for h in ev:
                d["E"][h[2:]].append(float(r[h]))
            d["P_KLPI"].append(float(r["P_KLPI"]))
            d["P_dev"].append(float(r["P_dev"]))
            d["events"].append(int(r["skips"]) + int(r["blowups"]) + int(r["unstable"]))
    for d in out.values():
        for k, v in d.items():
            d[k] = {a: np.array(b) for a, b in v.items()} if k == "E" else np.array(v)
        if setting == "random":
            keep = np.flatnonzero(d["events"] == 0)[:N_SEEDS_RANDOM]
            for k, v in d.items():
                d[k] = {a: b[keep] for a, b in v.items()} if k == "E" else v[keep]
    with open(os.path.join(outdir, "meta.json")) as fh:
        out["_T"] = json.load(fh)[setting]["T"]
    return out


def compare(setting: str, d: dict, plant: Plant, f: float, a: str, b: str, T: int) -> dict:
    """Ratio r (18) of variant ``a`` to variant ``b`` in one cell, with its 95% interval."""
    cell = Cell(setting, plant, f, list(d["seeds"]), (),
                "poor" if setting == "poor" else "standard", T)
    return ratio(d["E"][a], d["E"][b], rng_for(bootstrap_label(cell, a, b)))



# ============================================================================ figure tables
def write_tables(outdir: str = RESULTS) -> list[str]:
    """Write the tables of Figs. 1 and 2 into ``outdir/tables`` from the per-run files."""
    td = os.path.join(outdir, "tables")
    os.makedirs(td, exist_ok=True)
    written = []

    def save(name: str, text: str) -> None:
        with open(os.path.join(td, name), "w") as fh:
            fh.write(text)
        written.append(name)

    pp = paper_plants()
    # Fig. 1(a): r of KLPI to PGAC-iso (and to PGAC-shaped) against the budget, standard start
    std = read_setting("standard", outdir)
    T = std["_T"]
    for p, pl in pp.items():
        rows = ["f c r lo hi rs rslo rshi"]
        for f in FRACTIONS:
            d = std[(p, f)]
            k = compare("standard", d, pl, f, "KLPI", "PGAC-iso", T)
            s = compare("standard", d, pl, f, "KLPI", "PGAC-shaped", T)
            rows.append(f"{f:g} {budget(pl, f):.6g} {k['r']:.5f} {k['lo']:.5f} {k['hi']:.5f} "
                        f"{s['r']:.5f} {s['lo']:.5f} {s['hi']:.5f}")
        save(f"g1_wide_{p}.dat", "\n".join(rows) + "\n")
    # Fig. 1(c): windowed ratio of KLPI to PGAC-iso at c = C*
    with np.load(os.path.join(outdir, "windows_standard.npz")) as z:
        win = {k: z[k] for k in z.files}
    for p in pp:
        rows = windowed_ratio(*win[f"{p}_f1"], win["k_edges"])
        save(f"g1_time_{p}_f1.dat", "k r lo hi rc\n" + "".join(
            f"{q['k']:.4g} {q['r']:.5f} {q['lo']:.5f} {q['hi']:.5f} {q['rc']:.5f}\n"
            for q in rows))
    # Fig. 1(b): random plants against their input spread, c = C*/2
    rnd = read_setting("random", outdir)
    rows = ["id spread r lo hi rs rslo rshi Cs"]
    for i, (nm, pl) in enumerate(random_plants().items(), 1):
        d = rnd[(nm, RANDOM_FRACTION)]
        sp = sim_plant(pl)
        k = compare("random", d, pl, RANDOM_FRACTION, "KLPI", "PGAC-iso", rnd["_T"])
        s = compare("random", d, pl, RANDOM_FRACTION, "KLPI", "PGAC-shaped", rnd["_T"])
        rows.append(f"{i} {sp.spread():.6g} {k['r']:.5f} {k['lo']:.5f} {k['hi']:.5f} "
                    f"{s['r']:.5f} {s['lo']:.5f} {s['hi']:.5f} {sp.C_star:.6g}")
    save("g1_random.dat", "\n".join(rows) + "\n")
    # Fig. 2(a): ratios within the pairs that differ only in the averaging, poor-gain start
    poor = read_setting("poor", outdir)
    pairs = (("KLPI-iso", "PGAC-iso"), ("KLPI", "PGAC-shaped"), ("KLPI", "PGAC-iso"),
             ("PGAC-shaped", "PGAC-iso"))
    rows = ["x iso ep em klsh kp km kpg kpgp kpgm shpg shp shm"]
    for x, (p, pl) in enumerate(pp.items(), 1):
        cmp = [compare("poor", poor[(p, 1.0)], pl, 1.0, a, b, poor["_T"]) for a, b in pairs]
        rows.append(f"{x} " + " ".join(f"{d['r']:.6g} {d['hi'] - d['r']:.6g} "
                                       f"{d['r'] - d['lo']:.6g}" for d in cmp))
    save("g1_poor.dat", "\n".join(rows) + "\n")
    # Fig. 2(b): median relative gap of KLPI-iso over that of PGAC-iso, averaged over
    # logarithmic windows; formed from the medians printed to six digits
    with np.load(os.path.join(outdir, "gaps_poor.npz")) as z:
        g = {k: z[k] for k in z.files}
    k = g["k"] + 1
    for p in pp:   # rows of g[p]: KLPI, PGAC-iso, PGAC-shaped, KLPI-iso
        med = [np.array([float(f"{v:.6g}") for v in np.quantile(g[p][i].astype(float), 0.5,
                                                                axis=0)]) for i in (3, 1)]
        rows = log_window_ratio(k, med[0], med[1])
        save(f"g1_ratio_poor_{p}.dat", "k r\n" + "".join(f"{a:.5g} {b:.5f}\n" for a, b in rows))
    return written


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Rerun the simulations of Section VI of [1].")
    ap.add_argument("--quick", action="store_true",
                    help="24 seeds per plant (6 per random plant) instead of 480 (60)")
    ap.add_argument("--only", nargs="+", choices=("standard", "poor", "random"),
                    default=["standard", "poor", "random"])
    ap.add_argument("--workers", type=int, default=None, help="processes (default: all cores)")
    ap.add_argument("--horizon", type=int, default=T_RUN, help="online samples T per run")
    a = ap.parse_args(argv)
    out = os.path.join(RESULTS, "quick") if a.quick or a.horizon != T_RUN else RESULTS
    for st in a.only:
        ns = (6 if st == "random" else 24) if a.quick else None
        write_setting(st, run_setting(st, a.workers, n_seeds=ns, T=a.horizon), out)
    if set(a.only) == {"standard", "poor", "random"}:
        write_tables(out)
    print(f"wrote the results to {out}")


if __name__ == "__main__":
    main()
