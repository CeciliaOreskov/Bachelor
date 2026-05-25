#Things for KL proxies 
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import torch
import pandas as pd
from scipy.linalg import lu_factor, lu_solve
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

## DENSITY RATIO ESTIMATOR: 
# Classifier for VAN:
class RatioNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

    
def train_ratio_net(ratio_net, X_q, X_p, n_steps=100, lr=1e-3, opt_type_R = torch.optim.Adam):
    opt = opt_type_R(ratio_net.parameters(), lr=lr)

    y_q = torch.ones(X_q.shape[0], dtype=X_q.dtype, device=X_q.device)
    y_p = torch.zeros(X_p.shape[0], dtype=X_q.dtype, device=X_p.device)

    X = torch.cat([X_q, X_p], dim=0)
    y = torch.cat([y_q, y_p], dim=0)

    for _ in range(n_steps):
        logits = ratio_net(X)
        loss = F.binary_cross_entropy_with_logits(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()


# For MMD: 
def RBF_kernel(X, Y, sigma=1.0):
    pairwise_dist = torch.cdist(X, Y, p=2.0)**2
    K = torch.exp(-pairwise_dist / (2*sigma**2))
    return K

def MMD_loss_biased(X, Y, param=1.0):
    Kxx = RBF_kernel(X, X, param)
    Kyy = RBF_kernel(Y, Y, param)
    Kxy = RBF_kernel(X, Y, param)
    mmd2 = Kxx.mean() + Kyy.mean() - 2*Kxy.mean()
    return mmd2

def MMD_loss_unbiased(X, Y, param=1.0): # unbiased
    Kxx = RBF_kernel(X, X, param)
    Kyy = RBF_kernel(Y, Y, param)
    Kxy = RBF_kernel(X, Y, param)
    n = X.size(0)
    m = Y.size(0)
    Kxx_sum = torch.sum(Kxx) - torch.trace(Kxx)
    Kyy_sum = torch.sum(Kyy) - torch.trace(Kyy)
    Kxy_mean = Kxy.mean()
    mmd2 = (Kxx_sum / (n * (n - 1))) + (Kyy_sum / (m * (m - 1))) - 2 * Kxy_mean

    return mmd2

def MMD_fixed(X, Y, sigma): #unbiased
    Kxx = RBF_kernel(X, X, sigma)
    Kyy = RBF_kernel(Y, Y, sigma)
    Kxy = RBF_kernel(X, Y, sigma)
    n, m = X.shape[0], Y.shape[0]
    # zero the diagonals and average over off-diagonal entries
    sum_xx = (Kxx.sum() - Kxx.diag().sum()) / (n * (n - 1))
    sum_yy = (Kyy.sum() - Kyy.diag().sum()) / (m * (m - 1))
    sum_xy = Kxy.mean()
    return sum_xx + sum_yy - 2.0 * sum_xy


def median_bandwidth(X, Y=None):
    """Median of pairwise Euclidean distances (used for the median heuristic)."""
    Z = X if Y is None else torch.cat([X, Y], dim=0)
    with torch.no_grad():
        D = torch.cdist(Z, Z, p=2.0)
        n = D.shape[0]
        # upper triangle, exclude zero diagonal
        mask = torch.triu(torch.ones_like(D, dtype=torch.bool), diagonal=1)
        med = D[mask].median()
    return med.item()


def MMD_median(X, Y):
    """MMD^2 with bandwidth set to the median of pairwise distances at the
    *current* sample cloud. Bandwidth is detached so it does not leak gradient."""
    sigma = median_bandwidth(X, Y)
    sigma = max(sigma, 1e-12)  # safety
    return MMD_fixed(X, Y, sigma)


def MMD_multiscale(X, Y, sigmas):
    """Sum of MMD^2 across a list of fixed bandwidths."""
    total = 0.0
    for s in sigmas:
        total = total + MMD_fixed(X, Y, s)
    return total / len(sigmas)


def MMD_multiscale_median(X, Y, scales=(0.25, 0.5, 1.0, 2.0, 4.0)):
    """Multi-scale MMD^2 where bandwidths are `scale * median`. Self-tuning;
    no manual sigma needed."""
    med = max(median_bandwidth(X, Y), 1e-12)
    sigmas = [s * med for s in scales]
    return MMD_multiscale(X, Y, sigmas)


def VI_Brownian_MMD(
    mu_start,
    loss,
    sampler,
    mmd_fn,                    # callable taking (X_q, X_p) -> scalar tensor
    mmd_label="MMD",           # for printing only
    n_opt_steps=100,
    lr_VI=1e-2,
    beta=1.0,
    optimizer="adam",          # 'adam' or 'sgd'
    sgd_momentum=0.0,          # used only if optimizer='sgd'
    seed=None,
    drift=False,
    verbose_every=20,
):
    """
    Brownian VI with a configurable MMD-based KL surrogate and configurable optimizer.
    Mirrors the original VI_Brownian's MMD branch but lets you swap the MMD form.
    """
    if seed is not None:
        torch.manual_seed(seed)

    mu = torch.nn.Parameter(mu_start.detach().clone().to(dtype=torch.float64))

    if optimizer.lower() == "adam":
        opt_mu = torch.optim.Adam([mu], lr=lr_VI)
    elif optimizer.lower() == "sgd":
        opt_mu = torch.optim.SGD([mu], lr=lr_VI, momentum=sgd_momentum)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    dim = mu.shape[-1]
    mu_history = [mu.detach().cpu().numpy().copy()]
    obj_history, exp_history, kl_history = [], [], []

    for k in range(n_opt_steps):
        X_q = sampler.sample_q_endpoints(mu, detach_from_mu=False, drift=drift)
        X_p = sampler.sample_prior(mu.dtype, mu.device, dim)

        if k == 0:
            # Diagnostic: gradient norms separately
            mmd_only = mmd_fn(X_q, X_p)
            g_mmd_only = torch.autograd.grad(mmd_only, mu, retain_graph=True)[0]
            eL_only = loss.Loss_batch(X_q).mean()
            g_eL_only = torch.autograd.grad(eL_only, mu, retain_graph=True)[0]
            print(f"  [diag] ||∇MMD|| = {g_mmd_only.norm().item():.3e}, "
                f"||∇E[L]|| = {g_eL_only.norm().item():.3e}, "
                f"ratio = {g_eL_only.norm().item() / max(g_mmd_only.norm().item(), 1e-30):.2e}")
        

        kl_term  = mmd_fn(X_q, X_p)
        exp_term = loss.Loss_batch(X_q).mean()
        objective = exp_term + beta * kl_term

        opt_mu.zero_grad()
        objective.backward()
        opt_mu.step()

        obj_history.append(objective.item())
        exp_history.append(exp_term.item())
        kl_history.append(kl_term.item())
        mu_history.append(mu.detach().cpu().numpy().copy())

        if verbose_every and (k + 1) % verbose_every == 0:
            print(
                f"  [{mmd_label} | {optimizer}] iter {k+1:4d} | "
                f"obj={objective.item():.4f} | "
                f"E[L]={exp_term.item():.4f} | "
                f"MMD={kl_term.item():.4e} | "
                f"||mu||={mu.detach().norm().item():.3f}"
            )

    return (
        mu.detach().cpu(),
        np.array(mu_history),
        np.array(obj_history),
        np.array(exp_history),
        np.array(kl_history),
    )