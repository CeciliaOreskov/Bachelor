
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


 
class Gaussian_sampler:
    """
    Gaussian variational distribution with learnable mean and covariance.
 
    covariance_type='diag': mean-field. Stores mu and rho, sigma = softplus(rho).
    covariance_type='full': full covariance. Stores mu, the diagonal of A
        as rho_diag (with diag = softplus(rho_diag) > 0), and the strict
        upper triangle of A as A_off. Assembles A on the fly so Sigma = A^T A.

    """
    def __init__(self, dim, n_samples=100, init_mu=None, init_sigma=0.1,
                 covariance_type="diag",
                 dtype=torch.float64, device=None):
        if covariance_type not in ("diag", "full"):
            raise ValueError(f"covariance_type must be 'diag' or 'full', "
                             f"got {covariance_type!r}")
 
        self.n_samples       = n_samples
        self.dim             = dim
        self.dtype           = dtype
        self.device          = device
        self.covariance_type = covariance_type
 
 
        if init_mu is None:
            mu0 = torch.zeros(dim, dtype=dtype, device=device)
        else:
            mu0 = init_mu.detach().clone().to(dtype=dtype, device=device)
        self.mu = torch.nn.Parameter(mu0)
 
    
        # Inverse softplus so softplus(rho_init) = init_sigma
        rho0_val = float(np.log(np.expm1(init_sigma)))
 
        if covariance_type == "diag":
            rho0 = torch.full((dim,), rho0_val, dtype=dtype, device=device)
            self.rho = torch.nn.Parameter(rho0)
       
            self.rho_diag = None
            self.A_off    = None
 
        else:  # 'full'
          
            rho_diag0 = torch.full((dim,), rho0_val, dtype=dtype, device=device)
            A_off0    = torch.zeros((dim, dim), dtype=dtype, device=device)
 
            self.rho_diag = torch.nn.Parameter(rho_diag0)
            self.A_off    = torch.nn.Parameter(A_off0)

            self._upper_mask = torch.triu(
                torch.ones(dim, dim, dtype=dtype, device=device), diagonal=1
            )
            self.rho = None  
 
 
    def parameters(self):
        if self.covariance_type == "diag":
            return [self.mu, self.rho]
        else:
            return [self.mu, self.rho_diag, self.A_off]
 
    def sigma(self):
        """
        Per-dimension marginal standard deviation: sqrt(diag(Sigma)).
 
        For 'diag': sigma_i = softplus(rho_i).
        For 'full': sigma_i = sqrt((A^T A)_ii) = ||A[:, i]||.
        """
        if self.covariance_type == "diag":
            return torch.nn.functional.softplus(self.rho)
        else:
            A = self._build_A()
    
            return torch.sqrt((A * A).sum(dim=0) + 1e-30)
 
    def _build_A(self):
        """Assemble upper-triangular A with positive diagonal (full mode only)."""
        diag = torch.nn.functional.softplus(self.rho_diag)        
        A = self.A_off * self._upper_mask + torch.diag(diag)      
        return A
 
    
    def sample_q_endpoints(self, mu=None, detach_from_mu=False, drift=False,
                           random_state=None):
        """
        Returns (n_samples, dim) Gaussian samples via reparameterization.
 
        The `mu` argument is ignored (kept for interface compatibility);
        this sampler uses its own internal parameters.
        """
        if random_state is not None:
            torch.manual_seed(random_state)
 
        eps = torch.randn(self.n_samples, self.dim,
                          dtype=self.mu.dtype, device=self.mu.device)
 
        if self.covariance_type == "diag":
            sigma = self.sigma()                                
            if detach_from_mu:
                mu_use, sigma_use = self.mu.detach(), sigma.detach()
            else:
                mu_use, sigma_use = self.mu, sigma
            theta = mu_use.unsqueeze(0) + sigma_use.unsqueeze(0) * eps
 
        else:  # 'full'
            A = self._build_A()                               
            if detach_from_mu:
                mu_use, A_use = self.mu.detach(), A.detach()
            else:
                mu_use, A_use = self.mu, A
         
            theta = mu_use.unsqueeze(0) + eps @ A_use
 
        return theta.detach() if detach_from_mu else theta
 
    def sample_prior(self, dtype, device, dim):
        return torch.randn(self.n_samples, dim, dtype=dtype, device=device)
 

    def kl_to_prior(self, sigma_p=1.0):
        """
        Closed-form KL( q || N(0, sigma_p^2 I) ).
 
        For Gaussian q = N(mu, Sigma):
            KL = 0.5 * [ tr(Sigma)/sp^2 + ||mu||^2/sp^2
                         - dim - log det Sigma + dim * log sp^2 ]
        """
        mu = self.mu
        sp2 = float(sigma_p) ** 2
        log_sp2 = float(np.log(sp2))
 
        if self.covariance_type == "diag":
            sigma = self.sigma()
            tr_Sigma   = (sigma ** 2).sum()
            log_det    = 2.0 * torch.log(sigma).sum()
            
        else:  # 'full'
            A = self._build_A()
          
            tr_Sigma = (A * A).sum()
          
            diag_A   = torch.nn.functional.softplus(self.rho_diag)
            log_det  = 2.0 * torch.log(diag_A).sum()
 
        kl = 0.5 * (
            tr_Sigma / sp2
            + (mu * mu).sum() / sp2
            - self.dim
            - log_det
            + self.dim * log_sp2
        )
        return kl
 
 
def VI_Gaussian(mu_start, loss, n_samples, n_opt_steps, lr_VI=1e-2,
                beta=1.0, sigma_p=1.0, init_sigma=0.1,
                covariance_type="diag", seed=None, lr_mu=None, lr_rho=None, lr_Aoff=None):
    """
 
    covariance_type='diag': mean-field q = N(mu, diag(sigma^2)).
    covariance_type='full': full covariance q = N(mu, A^T A), A upper-triangular.

    """
    if seed is not None:
        torch.manual_seed(seed)
 
    mu_start = mu_start.to(dtype=torch.float64)
    dim = mu_start.shape[-1]
 
    sampler = Gaussian_sampler(
        dim=dim,
        n_samples=n_samples,
        init_mu=mu_start,
        init_sigma=init_sigma,
        covariance_type=covariance_type,
        dtype=torch.float64,
    )
 

    _lr_mu   = lr_mu   if lr_mu   is not None else lr_VI
    _lr_rho  = lr_rho  if lr_rho  is not None else lr_VI
    _lr_Aoff = lr_Aoff if lr_Aoff is not None else lr_VI

    if covariance_type == "diag":
        opt = torch.optim.Adam([
            {"params": [sampler.mu],  "lr": _lr_mu},
            {"params": [sampler.rho], "lr": _lr_rho},
        ])
    else:  # 'full'
        opt = torch.optim.Adam([
            {"params": [sampler.mu],       "lr": _lr_mu},
            {"params": [sampler.rho_diag], "lr": _lr_rho},
            {"params": [sampler.A_off],    "lr": _lr_Aoff},
        ])

 
    mu_history    = [sampler.mu.detach().cpu().numpy().copy()]
    sigma_history = [sampler.sigma().detach().cpu().numpy().copy()]
    obj_history   = []
    exp_history   = []
    kl_history    = []
 
    for k in range(n_opt_steps):
        theta_samples = sampler.sample_q_endpoints(detach_from_mu=False)
 
        exp_term = loss.Loss_batch(theta_samples).mean()
        kl_term  = sampler.kl_to_prior(sigma_p=sigma_p)
        objective = exp_term + beta * kl_term
 
        opt.zero_grad()
        objective.backward()
        opt.step()
 
        obj_history.append(objective.item())
        exp_history.append(exp_term.item())
        kl_history.append(kl_term.item())
        mu_history.append(sampler.mu.detach().cpu().numpy().copy())
        sigma_history.append(sampler.sigma().detach().cpu().numpy().copy())
 
        if (k + 1) % 10 == 0:
            print(
                f"iter {k+1:4d} | obj={objective.item():.6f} "
                f"| E[L]={exp_term.item():.6f} | KL={kl_term.item():.6f} "
                f"| ||mu||={sampler.mu.detach().norm().item():.3f} "
                f"| mean(sigma)={sampler.sigma().detach().mean().item():.4f} "
                f"| cov={covariance_type}"
            )
 
    return (
        sampler,
        np.array(mu_history),
        np.array(sigma_history),
        np.array(obj_history),
        np.array(exp_history),
        np.array(kl_history),
    )
 