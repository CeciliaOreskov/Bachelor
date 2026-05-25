#ALL THE LOVELY FUNCTIONS REWRITTEN IN CLASSES:
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

class NeuralNetworkModel(nn.Module):
    def __init__(self, layer_sizes, activation_type = 'tanh'):
        super().__init__()               
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        self.param_shapes = [((layer_sizes[i+1], layer_sizes[i]), (layer_sizes[i+1],)) for i in range(self.n_layers)]
        
        self.theta_dim = sum(self.layer_sizes[1:])

        for i in range(self.n_layers):
            self.theta_dim += self.layer_sizes[i]*self.layer_sizes[i+1]
        
        if activation_type == "softplus":
            self.activation = nn.Softplus()
        elif activation_type == "relu":
            self.activation = torch.relu
        elif activation_type == "tanh":
            self.activation = torch.tanh
        elif activation_type == "sigmoid":
            self.activation = torch.sigmoid
        elif activation_type == "silu":
            self.activation = F.silu
        else:
            raise ValueError("Unknown activation type")
        
    def unpack_theta(self, theta):
        layers = []
        idx_s, idx_e = 0, 0
        for slice in self.param_shapes:
            idx_s = idx_e
            idx_e += slice[0][0]*slice[0][1]
            W = theta[idx_s:idx_e].reshape(slice[0])
            idx_s = idx_e 
            idx_e += slice[1][0]
            b = theta[idx_s: idx_e]
            layers.append((W,b))
        return layers
    
    def forward_theta(self, theta, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=theta.dtype, device=theta.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        X = x
        layers = self.unpack_theta(theta)
        

        for W, b in layers[:-1]:
            X = torch.matmul(X, W.T) + b
            X = self.activation(X)

        W, b = layers[-1]
        y = torch.matmul(X, W.T) + b         # (N, d_hidden) @ (d_hidden, d_out) -> (N, d_out)

        return y.squeeze(-1)

    def unpack_theta_batch(self, theta_batch): # theta er en D, M stor matrice af vægte:
        M = theta_batch.shape[0]
        layers_batch = []
        idx_s, idx_e = 0, 0
        for slice in self.param_shapes:
            idx_s = idx_e
            idx_e += slice[0][0]*slice[0][1]
            W = theta_batch[:,idx_s:idx_e].reshape(M, slice[0][0], slice[0][1])
            idx_s = idx_e 
            idx_e += slice[1][0]
            b = theta_batch[:, idx_s: idx_e].reshape(M, slice[1][0])
            layers_batch.append((W,b))
        return layers_batch
    
    def forward_theta_batch(self, theta_batch, x):

        M = theta_batch.shape[0]              # number of sampled networks
        layers = self.unpack_theta_batch(theta_batch)

        x = torch.as_tensor(x, dtype=theta_batch.dtype, device=theta_batch.device)
        X = x.unsqueeze(0).expand(M, *x.shape)

        for W, b in layers[:-1]:
            X = torch.matmul(X, W.transpose(-1, -2)) + b.unsqueeze(1)
            X = self.activation(X)

        W, b = layers[-1]
        y = torch.matmul(X, W.transpose(-1, -2)) + b.unsqueeze(1)     # -> (M, N, d_out)

        return y.squeeze(-1)

class Losses_BANAN:
    def __init__(self, eps = 1e-12):
        self.eps = eps

    def Loss(self, theta):
        return (theta[0]*theta[1]-1)**2 + self.eps
    
    def Loss_batch(self, theta_batch):
        return (theta_batch[:, 0] * theta_batch[:, 1] - 1.0) ** 2 + self.eps
    
    def gradient_of_Loss(self, theta):
        t = theta[0] * theta[1] - 1.0
        return 2.0 * t * torch.stack([theta[1], theta[0]])

class Losses_MSE:
    def __init__(self, model, x, y_data, eps = 1e-12):
        self.model = model
        self.eps = eps
        self.x = torch.as_tensor(x, dtype=torch.float64).reshape(-1, 1)
        self.y_data = torch.as_tensor(y_data, dtype=torch.float64).reshape(-1)

    def Loss(self, theta):
        y_hat = self.model.forward_theta(theta,self.x)
        return torch.mean((self.y_data - y_hat)**2) + self.eps
    
    def Loss_batch(self, theta_batch):
        y_hat = self.model.forward_theta_batch(theta_batch, self.x)  # (M, N)
        return torch.mean((y_hat - self.y_data.unsqueeze(0))**2, dim=1) + self.eps  # (M,
    
    def gradient_of_Loss(self, theta):
        loss = self.Loss(theta)
        g = torch.autograd.grad(loss, theta, create_graph=True)[0]
        return g

class Losses_SSE:
    def __init__(self, model, x, y_data, eps = 1e-12):
        self.model = model
        self.eps = eps
        self.x = torch.as_tensor(x, dtype=torch.float64).reshape(-1, 1)
        self.y_data = torch.as_tensor(y_data, dtype=torch.float64).reshape(-1)

    def Loss(self, theta):
        y_hat = self.model.forward_theta(theta,self.x)
        return torch.sum((self.y_data - y_hat)**2) + self.eps
    
    def Loss_batch(self, theta_batch):
        y_hat = self.model.forward_theta_batch(theta_batch, self.x)  # (M, N)
        return torch.sum((y_hat - self.y_data.unsqueeze(0))**2, dim=1) + self.eps
    
    def gradient_of_Loss(self, theta):
        loss = self.Loss(theta)
        g = torch.autograd.grad(loss, theta, create_graph=True)[0]
        return g

class RiemannianMetric:
    def __init__(self, loss_func, eps = 1e-10): #hvis g er mindre end eps, bliver metrikken bare identiteten (euklidisk)
        self.loss_func = loss_func
        self.eps = eps

    def Metric(self, theta):
        g = self.loss_func.gradient_of_Loss(theta)
        I = torch.eye(len(g), dtype=g.dtype, device=g.device)
        sdot = torch.dot(g, g)
        if sdot.detach().item() < self.eps:
            return I
    
        LTL = torch.outer(g,g)
    
        return I + LTL
    
    def M_sqrt_inv_from_grad(self, theta):
        g = self.loss_func.gradient_of_Loss(theta)   
        I = torch.eye(len(g), dtype=g.dtype, device=g.device)
        s = torch.dot(g, g)

        if s.detach().item() < self.eps:
            return I
        
        coeff = 1.0 - 1.0 / torch.sqrt(1.0 + s)
        return I - coeff * torch.outer(g, g) / s
    
    def M_inv_from_grad(self, theta):
        g = self.loss_func.gradient_of_Loss(theta)
        I = torch.eye(len(g), dtype=g.dtype, device=g.device)
        s = torch.dot(g, g)

        if s.detach().item() < self.eps:
            return I

        return I - torch.outer(g, g) / (1.0 + s)

    def M_det_from_grad(self, theta):
        g = self.loss_func.gradient_of_Loss(theta)
        s = torch.dot(g, g)
        return 1.0 + s

    def compute_q(self, theta):
        M_inv = self.M_inv_from_grad(theta)
        M_det = self.M_det_from_grad(theta)
        q = torch.sqrt(M_det) * M_inv
        return q, M_det
    

    
    def compute_drift(self, theta):
        q, M_det = self.compute_q(theta)
        dq = torch.func.jacrev(lambda th: self.compute_q(th)[0])(theta) 
        sum_term = torch.einsum("lkl->k", dq)
        drift = (sum_term / (torch.sqrt(M_det)))
        return drift 
    

    def gradient_of_Loss_batch(self, theta_batch):
        grad_fn = torch.func.grad(self.loss_func.Loss)
        return torch.vmap(grad_fn)(theta_batch)

    def M_sqrt_inv_from_grad_batch(self, theta_batch):
        g = self.gradient_of_Loss_batch(theta_batch)   # (M, D)
        M, D = g.shape

        I = torch.eye(D, dtype=g.dtype, device=g.device).expand(M, D, D)
        s = torch.sum(g * g, dim=1, keepdim=True)      # (M, 1)

        mask = s < self.eps
        safe_s = torch.where(mask, torch.ones_like(s), s)

        coeff = 1.0 - 1.0 / torch.sqrt(1.0 + s)        # (M, 1)
        outer = g.unsqueeze(2) * g.unsqueeze(1)        # (M, D, D)

        out = I - coeff.view(M, 1, 1) * outer / safe_s.view(M, 1, 1)
        return torch.where(mask.view(M, 1, 1), I, out)




class Losses_SSE_Batched:
    """
    Same SSE loss as Losses_SSE, but with an extra method that returns
    S per-batch gradients at theta. The batches are FIXED at construction
    time so the walk is reproducible.
    """
    def __init__(self, model, x, y_data, S=4, eps=1e-12, batch_seed=0):
        self.model = model
        self.eps = eps
        self.S = S
        self.x = torch.as_tensor(x, dtype=torch.float64).reshape(-1, 1)
        self.y_data = torch.as_tensor(y_data, dtype=torch.float64).reshape(-1)

        N = self.x.shape[0]
        if S > N:
            raise ValueError(f"S={S} batches but only N={N} data points.")

        # Fixed random partition of {0,...,N-1} into S nearly-equal batches.
        g = torch.Generator().manual_seed(batch_seed)
        perm = torch.randperm(N, generator=g)
        self.batch_idx = [perm[i::S] for i in range(S)]  

        self.scale = [float(N) / float(len(idx)) for idx in self.batch_idx]

   
    def Loss(self, theta):
        y_hat = self.model.forward_theta(theta, self.x)
        return torch.sum((self.y_data - y_hat)**2) + self.eps

    def Loss_batch(self, theta_batch):
        y_hat = self.model.forward_theta_batch(theta_batch, self.x)
        return torch.sum((y_hat - self.y_data.unsqueeze(0))**2, dim=1) + self.eps

    def gradient_of_Loss(self, theta):
        loss = self.Loss(theta)
        g = torch.autograd.grad(loss, theta, create_graph=True)[0]
        return g


    def per_batch_gradients(self, theta):
        """
        Returns a (S, D) tensor whose b-th row is the unbiased mini-batch
        estimate of the full-data SSE gradient.
        """
        grads = []
        for idx, scale in zip(self.batch_idx, self.scale):
            x_b = self.x[idx]
            y_b = self.y_data[idx]
            y_hat = self.model.forward_theta(theta, x_b)
            
            loss_b = scale * torch.sum((y_b - y_hat)**2)
            g_b = torch.autograd.grad(loss_b, theta, create_graph=False,
                                      retain_graph=False)[0]
            grads.append(g_b)
        return torch.stack(grads, dim=0)   # (S, D)

    def per_batch_gradients_at_walkers(self, theta_batch):
        """
        Batched-over-motion version. theta_batch: (M, D) -> (M, S, D).
        Uses torch.func.grad + vmap so we don't loop over walkers.
        """
        M_walk = theta_batch.shape[0]
        S = self.S

       
        def make_batch_loss(b):
            idx = self.batch_idx[b]
            scale = self.scale[b]
            x_b = self.x[idx]
            y_b = self.y_data[idx]
            def loss_b(theta):
                y_hat = self.model.forward_theta(theta, x_b)
                return scale * torch.sum((y_b - y_hat)**2)
            return loss_b

        out = []
        for b in range(S):
            grad_fn = torch.func.grad(make_batch_loss(b))
            g_b = torch.vmap(grad_fn)(theta_batch)        # (M, D)
            out.append(g_b)
        return torch.stack(out, dim=1)                    # (M, S, D)


class Losses_MSE_Batched:

    def __init__(self, model, x, y_data, S=4, eps=1e-12, batch_seed=0):
        self.model = model
        self.eps = eps
        self.S = S
        self.x = torch.as_tensor(x, dtype=torch.float64).reshape(-1, 1)
        self.y_data = torch.as_tensor(y_data, dtype=torch.float64).reshape(-1)

        N = self.x.shape[0]
        if S > N:
            raise ValueError(f"S={S} batches but only N={N} data points.")

        g = torch.Generator().manual_seed(batch_seed)
        perm = torch.randperm(N, generator=g)
        self.batch_idx = [perm[i::S] for i in range(S)]

    def Loss(self, theta):
        y_hat = self.model.forward_theta(theta, self.x)
        return torch.mean((self.y_data - y_hat) ** 2) + self.eps

    def Loss_batch(self, theta_batch):
        y_hat = self.model.forward_theta_batch(theta_batch, self.x)
        return torch.mean((y_hat - self.y_data.unsqueeze(0)) ** 2, dim=1) + self.eps

    def gradient_of_Loss(self, theta):
        loss = self.Loss(theta)
        g = torch.autograd.grad(loss, theta, create_graph=True)[0]
        return g


    def per_batch_gradients(self, theta):
        grads = []
        for idx in self.batch_idx:
            x_b = self.x[idx]
            y_b = self.y_data[idx]
            y_hat = self.model.forward_theta(theta, x_b)
            loss_b = torch.mean((y_b - y_hat) ** 2)        # unbiased est. of full MSE
            g_b = torch.autograd.grad(loss_b, theta, create_graph=False,
                                      retain_graph=False)[0]
            grads.append(g_b)
        return torch.stack(grads, dim=0)


    def per_batch_gradients_at_walkers(self, theta_batch):
        def make_batch_loss(b):
            idx = self.batch_idx[b]
            x_b = self.x[idx]
            y_b = self.y_data[idx]
            def loss_b(theta):
                y_hat = self.model.forward_theta(theta, x_b)
                return torch.mean((y_b - y_hat) ** 2)
            return loss_b

        out = []
        for b in range(self.S):
            grad_fn = torch.func.grad(make_batch_loss(b))
            g_b = torch.vmap(grad_fn)(theta_batch)
            out.append(g_b)
        return torch.stack(out, dim=1)


class RiemannianMetricBatched:
   
    def __init__(self, loss_func, eps=1e-10, metric_scale=1.0):
        
        if not hasattr(loss_func, "per_batch_gradients_at_walkers"):
            raise TypeError(
                "RiemannianMetricBatched needs a loss with "
                "per_batch_gradients_at_walkers, e.g. Losses_SSE_Batched."
            )
        self.loss_func = loss_func
        self.S = loss_func.S
        self.eps = eps
        self.metric_scale = float(metric_scale)

    def _build_U(self, theta_batch):
        """theta_batch: (M, D)  ->  U: (M, D, S),  K: (M, S, S) where K = U^T U.

        With metric_scale=lambda we set U = sqrt(lambda/S) * G^T so that
        U U^T = (lambda / S) * G^T G  ==>  M = I + U U^T = I + (lambda/S) G^T G.
        All Woodbury formulas below stay unchanged.
        """
        G = self.loss_func.per_batch_gradients_at_walkers(theta_batch)   
        scale = (self.metric_scale / float(self.S)) ** 0.5
        U = G.transpose(1, 2) * scale                                    
        K = torch.matmul(U.transpose(1, 2), U)                           
        
        K = 0.5 * (K + K.transpose(1, 2))
        return U, K

    def M_sqrt_inv_from_grad_batch(self, theta_batch):
        """
        Returns (M, D, D) batch of M^{-1/2}.
        Uses Woodbury: M^{-1/2} = I - U f(K) U^T,
        where f(K) = V diag( (1 - 1/sqrt(1+lambda)) / lambda ) V^T
        and  K = V diag(lambda) V^T.
        """
        U, K = self._build_U(theta_batch)
        M_walk, D, S = U.shape

        # Eigendecompose the small S x S matrix.
        lam, V = torch.linalg.eigh(K)                       

        
        small = lam < self.eps
        safe_lam = torch.where(small, torch.ones_like(lam), lam)
        coef_full = (1.0 - 1.0 / torch.sqrt(1.0 + safe_lam)) / safe_lam
        coef = torch.where(small, torch.full_like(lam, 0.5), coef_full)  #

        fK = torch.matmul(V * coef.unsqueeze(1), V.transpose(1, 2))

        
        I = torch.eye(D, dtype=U.dtype, device=U.device).expand(M_walk, D, D)
        UfK = torch.matmul(U, fK)                           
        correction = torch.matmul(UfK, U.transpose(1, 2))   
        return I - correction

    def M_inv_from_grad_batch(self, theta_batch):
        """M^{-1} = I - U (I_S + K)^{-1} U^T"""
        U, K = self._build_U(theta_batch)
        M_walk, D, S = U.shape
        I_S = torch.eye(S, dtype=K.dtype, device=K.device).expand(M_walk, S, S)
        inner = torch.linalg.solve(I_S + K, U.transpose(1, 2))   # (M, S, D)
        I_D = torch.eye(D, dtype=U.dtype, device=U.device).expand(M_walk, D, D)
        return I_D - torch.matmul(U, inner)

    def M_det_from_grad_batch(self, theta_batch):
        """det M = det(I_S + K)"""
        _, K = self._build_U(theta_batch)
        M_walk, S, _ = K.shape
        I_S = torch.eye(S, dtype=K.dtype, device=K.device).expand(M_walk, S, S)
        return torch.linalg.det(I_S + K)


class Brownian_sampler_batched:
    """
    Same as Brownian_sampler but uses RiemannianMetricBatched. Drop-in:
    exposes the same sample_q_endpoints / sample_prior interface so
    plot_posterior_predictive(...) and VI_Brownian(...) both still work.
    """
    def __init__(self, metric, walk_len=500, step_size=0.01, n_samples=100):
        if not isinstance(metric, RiemannianMetricBatched):
            raise TypeError("Brownian_sampler_batched needs a RiemannianMetricBatched.")
        self.metric = metric
        self.walk_len = walk_len
        self.step_size = step_size
        self.n_samples = n_samples

    def sample_q_endpoints(self, mu, detach_from_mu=False, drift=False,
                           random_state=None):
        if drift:
            raise NotImplementedError("drift not implemented for batched metric yet.")
        if random_state is not None:
            torch.manual_seed(random_state)

        if detach_from_mu:
            theta = (mu.detach().unsqueeze(0)
                       .expand(self.n_samples, -1).clone().requires_grad_(True))
        else:
            theta = mu.unsqueeze(0).expand(self.n_samples, -1).clone()

        dt = torch.tensor(self.step_size, dtype=mu.dtype, device=mu.device)
        sqrt_dt = torch.sqrt(dt)

        for _ in range(self.walk_len):
            Minv_sqrt = self.metric.M_sqrt_inv_from_grad_batch(theta)   
            eps = torch.randn_like(theta)
            theta = theta + sqrt_dt * torch.einsum("mij,mj->mi", Minv_sqrt, eps)

        return theta.detach() if detach_from_mu else theta

    def sample_prior(self, dtype, device, dim):
        return torch.randn(self.n_samples, dim, dtype=dtype, device=device)


class Euclidean_sampler: 
    def __init__(self, walk_len = 500, step_size = 0.01, n_samples = 100): 
        self.walk_len = walk_len
        self.step_size = step_size
        self.n_samples = n_samples
        
    
    def compute_EU_walk(self, start_point, random_state=None):
        if random_state is not None:
            torch.manual_seed(random_state)

        theta = start_point.detach().clone().requires_grad_(True)
        path = [theta.clone()]
        dt = torch.tensor(self.step_size, dtype = theta.dtype, device = theta.device)
        sqrt_dt = torch.sqrt(dt)

        for _ in range(self.walk_len):
            eps = torch.randn(theta.shape[-1], dtype=theta.dtype, device=theta.device )       
            theta = theta + sqrt_dt * eps
            path.append(theta.detach().clone())
        return torch.stack(path)
    

    def sample_q_endpoints(self, mu, detach_from_mu=False, drift=False, random_state=None):
        if random_state is not None:
            torch.manual_seed(random_state)

        if detach_from_mu:
            theta = (
                mu.detach()
                .unsqueeze(0)
                .expand(self.n_samples, -1)
                .clone()
                .requires_grad_(True)
            )
        else:
            theta = mu.unsqueeze(0).expand(self.n_samples, -1).clone()

        dt = torch.tensor(self.step_size, dtype=mu.dtype, device=mu.device)
        sqrt_dt = torch.sqrt(dt)

        for _ in range(self.walk_len):
            eps = torch.randn_like(theta)
            theta = theta + sqrt_dt * eps

        return theta.detach() if detach_from_mu else theta
    
    def sample_prior(self, dtype, device, dim):
        return torch.randn(self.n_samples, dim, dtype = dtype, device = device) 

   

class Brownian_sampler:
    def __init__(self, metric, walk_len = 500, step_size = 0.01, n_samples = 100):
        self.metric = metric
        self.walk_len = walk_len
        self.step_size = step_size
        self.n_samples = n_samples
        

    def compute_BM(self, start_point, random_state = None, drift = False):
        if random_state is not None:
            torch.manual_seed(random_state)
    

        theta = start_point.detach().clone().requires_grad_(True)
        path = [theta.clone()]
        losses = []
        dt = torch.tensor(self.step_size, dtype = theta.dtype, device = theta.device)
        sqrt_dt = torch.sqrt(dt)

        for _ in range(self.walk_len):
            Minv_sqrt = self.metric.M_sqrt_inv_from_grad(theta)
            eps = torch.randn(theta.shape[-1], dtype=theta.dtype, device=theta.device )       

            if drift:
                drift_term = self.metric.compute_drift(theta)
                theta = theta + 0.5 * dt * drift_term + sqrt_dt * (Minv_sqrt @ eps)
            else: 
                theta = theta + sqrt_dt * (Minv_sqrt @ eps)
        
            path.append(theta.detach().clone())
            losses.append(self.metric.loss_func.Loss(theta))
        return torch.stack(path)
    

    def sample_q_endpoints(self, mu, detach_from_mu=False, drift=False, random_state=None):
        if random_state is not None:
            torch.manual_seed(random_state)

        if detach_from_mu:
            theta = (
                mu.detach()
                .unsqueeze(0)
                .expand(self.n_samples, -1)
                .clone()
                .requires_grad_(True)
            )
        else:
            theta = mu.unsqueeze(0).expand(self.n_samples, -1).clone()

        dt = torch.tensor(self.step_size, dtype=mu.dtype, device=mu.device)
        sqrt_dt = torch.sqrt(dt)

        if drift:
            raise NotImplementedError(
                "Batch version below supports drift=False first. "
                "Keep drift=False until the rest is fast."
            )

        for _ in range(self.walk_len):
            Minv_sqrt = self.metric.M_sqrt_inv_from_grad_batch(theta)   # (M, D, D)
            eps = torch.randn_like(theta)                               # (M, D)
            theta = theta + sqrt_dt * torch.einsum("mij,mj->mi", Minv_sqrt, eps)

        return theta.detach() if detach_from_mu else theta
    
  
   
    
    def sample_prior(self, dtype, device, dim):
        return torch.randn(self.n_samples, dim, dtype = dtype, device = device) 

class LangevinSampler:
    """
    Riemannian Langevin sampler for target
        π(θ) ∝ exp(−L(θ))

    Uses metric supplied via RiemannianMetric.
    """

    def __init__(
        self,
        metric,
        step_size=1e-2,
        walk_len=100,
        n_samples=100,
        use_drift_correction=True,
    ):
        self.metric = metric
        self.step_size = step_size
        self.walk_len = walk_len
        self.n_samples = n_samples
        self.use_drift_correction = use_drift_correction

    def step(self, theta):
        """
        Single Langevin step
        """
        dt = torch.tensor(self.step_size, dtype=theta.dtype, device=theta.device)
        sqrt_2dt = torch.sqrt(2.0 * dt)

        # Gradient of loss
        g = self.metric.loss_func.gradient_of_Loss(theta)

        # Metric terms
        M_inv = self.metric.M_inv_from_grad(theta)
        M_inv_sqrt = self.metric.M_sqrt_inv_from_grad(theta)

        # Langevin drift
        drift = - M_inv @ g

        # Riemannian correction term Γ(θ)
        if self.use_drift_correction:
            drift = drift + self.metric.compute_drift(theta)

        noise = torch.randn_like(theta)

        theta_new = (
            theta
            + dt * drift
            + sqrt_2dt * (M_inv_sqrt @ noise)
        )

        return theta_new

    def sample_path(self, start_point, random_state=None):
        """
        Returns full Langevin path: (walk_len+1, dim)
        """
        if random_state is not None:
            torch.manual_seed(random_state)

        theta = start_point.detach().clone().requires_grad_(True)
        path = [theta.detach().clone()]

        for _ in range(self.walk_len):
            theta = self.step(theta)
            theta = theta.detach().clone().requires_grad_(True)
            path.append(theta.detach().clone())

        return torch.stack(path)

    def sample_endpoints(self, mu, detach_from_mu=False, random_state=None):
        """
        Used inside VI loop.
        Returns n_samples endpoints of Langevin walks.
        """
        if random_state is not None:
            torch.manual_seed(random_state)

        samples = []

        for _ in range(self.n_samples):
            if detach_from_mu:
                theta = mu.detach().clone().requires_grad_(True)
            else:
                theta = mu

            for _ in range(self.walk_len):
                theta = self.step(theta)

            samples.append(theta.detach() if detach_from_mu else theta)

        return torch.stack(samples, dim=0)

    def sample_prior(self, dtype, device, dim):
        """
        Standard Gaussian prior
        """
        return torch.randn(self.n_samples, dim, dtype=dtype, device=device)
    
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

    
def train_ratio_net(ratio_net, X_q, X_p, n_steps=100, lr=1e-3):
    opt = torch.optim.Adam(ratio_net.parameters(), lr=lr)

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
#N_q and N_p are the number of samples from posterior and prior in each step
#VAN is for the KL divergence based on a classifier trying to distinguish samples from prior and posterior
#MMD: the MMD distance between posterior and prior samples
#Everything is casted to float 64 for better precosion



def VI_Brownian(mu_start, loss, sampler, n_opt_steps = 100, lr_VI = 1e-1, beta = 1.0, sigma = 1.0, KL = 'VAN', seed = None, drift = False):

    if seed is not None: 
        torch.manual_seed(seed)
    
    mu = torch.nn.Parameter(mu_start.to(dtype=torch.float64))

    opt_mu = torch.optim.Adam([mu], lr=lr_VI)
    dim = mu.shape[-1]

    mu_history = [mu.detach().cpu().numpy().copy()]
    obj_history = []
    exp_history = []
    kl_history = []

    if KL == 'VAN':
        ratio_net = RatioNet(dim).to(dtype=torch.float64)

        for k in range(n_opt_steps): 
            #Train discriminator on detached samples
            X_q_det = sampler.sample_q_endpoints(mu, detach_from_mu=True, drift = drift)
            X_p_det = sampler.sample_prior(mu.dtype, mu.device, dim)
            train_ratio_net(ratio_net, X_q_det, X_p_det)

              # DIAGNOSTIC
            with torch.no_grad():
                logits_q = ratio_net(X_q_det)
                logits_p = ratio_net(X_p_det)
                acc = ((logits_q > 0).double().mean()
                       + (logits_p < 0).double().mean()) / 2
                print(
                    f"  [diag] acc={acc.item():.3f} | "
                    f"logits_q={logits_q.mean().item():+.2f}±{logits_q.std().item():.2f} | "
                    f"logits_p={logits_p.mean().item():+.2f}±{logits_p.std().item():.2f} | "
                    f"||mu||={mu.detach().norm().item():.2f}"
                )
        # 

            #Freeze discriminator and update mu
            for p in ratio_net.parameters():
                p.requires_grad_(False)
            
            X_q = sampler.sample_q_endpoints(mu, detach_from_mu=False, drift = drift)

            kl_term = ratio_net(X_q).mean()
            exp_term = loss.Loss_batch(X_q).mean()

            objective = exp_term + beta * kl_term   # minimize negative ELBO

            opt_mu.zero_grad()
            objective.backward()
            opt_mu.step()

            for p in ratio_net.parameters():
                p.requires_grad_(True)
            
            obj_history.append(objective.item())
            exp_history.append(exp_term.item())
            kl_history.append(kl_term.item())
            mu_history.append(mu.detach().cpu().numpy().copy())


            if (k + 1) % 10 == 0:
                    print(
                    f"iter {k+1:4d} | obj={objective.item():.6f} "
                    f"| E[L]={exp_term.item():.6f} | KL={kl_term.item():.6f} "
                    f"| mu={mu.detach().cpu().numpy()}")

    if KL == 'MMD':
            for k in range(n_opt_steps):
                X_q = sampler.sample_q_endpoints(mu, detach_from_mu=False, drift = drift)
                X_p = sampler.sample_prior( mu.dtype, mu.device, dim)
                kl_term = MMD_loss_unbiased(X_q, X_p, sigma)
        
                exp_term = loss.Loss_batch(X_q).mean()
                objective = exp_term + beta * kl_term   # minimize negative ELBO

                opt_mu.zero_grad()
                objective.backward()
                opt_mu.step()

                obj_history.append(objective.item())
                exp_history.append(exp_term.item())
                kl_history.append(kl_term.item())
                mu_history.append(mu.detach().cpu().numpy().copy())
                if (k + 1) % 10 == 0:
                    print(
                    f"iter {k+1:4d} | obj={objective.item():.6f} "
                    f"| E[L]={exp_term.item():.6f} | KL={kl_term.item():.6f} "
                    f"| mu={mu.detach().cpu().numpy()}")
    
   

    return (
        mu.detach().cpu(), 
        np.array(mu_history),
        np.array(obj_history),
        np.array(exp_history),
        np.array(kl_history))


 
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
            # placeholders so attribute access never breaks
            self.rho_diag = None
            self.A_off    = None
 
        else:  # 'full'
            
            # Initialize A = init_sigma * I so Sigma = init_sigma^2 * I,
            
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
            # Sigma_ii = sum_k A[k,i]^2 = ||A[:,i]||^2
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
 
    # ---------- KL ----------
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
            # tr(Sigma) = tr(A^T A) = ||A||_F^2
            tr_Sigma = (A * A).sum()
            # log det Sigma = 2 * sum log diag(A)
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
 
    # Resolve per-group learning rates, falling back to lr_VI
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
 