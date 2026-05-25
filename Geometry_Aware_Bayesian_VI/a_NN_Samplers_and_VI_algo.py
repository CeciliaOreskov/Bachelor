#ALL THE LOVELY FUNCTIONS REWRITTEN IN CLASSES:
from b_KL_functions import *
from c_MAP import *
from d_plot_functions import *

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
        y = torch.matmul(X, W.T) + b        # (N, d_hidden) @ (d_hidden, d_out) -> (N, d_out)

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

        self.x = torch.as_tensor(x, dtype=torch.float64)
        if self.x.dim() == 1:
            self.x = self.x.reshape(-1, 1)

        self.y_data = torch.as_tensor(y_data, dtype=torch.float64).reshape(-1)

        # self.x = torch.as_tensor(x, dtype=torch.float64).reshape(-1, 1)
        # self.y_data = torch.as_tensor(y_data, dtype=torch.float64).reshape(-1)

    def Loss(self, theta):
        y_hat = self.model.forward_theta(theta,self.x)
        return torch.mean((self.y_data - y_hat)**2) + self.eps
    
    def Loss_batch(self, theta_batch):
        y_hat = self.model.forward_theta_batch(theta_batch, self.x)
        return torch.mean((y_hat - self.y_data.unsqueeze(0))**2, dim=1)  + self.eps
    
    def gradient_of_Loss(self, theta):
        loss = self.Loss(theta)
        g = torch.autograd.grad(loss, theta, create_graph=True)[0]
        return g

class Losses_SSE:
    def __init__(self, model, x, y_data, eps = 1e-12):
        self.model = model
        self.eps = eps

        self.x = torch.as_tensor(x, dtype=torch.float64)
        if self.x.dim() == 1:
            self.x = self.x.reshape(-1, 1)

        self.y_data = torch.as_tensor(y_data, dtype=torch.float64).reshape(-1)

        # self.x = torch.as_tensor(x, dtype=torch.float64).reshape(-1, 1)
        # self.y_data = torch.as_tensor(y_data, dtype=torch.float64).reshape(-1)

    def Loss(self, theta):
        y_hat = self.model.forward_theta(theta,self.x)
        return torch.sum((self.y_data - y_hat)**2) + self.eps
    
    def Loss_batch(self, theta_batch):
        y_hat = self.model.forward_theta_batch(theta_batch, self.x)
        return torch.sum((y_hat - self.y_data.unsqueeze(0))**2, dim=1)  + self.eps
    
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

        # if s.detach().item() < self.eps:
        #     return I
        # 
        # coeff = 1.0 - 1.0 / torch.sqrt(1.0 + s)
        coeff = 1.0 / (1.0 + s + torch.sqrt(1.0 + s))

        return I - coeff * torch.outer(g, g) #/s
    
    def M_inv_from_grad(self, theta):
        g = self.loss_func.gradient_of_Loss(theta)
        I = torch.eye(len(g), dtype=g.dtype, device=g.device)
        s = torch.dot(g, g)

        # if s.detach().item() < self.eps:
        #     return I

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
    
    #def compute_q(self, theta):
    #    g = self.loss_func.gradient_of_Loss(theta)
    #    I = torch.eye(len(g), dtype=g.dtype, device=g.device)
    #    s = torch.dot(g, g)
    #
    #    if s.detach().item() < self.eps:
    #        return I, torch.tensor(1.0, dtype=g.dtype, device=g.device)
    #
    #    root = torch.sqrt(1.0 + s)
    #    q = root * I - torch.outer(g, g) / root
    #    M_det = 1.0 + s
    #    return q, M_det
    
    def compute_drift(self, theta):
        q, M_det = self.compute_q(theta)
        dq = torch.func.jacrev(lambda th: self.compute_q(th)[0])(theta) 
        sum_term = torch.einsum("lkl->k", dq)
        drift = (sum_term / (torch.sqrt(M_det)))
        return drift 
    
    #ADDED FOR SPEED
    def gradient_of_Loss_batch(self, theta_batch):
        grad_fn = torch.func.grad(self.loss_func.Loss)
        return torch.vmap(grad_fn)(theta_batch)

#    def M_sqrt_inv_from_grad_batch(self, theta_batch):
#        g = self.gradient_of_Loss_batch(theta_batch)   # (M, D)
#        M, D = g.shape
#
#        I = torch.eye(D, dtype=g.dtype, device=g.device).expand(M, D, D)
#        s = torch.sum(g * g, dim=1, keepdim=True)      # (M, 1)
#
#        mask = s < self.eps
#        safe_s = torch.where(mask, torch.ones_like(s), s)
#
#        coeff = 1.0 - 1.0 / torch.sqrt(1.0 + s)        # (M, 1)
#        outer = g.unsqueeze(2) * g.unsqueeze(1)        # (M, D, D)
#
#        out = I - coeff.view(M, 1, 1) * outer / safe_s.view(M, 1, 1)
#        return torch.where(mask.view(M, 1, 1), I, out)

    def M_sqrt_inv_from_grad_batch(self, theta_batch):
        g = self.gradient_of_Loss_batch(theta_batch)        # (M, D)
        M, D = g.shape
    
        I = torch.eye(D, dtype=g.dtype, device=g.device).expand(M, D, D)
        s = torch.sum(g * g, dim=1, keepdim=True)           # (M, 1)
    
        coeff = 1.0 / (1.0 + s + torch.sqrt(1.0 + s))       # (M, 1)
        outer = g.unsqueeze(2) * g.unsqueeze(1)             # (M, D, D)
    
        return I - coeff.view(M, 1, 1) * outer

class Euclidean_sampler: 
    def __init__(self, walk_len = 500, step_size = 0.01, n_samples = 100): #prior_mean=None, prior_std=1.0):
        self.walk_len = walk_len
        self.step_size = step_size
        self.n_samples = n_samples
        #self.prior_mean = prior_mean
        #self.prior_std = prior_std
    
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
    
    #Changed to faster version
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

    #def sample_prior(self, dtype, device, dim):
    #    z = torch.randn(self.n_samples, dim, dtype=dtype, device=device)
    #    if self.prior_mean is not None:
    #        return self.prior_mean.to(dtype=dtype, device=device) + self.prior_std * z
    #    return self.prior_std * z

class Brownian_sampler:
    def __init__(self, metric, walk_len = 500, step_size = 0.01, n_samples = 100):# prior_mean=None, prior_std=1.0):
        self.metric = metric
        self.walk_len = walk_len
        self.n_samples = n_samples
        self.step_size = step_size

        #self.prior_mean = prior_mean
        #self.prior_std = prior_std

    def compute_BM(self, start_point, random_state = None, drift = False, step_size = None, walk_len = None ):
        if random_state is not None:
            torch.manual_seed(random_state)
    
        theta = start_point.detach().clone().requires_grad_(True)
        path = [theta.clone()]
        losses = []

        if step_size is None: 
            dt = torch.tensor(self.step_size, dtype = theta.dtype, device = theta.device)
        else: 
            dt = step_size
        sqrt_dt = torch.sqrt(dt)

        if walk_len is None: 
            T = self.walk_len
        else: 
            T = walk_len
        
        for _ in range(T):
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
    
    #Changed for speed
    def sample_q_endpoints(self, mu, detach_from_mu=False, drift=False, random_state=None, step_size = None, walk_len = None):
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

        if step_size is None: 
            dt = torch.tensor(self.step_size, dtype=mu.dtype, device=mu.device)
        else: 
            dt = step_size
        if walk_len is None: 
            T = self.walk_len
        else: 
            T = walk_len

        sqrt_dt = torch.sqrt(dt)

        # if drift:
        #     raise NotImplementedError(
        #         "Batch version below supports drift=False first. "
        #         "Keep drift=False until the rest is fast."
        #     )

        #for _ in range(self.walk_len):
        #    Minv_sqrt = self.metric.M_sqrt_inv_from_grad_batch(theta)   # (M, D, D)
        #    eps = torch.randn_like(theta)                               # (M, D)
#
        #    if drift:
        #        drift_term = self.metric.compute_drift(theta)
        #        theta = theta + 0.5 * dt * drift_term + sqrt_dt * torch.einsum("mij,mj->mi", Minv_sqrt, eps)
        #    else:
        #        theta = theta + sqrt_dt * torch.einsum("mij,mj->mi", Minv_sqrt, eps)
#
        #return theta.detach() if detach_from_mu else theta
        for _ in range(T):
            Minv_sqrt = self.metric.M_sqrt_inv_from_grad_batch(theta)   # (M, D, D)
            eps = torch.randn_like(theta)                               # (M, D)

            if drift:
                drift_term = torch.stack([self.metric.compute_drift(theta[m])
                    for m in range(theta.shape[0])])
                theta = theta + 0.5 * dt * drift_term + sqrt_dt * torch.einsum("mij,mj->mi", Minv_sqrt, eps)
            else:
                theta = theta + sqrt_dt * torch.einsum("mij,mj->mi", Minv_sqrt, eps)

            #if drift:
            #    drift_term = self.metric.compute_drift(theta)
            #    theta = theta + 0.5 * dt * drift_term + sqrt_dt * torch.einsum("mij,mj->mi", Minv_sqrt, eps)
            #else:
            #    theta = theta + sqrt_dt * torch.einsum("mij,mj->mi", Minv_sqrt, eps)

        return theta.detach() if detach_from_mu else theta
    
    #def sample_prior(self, dtype, device, dim):
    #    z = torch.randn(self.n_samples, dim, dtype=dtype, device=device)
    #    if self.prior_mean is not None:
    #        return self.prior_mean.to(dtype=dtype, device=device) + self.prior_std * z
    #    return self.prior_std * z
    #
    
    def sample_prior(self, dtype, device, dim):
        return torch.randn(self.n_samples, dim, dtype = dtype, device = device) 
    

#N_q and N_p are the number of samples from posterior and prior in each step
#VAN is for the KL divergence based on a classifier trying to distinguish samples from prior and posterior
#MMD: the MMD distance between posterior and prior samples
#Everything is casted to float 64 for better precosion

def VI_Brownian(mu_start, loss, sampler, 
                n_opt_steps = 100, # antal VI iterations
                lr_VI = 1e-1, #learning rate på mu
                beta = 1.0, #beta * KL
                sigma = 1.0, #Kernel Size på MMD
                KL = 'VAN', #type of KL apprx
                VAN_start_opt_N = 100,
                VAN_opt_N = 10,
                seed = None, 
                drift = False, 
                opt_step_size = False, #optimize on stepsize or not
                lr_dt = 1e-1, #lr of step size
                bound_dt = False, #tau = T*dt
                dt_max = 0.01,
                opt_type = torch.optim.SGD #or Adam
                ):

    if seed is not None: 
        torch.manual_seed(seed)
    
    mu = torch.nn.Parameter(mu_start.to(dtype=torch.float64))
    eps = 1e-12 #num stability

    T = None
    dt = None #use default from sampler

    if opt_step_size: 
        raw_dt = torch.nn.Parameter(torch.tensor(sampler.step_size, dtype=mu.dtype, device=mu.device))
        optimizer = opt_type(
        [{"params": [mu],     "lr": lr_VI},
        {"params": [raw_dt], "lr": lr_dt},])
        
        if bound_dt: #vary T 
            T = sampler.walk_len

    else: 
        optimizer = opt_type([mu], lr=lr_VI)
        
    dim = mu.shape[-1]

    mu_history = [mu.detach().cpu().numpy().copy()]
    obj_history = [] 
    exp_history = []
    kl_history = []
    dt_history = []
    T_history = []
    

    if KL == 'VAN':
        ratio_net = RatioNet(dim).to(dtype=torch.float64)
        if opt_step_size:
                dt = F.relu(raw_dt) + eps
                if bound_dt and dt > dt_max: 
                    T = T + 1
                    dt = dt * (T - 1) / T
                    raw_dt.data = dt
        X_q_det = sampler.sample_q_endpoints(mu, detach_from_mu=True, drift = drift, step_size = dt, walk_len = T)
        X_p_det = sampler.sample_prior(mu.dtype, mu.device, dim)

        #Train discriminator on detached samples            
        train_ratio_net(ratio_net, X_q_det, X_p_det,  opt_type_R = opt_type, n_steps = VAN_start_opt_N)

        for k in range(n_opt_steps): 

            if opt_step_size:
                dt = F.relu(raw_dt) + eps
                if bound_dt and dt > dt_max: 
                    T = T + 1
                    dt = dt * (T - 1) / T
                    raw_dt.data = dt

            X_q_det = sampler.sample_q_endpoints(mu, detach_from_mu=True, drift = drift, step_size = dt, walk_len = T)
            X_p_det = sampler.sample_prior(mu.dtype, mu.device, dim)

            #Train discriminator on detached samples            
            train_ratio_net(ratio_net, X_q_det, X_p_det,  opt_type_R = opt_type, n_steps = VAN_opt_N)
              # ---- DIAGNOSTIC: how separable are q and p to the classifier? ----
            #with torch.no_grad():
            #    logits_q = ratio_net(X_q_det)
            #    logits_p = ratio_net(X_p_det)
            #    acc = ((logits_q > 0).double().mean()
            #           + (logits_p < 0).double().mean()) / 2
            #    print(
            #        f"  [diag] acc={acc.item():.3f} | "
            #        f"logits_q={logits_q.mean().item():+.2f}±{logits_q.std().item():.2f} | "
            #        f"logits_p={logits_p.mean().item():+.2f}±{logits_p.std().item():.2f} | "
            #        f"||mu||={mu.detach().norm().item():.2f}"
            #    )

            #Freeze discriminator and update mu
            for p in ratio_net.parameters():
                p.requires_grad_(False)
            
            X_q = sampler.sample_q_endpoints(mu, detach_from_mu=False, drift = drift, step_size = dt, walk_len = T)

            kl_term = ratio_net(X_q).mean()
            exp_term = loss.Loss_batch(X_q).mean()

            objective = exp_term + beta * kl_term   # minimize negative ELBO

            optimizer.zero_grad()
            objective.backward()
            optimizer.step()

            for p in ratio_net.parameters():
                p.requires_grad_(True)
            
            obj_history.append(objective.item())
            exp_history.append(exp_term.item())
            kl_history.append(kl_term.item())
            mu_history.append(mu.detach().cpu().numpy().copy())

            if opt_step_size: 
                dt_history.append(dt.detach().cpu().numpy().copy())
                T_history.append(T)
        


            if (k + 1) % 10 == 0: #print stuff
                    if opt_step_size:
                        print(
                        f"iter {k+1:4d} | obj={objective.item():.6f} "
                        f"| E[L]={exp_term.item():.6f} | KL={kl_term.item():.6f} "
                        f"| mu={mu.detach().cpu().numpy()}", 
                        f"| step size={dt.detach().cpu().numpy()}"
                        f"| walk len={T}")
                    else: 
                        print(
                        f"iter {k+1:4d} | obj={objective.item():.6f} "
                        f"| E[L]={exp_term.item():.6f} | KL={kl_term.item():.6f} "
                        f"| mu={mu.detach().cpu().numpy()}")

    if KL == 'MMD':
        for k in range(n_opt_steps):
            if opt_step_size:
                dt = F.relu(raw_dt) + eps
                if bound_dt and dt > dt_max: 
                    T = T + 1
                    dt = dt * (T - 1) / T
                    raw_dt.data = dt
            X_q = sampler.sample_q_endpoints(mu, detach_from_mu = False, drift = drift, step_size = dt, walk_len = T)
            X_p = sampler.sample_prior(mu.dtype, mu.device, dim)
            
            kl_term = MMD_loss_biased(X_q, X_p, sigma)
    
            exp_term = loss.Loss_batch(X_q).mean()
            objective = exp_term + beta * kl_term   # minimize negative ELBO

            optimizer.zero_grad()
            objective.backward()
            optimizer.step()

            obj_history.append(objective.item())
            exp_history.append(exp_term.item())
            kl_history.append(kl_term.item())
            mu_history.append(mu.detach().cpu().numpy().copy())
            if opt_step_size: 
                dt_history.append(dt.detach().cpu().numpy().copy())
                T_history.append(T)


            if (k + 1) % 10 == 0: #print stuff
                    if opt_step_size:
                        print(
                        f"iter {k+1:4d} | obj={objective.item():.6f} "
                        f"| E[L]={exp_term.item():.6f} | KL={kl_term.item():.6f} "
                        f"| mu={mu.detach().cpu().numpy()}", 
                        f"| step size={dt.detach().cpu().numpy()}"
                        f"| walk len={T}")
                    else: 
                        print(
                        f"iter {k+1:4d} | obj={objective.item():.6f} "
                        f"| E[L]={exp_term.item():.6f} | KL={kl_term.item():.6f} "
                        f"| mu={mu.detach().cpu().numpy()}")
    if opt_step_size:
        return (
            mu.detach().cpu(), 
            np.array(mu_history),
            np.array(obj_history),
            np.array(exp_history),
            np.array(kl_history),
            np.array(dt_history),
            np.array(T_history))
    else:
        return (
                mu.detach().cpu(), 
                np.array(mu_history),
                np.array(obj_history),
                np.array(exp_history),
                np.array(kl_history))
    