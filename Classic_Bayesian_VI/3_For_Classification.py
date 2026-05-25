
# Bayesian Neural Network for Classification
# Note this example was omitted in the project report
import torch
import torch.nn as nn
import torch.distributions as D
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
SEED = 0
rng = np.random.default_rng(SEED)
DTYPE  = torch.float64

#torch.manual_seed(SEED)
#np.random.seed(SEED)

#Import data: 
df = pd.read_csv('/Users/kirstineholt/Desktop/Skole/Bachelor kode/Classic_Bayesian_VI/banana.csv')
banan = df.values.tolist()
banan = np.array(banan)
X_train, X_test, y_train, y_test = train_test_split(banan[:,0:2], banan[:,2], test_size=0.1)
input_dim_data = np.shape(X_train)[1]
X_train = torch.tensor(X_train, dtype= DTYPE)
X_test = torch.tensor(X_test, dtype= DTYPE)
y_train = torch.tensor(y_train, dtype= DTYPE).unsqueeze(1)
y_test = torch.tensor(y_test, dtype= DTYPE).unsqueeze(1)

#SET VARS 
beta = 0.01                    # KL scaling (VERY IMPORTANT)
prior_type = "gaussian"        # "gaussian" or "mixture"
sigma_y = 0.1   #data variance
n_classes = 2
N = 20 #number of data points
epochs = 7000 # number of iterations
S = 5                         # MC samples per iteration
hidden_N = [20,20]
activation = 'tanh'

def softplus(x):
    return np.logaddexp(0.0, x)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bernoulli_loglikelihood(y, logits):
    return (
        y * (-torch.logaddexp(torch.zeros_like(logits), -logits))
        + (1.0 - y) * (-torch.logaddexp(torch.zeros_like(logits), logits))
    )


class BayesianLinear(nn.Module, ):
    def __init__(self, beta, in_features, out_features, prior_type="gaussian"):
        super().__init__()

        self.beta = beta
        self.prior_type = prior_type

        # Variational parameters
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features,dtype = DTYPE))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -3.0,dtype = DTYPE))
        self.bias_mu   = nn.Parameter(torch.zeros(out_features,dtype = DTYPE))
        self.bias_rho  = nn.Parameter(torch.full((out_features,), -3.0,dtype = DTYPE))

        # Prior hyperparameters
        self.sigma_p = 1.0
        self.alpha = 0.5
        self.sigma_p1 = 1.5
        self.sigma_p2 = 0.1


    def softplus(self, rho):
        return torch.log1p(torch.exp(rho))

    def log_prior(self, w):
        if self.prior_type == "gaussian":
            return D.Normal(0, self.sigma_p).log_prob(w).sum()
        else:
            p1 = torch.exp(D.Normal(0, self.sigma_p1).log_prob(w))
            p2 = torch.exp(D.Normal(0, self.sigma_p2).log_prob(w))
            return torch.log(self.alpha*p1 + (1-self.alpha)*p2 + 1e-8).sum()

    def log_variational(self, w, mu, sigma):
        return D.Normal(mu, sigma).log_prob(w).sum()
    
    
    def forward(self, x):
        w_sigma = self.softplus(self.weight_rho)
        b_sigma = self.softplus(self.bias_rho)

        eps_w = torch.randn_like(w_sigma)
        eps_b = torch.randn_like(b_sigma)

        weight = self.weight_mu + w_sigma * eps_w
        bias   = self.bias_mu + b_sigma * eps_b

        log_q = (
            self.log_variational(weight, self.weight_mu, w_sigma)
            + self.log_variational(bias, self.bias_mu, b_sigma))
        
        log_p = self.log_prior(weight) + self.log_prior(bias)

        return x @ weight.t() + bias, self.beta * (log_q - log_p)

class BayesianNN_classification(nn.Module):
    def __init__(self,beta,input_dim=input_dim_data,hidden_dims=(20,),      # 1 layer: e.g. (20,) or 2 layers: (20, 20)
        output_dim=1, #logits
        prior_type="gaussian",
        activation_type  = 'tanh'):

        super().__init__()

        dims = [input_dim] + list(hidden_dims) + [output_dim]

        self.layers = nn.ModuleList([
            BayesianLinear(beta, dims[i], dims[i+1], prior_type)
            for i in range(len(dims) - 1)])
        
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
        

    def forward(self, x):
        kl_total = 0.0

        for i, layer in enumerate(self.layers):
            x, kl = layer(x)
            kl_total += kl

            # Apply activation to all but last layer
            if i < len(self.layers) - 1:
                x = self.activation(x)
    
            
        return x, kl_total


# INDSTIL modeller

if len(hidden_N) == 1: 
    model = BayesianNN_classification(beta,
    hidden_dims=(hidden_N[0],), 
    beta=beta,
    prior_type=prior_type, 
    activation_type = activation
)
if len(hidden_N) == 2:
    model = BayesianNN_classification(beta,
        hidden_dims=(hidden_N[0], hidden_N[1]),
        prior_type=prior_type, activation_type = activation
    )

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


lik_hist, kl_hist, elbo_hist = [], [], []

# Training Loop

for _ in range(epochs):
    optimizer.zero_grad()

    log_lik = 0.0
    kl = 0.0

    for _ in range(S):
        logits, kl_term = model(X_train)
        log_lik += bernoulli_loglikelihood(y_train, logits).mean()
        kl += kl_term

    log_lik /= S
    kl /= S
    elbo = log_lik - kl

    (-elbo).backward()
    optimizer.step()

    lik_hist.append(-(log_lik.item()))
    kl_hist.append((kl.item()))
    elbo_hist.append(-(elbo.item()))


# Plot 1: ELBO convergence

plt.figure(figsize=(8,4))
plt.plot(lik_hist, label="Likelihood")
plt.plot(kl_hist, label="β·KL")
plt.plot(elbo_hist, label="ELBO")
plt.legend()
plt.title("ELBO convergence")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

def plot_predictive_probability(
    model,
    X,
    y,
    xlim=(-5, 5),
    ylim=(-5, 5),
    grid_size=300,
    n_samples=200,
    title="Predictive probability $P(y=1\\mid x)$"
):
    x1 = np.linspace(*xlim, grid_size)
    x2 = np.linspace(*ylim, grid_size)
    X1, X2 = np.meshgrid(x1, x2)
    grid = torch.tensor(
        np.column_stack([X1.ravel(), X2.ravel()]),
        dtype=X.dtype
    )

    probs = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits, _ = model(grid)
            probs.append(torch.sigmoid(logits))

    probs = torch.stack(probs)
    mean_prob = probs.mean(0).reshape(grid_size, grid_size).cpu().numpy()

    plt.figure(figsize=(8, 6))
    h = plt.contourf(
        X1, X2, mean_prob,
        levels=40,
        cmap="RdBu",
        vmin=0.0,
        vmax=1.0,
        alpha=0.85
    )
    cbar = plt.colorbar(h)
    cbar.set_label("$P(y=1 \\mid x)$")

    plt.scatter(
        X[y.squeeze() == -1, 0].cpu(),
        X[y.squeeze() == -1, 1].cpu(),
        c="royalblue",
        s=20,
        edgecolor="k",
        label="Class 0"
    )
    plt.scatter(
        X[y.squeeze() == 1, 0].cpu(),
        X[y.squeeze() == 1, 1].cpu(),
        c="crimson",
        s=20,
        edgecolor="k",
        label="Class 1"
    )

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title(title)
    plt.legend()
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.tight_layout()
    plt.show()

def plot_epistemic_uncertainty(
    model,
    X,
    y,
    xlim=(-5, 5),
    ylim=(-5, 5),
    grid_size=300,
    n_samples=200,
    title="Epistemic uncertainty $\\mathrm{Var}_{q(\\theta)}[P(y=1\\mid x)]$"
):
    x1 = np.linspace(*xlim, grid_size)
    x2 = np.linspace(*ylim, grid_size)
    X1, X2 = np.meshgrid(x1, x2)
    grid = torch.tensor(
        np.column_stack([X1.ravel(), X2.ravel()]),
        dtype=X.dtype
    )

    probs = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits, _ = model(grid)
            probs.append(torch.sigmoid(logits))

    probs = torch.stack(probs)
    var_prob = probs.var(0).reshape(grid_size, grid_size).cpu().numpy()

    plt.figure(figsize=(8, 6))
    h = plt.contourf(
        X1, X2, var_prob,
        levels=40,
        cmap="magma",
        alpha=0.85
    )
    cbar = plt.colorbar(h)
    cbar.set_label("uncertainty (Posterior predictive standard deviation)")

    plt.scatter(
        X[y.squeeze() == -1, 0].cpu(),
        X[y.squeeze() == -1, 1].cpu(),
        c="royalblue",
        s=20,
        edgecolor="k",
        label="Class 0"
    )
    plt.scatter(
        X[y.squeeze() == 1, 0].cpu(),
        X[y.squeeze() == 1, 1].cpu(),
        c="crimson",
        s=20,
        edgecolor="k",
        label="Class 1"
    )

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title(title)
    plt.legend()
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.tight_layout()
    plt.show()



plot_predictive_probability(model, X_test, y_test)
plot_epistemic_uncertainty(model, X_test, y_test)
