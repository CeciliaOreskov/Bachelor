
# Bayesian Neural Network via Bayes-by-Backprop for Regression tasks

import torch
import torch.nn as nn
import torch.distributions as D
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(7)
np.random.seed(7)

#SET VARS 
beta = 1                    # KL scaling (VERY IMPORTANT)
#prior_type = "gaussian"        # "gaussian" or "mixture"
sigma_y = 0.1   #data variance
N = 100 #number of data points
epochs = 1000 # number of iterations
S = 5                         # MC samples per iteration
hidden_N = [16,16]
activation = 'tanh'
save_to_file = 'file_name'

# Data
x = torch.linspace(-0.9, 0.9, N).unsqueeze(1)
noise = (torch.randn(N)*sigma_y).unsqueeze(1)
y = torch.sin(3*x) + noise 

class BayesianLinear(nn.Module, ):
    def __init__(self, in_features, out_features, prior_type="gaussian", beta=1.0):
        super().__init__()

        self.beta = beta
        self.prior_type = prior_type

        # Variational parameters
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -3.0))
        self.bias_mu   = nn.Parameter(torch.zeros(out_features))
        self.bias_rho  = nn.Parameter(torch.full((out_features,), -3.0))

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

class BayesianNN(nn.Module):
    def __init__(self,input_dim=1,hidden_dims=(20,),      # 1 layer: e.g. (20,) or 2 layers: (20, 20)
        output_dim=1,
        beta=1,
        prior_type="gaussian",
        activation_type  = 'tanh'):

        super().__init__()

        dims = [input_dim] + list(hidden_dims) + [output_dim]

        self.layers = nn.ModuleList([
            BayesianLinear(dims[i], dims[i+1], prior_type, beta)
            for i in range(len(dims) - 1)
        ])
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

def fit_NN(model):
    lik_hist, kl_hist, elbo_hist = [], [], []
    for _ in range(epochs):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        log_lik = 0.0
        kl = 0.0

        for _ in range(S):
            y_hat, kl_term = model(x)
            log_lik += D.Normal(y_hat, sigma_y).log_prob(y).sum()
            kl += kl_term

        log_lik /= S
        kl /= S
        elbo = log_lik - kl

        (-elbo).backward()
        optimizer.step()

        lik_hist.append(-(log_lik.item()))
        kl_hist.append((kl.item()))
        elbo_hist.append(-(elbo.item()))
    return lik_hist, kl_hist, elbo_hist

# Plot 1: ELBO convergence
def plot_Convergence(lik_hist, kl_hist, elbo_hist, plot_save, plot_title = '1 layer model, $p(θ) =\mathcal{N}(0,1)$' ):
    plt.figure(figsize=(8,4))
    plt.plot(lik_hist, label="negative Likelihood")
    plt.plot(kl_hist, label= "KL(q||p)")
    plt.plot(elbo_hist, label="negative ELBO")
    plt.legend()
    plt.title(f"ELBO convergence, \n {plot_title} ")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig(f"{plot_save}.pdf") 
    plt.show()

# Posterior Predictive Plot
def plot_predictions(model, plot_save, color_mean = 'red', color_er = "tab:red", plot_title = '1 layer model, $p(θ) =\mathcal{N}(0,1)$'):
    x_test = torch.linspace(-2, 2, 300).unsqueeze(1)

    S_pred = 200        # total posterior samples
    num_models = 20     # number of sampled models to plot

    preds = []

    for _ in range(S_pred):
        y_hat, _ = model(x_test)   # resample weights every pass
        preds.append(y_hat.detach().numpy())

    preds = np.array(preds)        # shape: [S_pred, N, 1]

    # Prepare data for plotting
    x_plot = x_test.squeeze().detach().numpy()
    pred_mean = preds.mean(axis=0).squeeze()
    pred_std  = preds.std(axis=0).squeeze()

    plt.figure(figsize=(8, 5))

    # Data
    plt.scatter(
        x.numpy(),
        y.numpy(),
        c='black',
        s=10,
        label=f"Data (σ = {sigma_y})"
    )

    # Sampled posterior models
    for i in range(num_models):
        plt.plot(
            x_plot,
            preds[i].squeeze(),
            color="steelblue",
            alpha=0.3,
            linewidth=1
        )

    # Predictive mean
    plt.plot(
        x_plot,
        pred_mean,
        color=color_mean,
        linewidth=2.5,
        label="Predictive mean"
    )

    # Epistemic uncertainty ±3σ
    plt.fill_between(
        x_plot,
        pred_mean - 3 * pred_std,
        pred_mean + 3 * pred_std,
        color=color_er,
        alpha=0.25,
        label="Epistemic uncertainty (±3σ)"
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Bayesian Neural Network Posterior Predictive\n {plot_title}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_save}.pdf")  
    plt.show()

# Posterior Predictive Plot multible models
def plot__multible_model_predictions(models, names, plot_save):
    colors = ['firebrick', 'darkorange', 'darkgreen', 'cornflowerblue']
    colors_uncert = ["tab:red", 'tab:orange', 'tab:green', 'tab:blue']
    x_test = torch.linspace(-2, 2, 300).unsqueeze(1)
    S_pred = 200        # total posterior samples
    # Prepare data for plotting
    x_plot = x_test.squeeze().detach().numpy()
    
    

    plt.figure(figsize=(8, 5))

    # Data
    plt.scatter(
        x.numpy(),
        y.numpy(),
        c='black',
        s=10,
        label=f"Data (σ = {sigma_y})",
        zorder = 10
    )

    for i in range(len(models)):
        preds = []
        for _ in range(S_pred):
            y_hat, _ = models[i](x_test)   # resample weights every pass
            preds.append(y_hat.detach().numpy())

        preds = np.array(preds)        # shape: [S_pred, N, 1]
        pred_mean = preds.mean(axis=0).squeeze()
        pred_std  = preds.std(axis=0).squeeze()

        # Predictive mean
        plt.plot(
            x_plot,
            pred_mean,
            color=colors[i],
            linewidth=2.5,
            label=f"Pred mean {names[i]} "
        )

        # Epistemic uncertainty ±3σ
        plt.fill_between(
            x_plot,
            pred_mean - 3 * pred_std,
            pred_mean + 3 * pred_std,
            color=colors_uncert[i],
            alpha=0.25,
            label=f"(μ±3σ) for {names[i]}"
        )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Bayesian Neural Networks Posterior Predictive models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_save}.pdf")
    plt.show()


model_1_layer_N_p = BayesianNN( hidden_dims=(hidden_N[0],),beta=beta,prior_type='gaussian', activation_type = activation)
model_1_layer_mix_p = BayesianNN( hidden_dims=(hidden_N[0],),beta=beta,prior_type='mixture', activation_type = activation)
model_2_layer_N_p = BayesianNN(hidden_dims=(hidden_N[0], hidden_N[1]),beta=beta,prior_type='gaussian', activation_type = activation)
model_2_layer_mix_p = BayesianNN(hidden_dims=(hidden_N[0], hidden_N[1]),beta=beta,prior_type='mixture', activation_type = activation)

lik_hist, kl_hist, elbo_hist = fit_NN(model_1_layer_N_p)
plot_Convergence(lik_hist, kl_hist, elbo_hist, save_to_file, plot_title = '1 layer model, $p(θ) =\mathcal{N}(0,1)$')
plot_predictions(model_1_layer_N_p, save_to_file,  color_mean = 'red', color_er = "tab:red", plot_title = '1 layer model, $p(θ) =\mathcal{N}(0,1)$')

lik_hist, kl_hist, elbo_hist = fit_NN(model_1_layer_mix_p)
plot_Convergence(lik_hist, kl_hist, elbo_hist, save_to_file, plot_title = '1 layer model, $p(θ) =$mix')
plot_predictions(model_1_layer_mix_p, save_to_file,color_mean = 'orange', color_er = 'tab:orange', plot_title = '1 layer model,  $p(θ) =$mix')

lik_hist, kl_hist, elbo_hist = fit_NN(model_2_layer_N_p)
plot_Convergence(lik_hist, kl_hist, elbo_hist, save_to_file, plot_title = '2 layer model, $p(θ) = \mathcal{N}(0,1)$')
plot_predictions(model_1_layer_mix_p, save_to_file, color_mean = 'darkgreen', color_er = "tab:green", plot_title = '2 layer model, $p(θ) =\mathcal{N}(0,1)$')

lik_hist, kl_hist, elbo_hist = fit_NN(model_2_layer_mix_p)
plot_Convergence(lik_hist, kl_hist, elbo_hist, save_to_file, plot_title = '2 layer model, $p(θ) =$mix')
plot_predictions(model_1_layer_mix_p, save_to_file, color_mean = 'cornflowerblue', color_er = 'tab:blue',plot_title = '2 layer model,  $p(θ) =$mix' )

plot__multible_model_predictions([model_1_layer_N_p, model_1_layer_mix_p, model_2_layer_N_p, model_2_layer_mix_p], ['1 layer $p(θ) =\mathcal{N}(0,1)$', '1 layer model, $p(θ) =$mix', '2 layer $p(θ) =\mathcal{N}(0,1)$', '2 layer model, $p(θ) =$mix'], 'together')

