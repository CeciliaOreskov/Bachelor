import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import torch
import pandas as pd
from scipy.linalg import lu_factor, lu_solve
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
#PLOT FUNCTIONS: 

def Banan(theta):
    return (theta[0]*theta[1]-1)**2

def plot_loss(loss_fct, w_1_range,w_2_range, alone = 'no', color = 'reds', label = ' ', title = 'Heatmap af weightspace (loss landscape)'):
    w1 = np.linspace(w_1_range[0],w_1_range[1],1000)
    w2 = np.linspace(w_2_range[0],w_2_range[1],1000)
    W1, W2 = np.meshgrid(w1, w2)
    Loss = loss_fct(W1,W2)

    if alone == 'yes':
        plt.figure(figsize=(6, 5))
    plt.imshow(Loss, extent=[w1.min(), w1.max(), w2.min(), w2.max()],origin='lower',aspect='auto',cmap= color, alpha = 0.5, vmin=0, vmax=2)
    plt.colorbar(label='$\mathcal{L}(w_1, w_2)$')
    plt.xlabel('w1')
    plt.ylabel('w2')
    plt.title(title)
    

def plot_brownian(BM, c_line = 'black', c_dots = 'blue'): 
    plt.plot(BM[:, 0], BM[:, 1],  color=c_line , linewidth=1, label='path')
    plt.scatter(BM[:, 0], BM[:, 1],color=c_dots,  s=5)
    plt.scatter(BM[0,0], BM[0,1], c = 'green', s=50, label="start")
    plt.scatter(BM[-1,0], BM[-1,1], c='red', edgecolor='black', s=50, label="end")
    plt.legend()

def plot_optimization_steps(mu_history):
    plt.scatter(mu_history[:,0], mu_history[:,1], s=15, c='red', alpha=0.9)

    # start and end
    plt.scatter(mu_history[0,0], mu_history[0,1],
            s=100, c='yellow', edgecolor='black', label="start")

    plt.scatter(mu_history[-2,0], mu_history[-2,1],
            s=120, c='cyan', edgecolor='black', label="final μ")
    plt.legend()

def plot_loss_opt(steps,loss):
    plt.plot(steps,loss)
    plt.xlabel('nr of step')
    plt.ylabel('loss')
    plt.title('loss over optimisation')

def plot_optimization_overview(loss_fct, w_1_range, w_2_range, mu_history, steps, loss):
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    # ---- Left: Loss landscape + optimization path ----
    plt.sca(axes[0])  # set current axis
    plot_loss(loss_fct, w_1_range, w_2_range)
    plot_optimization_steps(mu_history)
    axes[0].set_title("Loss landscape with optimization steps")

    # ---- Right: Loss vs steps ----
    plt.sca(axes[1])
    plot_loss_opt(steps, loss)
    axes[1].set_title("Loss over optimization")

    plt.tight_layout()
    plt.show()