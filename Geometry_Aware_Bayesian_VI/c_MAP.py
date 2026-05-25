import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.optim as optim

torch.set_default_dtype(torch.float64) 

#Trains the Network to optimality - WEIGHTS ARE NOT DISTRIBUTIONS

class Net_1_Layer(nn.Module):
  def __init__(self,n_hidden, activation_type = 'tanh'):
    super(Net_1_Layer, self).__init__()
    self.hidden = nn.Linear(1, n_hidden)
    self.output = nn.Linear(n_hidden, 1)

    # vælg activation
    if activation_type == "softplus":
            self.activation = nn.Softplus()
    elif activation_type == "relu":
        self.activation = torch.relu
    elif activation_type == "tanh":
        self.activation = torch.tanh
    elif activation_type == "sigmoid":
        self.activation = torch.sigmoid
    elif activation_type == "silu":
        self.activation = nn.SiLU()
    else:
        raise ValueError("Unknown activation type")
  
  def forward(self, x):
    x = self.activation(self.hidden(x))
    x = self.output(x)
    return x

class Net_2_Layer(nn.Module):
  def __init__(self,n_hidden_layer_1, n_hidden_layer_2, activation_type = 'tanh'):
    super(Net_2_Layer, self).__init__()
    self.hidden1 = nn.Linear(1, n_hidden_layer_1)
    self.hidden2 = nn.Linear(n_hidden_layer_1, n_hidden_layer_2)
    self.output = nn.Linear(n_hidden_layer_2, 1)

    # vælg activation
    if activation_type == "softplus":
            self.activation = nn.Softplus()
    elif activation_type == "relu":
        self.activation = torch.relu
    elif activation_type == "tanh":
        self.activation = torch.tanh
    elif activation_type == "sigmoid":
        self.activation = torch.sigmoid
    elif activation_type == "silu":
        self.activation = nn.SiLU()
    else:
        raise ValueError("Unknown activation type")
  
  def forward(self, x):
    x = self.activation(self.hidden1(x))
    x = self.activation(self.hidden2(x))
    x = self.output(x)
    return x
  
#x and y data must be tensors
def Train_Net(model, n_opt, x_data, y_data, lr =0.001 ):
    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = lr) 
    loss_triggered = False
    for step in range(n_opt): 
        outputs = model(x_data)
        step_loss = loss(outputs, y_data)
        optimizer.zero_grad()
        step_loss.backward()
        optimizer.step()

        if step == n_opt-1:
            print(f"step [{step+1}/{n_opt}], Loss: {step_loss.item():.4f}")
        if step_loss == 0 and not loss_triggered: 
            print(f"Loss zero at [{step+1}/{n_opt}], Loss: {step_loss.item():.4f}")
            loss_triggered = True

    theta = torch.cat([p.detach().flatten() for p in model.parameters()])
    return theta
