from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import platform
import torch
import torch.nn as nn
import torch.nn.functional as F

if True:
    import sys
    system = platform.system()
    if system == "Windows":
        sys.path.insert(0, 'C:/Users/Lorenzo/Desktop/Workspace/Github/Project-5/src')
    elif system == "Darwin":
        sys.path.insert(0, '/Users/lorenzogurrola/workspace/github.com/LorenzoGurrola/Project-5/src')
    from data_loader import prepare_train, prepare_test

class NeuralNetwork(nn.Module):
    def __init__(self, n, h):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(n, h)
        self.l2 = nn.Linear(h, 1)
        self.loss_function = nn.BCELoss()
    
    def forward(self, X):
        A1 = F.relu(self.l1(X))
        A2 = torch.sigmoid(self.l2(A1))
        return A2
    
    def calculate_loss(self, yhat, y):
        loss = self.loss_function(yhat, y)
        return loss
    
    def backward(self, loss):
        loss.backward()

def load_data():
    data = pd.read_csv('../framingham.csv')
    data = data.dropna()
    train, test = train_test_split(data, train_size=0.85, random_state=10)
    X_train, y_train, scalers = prepare_train(train)
    X_test, y_test = prepare_test(test, scalers)
    return X_train, y_train, X_test, y_test

def initialize_params(n, h):
    W1 = np.random.randn(n, h) * 0.1
    b1 = np.zeros((1, h))
    w2 = np.random.randn(h, 1) * 0.1
    b2 = np.zeros((1, 1))
    params = {'W1':W1, 'b1':b1, 'w2':w2, 'b2':b2}
    param_count = n * h + 2 * h + 1
    print(f'initialized {param_count} total trainable params with {h} hidden units and {n} input features')
    return params

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    a = 1/(1 + np.exp(-z))
    return a

def forward(X, params):
    W1 = params['W1']
    b1 = params['b1']
    w2 = params['w2']
    b2 = params['b2']

    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    inter_vals = {'Z1':Z1, 'A1':A1}

    z2 = A1 @ w2 + b2
    a2 = sigmoid(z2)

    return a2, inter_vals

def calculate_cost(yhat, y):
    m = y.shape[0]
    losses = y * np.log(yhat) + (1 - y) * np.log(1 - yhat)
    cost = -np.sum(losses, axis=0, keepdims=True)/m
    return cost

def backward(y, yhat, inter_vals, X, params):
    m = y.shape[0]
    A1 = inter_vals['A1']
    Z1 = inter_vals['Z1']
    w2 = params['w2']
    dc_dyhat = (-1/m) * ((y/yhat) - ((1 - y)/(1 - yhat)))
    dyhat_dz2 = yhat * (1 - yhat)
    dc_dz2 = dc_dyhat * dyhat_dz2
    dc_db2 = np.sum(dc_dz2, axis=0, keepdims=True)
    dc_dw2 = np.matmul(A1.T, dc_dz2)

    dc_dA1 = np.matmul(dc_dz2, w2.T)
    dA1_dZ1 = np.where(Z1 >= 0, 1, 0)
    dc_dZ1 = dc_dA1 * dA1_dZ1
    dc_db1 = np.sum(dc_dZ1, axis=0, keepdims=True)
    dc_dW1 = np.matmul(X.T, dc_dZ1)

    grads = {'dW1':dc_dW1, 'db1':dc_db1, 'dw2':dc_dw2, 'db2':dc_db2}
    return grads