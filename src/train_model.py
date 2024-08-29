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

def load_data():
    data = pd.read_csv('../framingham.csv')
    data = data.dropna()
    train, test = train_test_split(data, train_size=0.85, random_state=10)
    X_train, y_train, scalers = prepare_train(train)
    X_test, y_test = prepare_test(test, scalers)
    return X_train, y_train, X_test, y_test

def initialize_params(H):
    assert H[-1] == 1
    params = {}
    param_count = 0
    msg = ''
    for l in range(1, len(H)):
        #If L=3, this will go l=1,2,3
        params['W' + str(l)] = np.random.randn(H[l-1], H[l]) * 0.1
        params['b' + str(l)] = np.zeros((1, H[l]))
        weight_count = H[l-1] * H[l]
        bias_count = H[l]
        msg += f'\nlayer {l}: {H[l]} hidden units, {weight_count + bias_count} params, {weight_count} weights, {bias_count} biases'
        param_count += weight_count + bias_count

    print(f'initialized {param_count} total trainable params over {len(H) -1} layers \n' + msg)
    return params

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    a = 1/(1 + np.exp(-z))
    return a

def forward(X, params, H):
    activations = {'A0':X}
    inter_values = {}
    for l in range(1, len(H)):
        inter_values['Z' + str(l)] = activations['A' + str(l-1)] @ params['W' + str(l)] + params['b' + str(l)]
        if(l < len(H) - 1): #ReLU until last layer, then sigmoid
            activations['A' + str(l)] = relu(inter_values['Z' + str(l)])
            #print(f'model activations A{str(l)} {activations['A' + str(l)]}')
        else:
            activations['A' + str(l)] = sigmoid(inter_values['Z' + str(l)])
            #print(f'model activations A{str(l)} {activations['A' + str(l)]}')

    return activations, inter_values