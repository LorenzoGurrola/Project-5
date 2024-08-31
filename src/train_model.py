from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import platform
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

if True:
    import sys
    system = platform.system()
    if system == "Windows":
        sys.path.insert(0, 'C:/Users/Lorenzo/Desktop/Workspace/Github/Project-5/src')
    elif system == "Darwin":
        sys.path.insert(0, '/Users/lorenzogurrola/workspace/github.com/LorenzoGurrola/Project-5/src')
    from data_loader import load_sets


class DenseNeuralNetwork():
    def __init__(self, config):
        self.name = config['name']
        self.hidden_units = config['hidden_units']
        self.path = f'../src/models/{self.name}'
        if os.path.exists(self.path):
            print(f'ERROR: model {self.name} already exists')
        else:
            os.makedirs(self.path)
            print(f'created new model {self.name}')
            np.save(f'{self.path}/hidden_units.npy', self.hidden_units)
            self.params = self.initialize_params(self.hidden_units)
            for p in self.params:
                param_path = f'{self.path}/{p}.npy'
                np.save(param_path, self.params[p])
    
    def load_data():
        X_train, y_train, X_test, y_test = load_sets()
        return X_train, y_train, X_test, y_test
    
    def initialize_params(self, H):
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
        print(self.path)
        return params
    
    def sigmoid(z):
        a = 1/(1 + np.exp(-z))
        return a
    
    def relu(z):
        return np.maximum(0, z)
    
    def forward(self, X, params, H):
        activations = {'A0':X}
        inter_values = {}
        for l in range(1, len(H)):
            inter_values['Z' + str(l)] = activations['A' + str(l-1)] @ params['W' + str(l)] + params['b' + str(l)]
            print(f'model Z{str(l)} {inter_values['Z' + str(l)]}')
            if(l < len(H) - 1): #ReLU until last layer, then sigmoid
                activations['A' + str(l)] = self.relu(inter_values['Z' + str(l)])
                print(f'model activations A{str(l)} {activations['A' + str(l)]}')
            else:
                activations['A' + str(l)] = self.sigmoid(inter_values['Z' + str(l)])
                print(f'model activations A{str(l)} {activations['A' + str(l)]}')

        return activations, inter_values
    
    def calculate_cost(yhat, y):
        m = y.shape[0]
        losses = y * np.log(yhat) + (1 - y) * np.log(1 - yhat)
        cost = -np.sum(losses, axis=0, keepdims=True)/m
        return cost
    
    def backward(y, activations, inter_values, params, H):
        m = y.shape[0]
        L = len(H) - 1
        yhat = activations['A' + str(L)]
        derivatives = {}
        grads = {}
        activations[f'dc_dA{L}'] = (-1/m) * ((y/yhat) - ((1 - y)/(1 - yhat)))
        for layer in range(L, 0, -1):
            l = str(layer)
            if(layer == L):
                derivatives[f'dA{l}_dZ{l}'] = yhat * (1 - yhat)
            else:
                derivatives[f'dA{l}_dZ{l}'] = np.where(activations[f'Z{l}'] >= 0, 1, 0)
            derivatives[f'dc_dZ{l}'] = derivatives[f'dA{l}_dZ{l}'] * activations[f'dc_dA{l}']
            grads[f'db{l}'] = np.sum(derivatives[f'dc_dZ{l}'], axis=0, keepdims=True)
            grads[f'dZ{l}_dW{l}'] = np.matmul(activations[f'A{l}'].T, derivatives[f'dc_dZ{l}'])
        
        return grads, derivatives

    def update_params(params, grads, lr):
        for p in params:
            params[p] = params[p] - lr * grads[f'd{params[p]}']

        return params
    
    def train_loop(self, model, epochs, lr, X_train, y_train, h):
        params = self.initialize_params(X_train.shape[1], h)
        params = self.load_params(model, params)
        for epoch in range(epochs):
            yhat, values = self.forward(X_train, params)
            cost = self.calculate_cost(yhat, y_train)
            grads = self.backward(y_train, yhat, values, X_train, params)
            params = self.update_params(params, grads, lr)
            print(f'epoch {epoch} cost {cost}')
        self.save_params(model, params)
        print(f'saved params in model {model}')

    