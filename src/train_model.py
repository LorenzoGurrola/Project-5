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
        print(self.path)
        return params

def save_params(model, params):
    path = '../src/models/' + model
    for p in params:
        param_path = path + '/' + p + '.npy'
        np.save(param_path, params[p])

    