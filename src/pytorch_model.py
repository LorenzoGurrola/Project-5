import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, H):
        super(Network, self).__init__()
        self.layers = {}
        for l in range(1, len(H)):
            self.layers['l' + str(l)] = nn.Linear(H[l-1], H[l])
        print(self.layers)
        self.loss_function = nn.BCELoss()
    
    def forward(self, X, H):
        L = len(H) - 1
        activations = {'A0':X}
        inter_values = {}
        for l in range(1, L+1):
            inter_values['Z' + str(l)] = self.layers['l' + str(l)](activations['A' + str(l-1)])
            if(l < len(H) - 1):
                activations['A' + str(l)] = F.relu(inter_values['Z' + str(l)])
                #print(f'Pytorch Network activations A{str(l)} {activations['A' + str(l)]}')
            else:
                activations['A' + str(l)] = torch.sigmoid(inter_values['Z' + str(l)])
                #print(f'Pytorch Network activations A{str(l)} {activations['A' + str(l)]}')
        return activations, inter_values
    
    def calculate_loss(self, yhat, y):
        loss = self.loss_function(yhat, y)
        return loss
    
    def backward(self, loss):
        loss.backward()