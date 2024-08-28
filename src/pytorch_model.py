import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, H):
        super(Network, self).__init__()
        self.layers = {}
        for l in range(1, len(H)):
            self.layers['l' + str(l)] = nn.Linear(H[l-1], H[l])
        self.loss_function = nn.BCELoss()
    
    def forward(self, X, H):
        L = len(H) - 1
        activations = {'A0':X}
        for l in range(1, L+1):
            if(l < len(H) - 1):
                activations['A' + str(l)] = F.relu(self.layers['l' + str(l)](activations['A' + str(l-1)]))
            else:
                activations['A' + str(l)] = torch.sigmoid(self.layers['l' + str(l)](activations['A' + str(l-1)]))
        return activations['A' + str(L)]
    
    def calculate_loss(self, yhat, y):
        loss = self.loss_function(yhat, y)
        return loss
    
    def backward(self, loss):
        loss.backward()