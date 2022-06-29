from turtle import forward
from torch import nn
import torch

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l1 = nn.Linear(hidden_size, hidden_size)
        self.l1 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(x)
        out = self.relu(out)
        out = self.l3(x)
        
        return out