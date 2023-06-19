import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import argparse

class MultiOutputSignalClassifier(nn.Module):
    def __init__(self, num_positions, num_directions, input_dim, dense_layers, dropout_rate=0.2, batch_norm=True,
                 use_bias=True, activation='relu', l1_reg=0.01, temperature=1.0):
        super(MultiOutputSignalClassifier, self).__init__()
        self.num_positions = num_positions
        self.num_directions = num_directions
        self.input_dim = input_dim
        self.dense_layers = dense_layers
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.activation = activation
        self.l1_reg = l1_reg
        self.temperature = temperature
        self.use_bias = use_bias

        self.hidden_layers = nn.ModuleList()
        for unit in self.dense_layers:
            self.hidden_layers.append(nn.Linear(self.input_dim, unit, bias=self.use_bias))
            self.input_dim = unit

        self.position_layer = nn.Linear(self.input_dim, self.num_positions)
        self.direction_layer = nn.Linear(self.input_dim, self.num_directions)

        self.activation_fn = nn.ReLU(inplace=True)
        self.batch_norm_layers = nn.ModuleList([nn.BatchNorm1d(unit) for unit in self.dense_layers])
        self.dropout_layer = nn.Dropout(self.dropout_rate)

    def forward(self, inputs):
        x = inputs
        penultimate_output = None
        for i in range(len(self.hidden_layers)):
            x = self.hidden_layers[i](x)
            x = self.activation_fn(x)
            if self.batch_norm:
                x = self.batch_norm_layers[i](x)
            x = self.dropout_layer(x)
            # Update penultimate_output after the second-to-last hidden layer
            if i == len(self.hidden_layers) - 1:
                penultimate_output = x

        position = self.position_layer(x)
        direction = self.direction_layer(x)
        return position, direction


