import torch
import torch.nn as nn


class BPNet(nn.Module):
    def __init__(self, input_dim, input_channel, hidden_dim, output_dim):
        super(BPNet, self).__init__()
        self.fc1 = nn.Linear(input_dim * input_channel, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim) 

    def forward(self, x):
        return self.fc2(self.fc1(x))


class BioNet(nn.Module):

    def __init__(self, hidden_weights, output_dim):
        super(BioNet, self).__init__()
        self.bio_layeer = nn.Linear(hidden_weights.shape[1], hidden_weights.shape[0], bias=False)
        self.fc = nn.Linear(hidden_weights.shape[0], output_dim)
        # set bio layer weight
        self.bio_layeer.weight.data = torch.tensor(hidden_weights)

    def forward(self, x):
        return self.fc(self.bio_layer(x)) 
