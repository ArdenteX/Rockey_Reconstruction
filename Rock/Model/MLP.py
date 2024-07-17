import torch
import torch.nn as nn
import torch.nn.functional as F
from Rock.Model.SelfAttention import SelfAttention


class MLP_Attention(nn.Module):
    def __init__(self, input_size=10, layer1_size=32, layer2_size=16, output_size=1, mode="cov"):
        super(MLP_Attention, self).__init__()

        self.input_size = input_size
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.output_size = output_size
        self.mode = mode

        self.layer1 = nn.Linear(self.input_size, self.layer1_size).double()
        self.layer2 = nn.Linear(self.layer1_size, self.layer2_size).double()
        self.output = SelfAttention(self.layer2_size, self.output_size, self.mode).double()

    def forward(self, x):
        x = F.dropout(self.layer1(x), p=0.1)
        x = torch.tanh(x)
        x = torch.tanh(self.layer2(x))
        x = self.output(x)
        return x




