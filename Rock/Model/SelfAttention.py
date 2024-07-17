import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, input_feature, output_feature, mode="cov"):
        super().__init__()
        self.output_dim = output_feature
        self.input_dim = input_feature
        self.d_k = output_feature ** 0.5
        self.mode = mode

        if self.mode == "cov":
            self.Q_layer = nn.Linear(self.input_dim, self.output_dim * self.output_dim)
            self.K_layer = nn.Linear(self.input_dim, self.output_dim * self.output_dim)
            self.V_layer = nn.Linear(self.input_dim, self.output_dim * self.output_dim)

        else:

            self.Q_layer = nn.Linear(self.input_dim, self.output_dim)
            self.K_layer = nn.Linear(self.input_dim, self.output_dim)
            self.V_layer = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        Q = self.Q_layer(x)
        K = self.K_layer(x)
        V = self.V_layer(x)

        scores = torch.matmul(Q, K.T) / self.d_k
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        if self.mode == "cov":
            attn_output = attn_output.view(-1, self.output_dim, self.output_dim)

        return attn_output, attn_weights


def torch_cov(X):
    X_mean = torch.mean(X, dim=0)
    X_centered = X - X_mean
    n_samples = X.shape[0]
    covariance_matrix = torch.mm(X_centered.T, X_centered) / (n_samples - 1)
    return covariance_matrix
