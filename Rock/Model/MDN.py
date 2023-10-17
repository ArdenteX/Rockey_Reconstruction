import torch.nn as nn
import torch
import math


class mdn(nn.Module):
    def __init__(self, input_size, output_size, num_gaussian, num_hidden):
        super(mdn).__init__()
        self.i_s = input_size
        self.o_s = output_size
        self.n_g = num_gaussian
        self.n_h = num_hidden

        self.pi = nn.Sequential(
            nn.Linear(self.i_s, self.n_h),
            nn.ReLU(),
            nn.Linear(self.n_h, self.n_h),
            nn.ReLU(),
            nn.Linear(self.n_h, self.n_g)
        )

        self.normal_layer = nn.Sequential(
            nn.Linear(self.i_s, self.n_h),
            nn.ReLU(),
            nn.Linear(self.n_h, self.n_h),
            nn.ReLU(),
            nn.Linear(self.n_h, 2 * (self.o_s * self.n_g))
        )

        # Test the performance after the origin
        # self.mu = nn.Sequential(
        #     nn.Linear(self.i_s, self.n_h),
        #     nn.ReLU(),
        #     nn.Linear(self.n_h, self.n_h),
        #     nn.ReLU(),
        #     nn.Linear(self.n_h, self.o_s * self.n_g)
        # )
        #
        # self.sigma = nn.Sequential(
        #     nn.Linear(self.i_s, self.n_h),
        #     nn.ReLU(),
        #     nn.Linear(self.n_h, self.n_h),
        #     nn.ReLU(),
        #     nn.Linear(self.n_h, self.o_s * self.n_g)
        # )

    def forward(self, x, eps=1e-6):
        pi = torch.log_softmax(self.pi(x), dim=-1)
        mu = self.normal_layer[:, :self.o_s * self.n_g]
        sigma = self.normal_layer[:, self.o_s * self.n_g:]
        sigma = torch.exp(sigma + eps)

        return pi, mu.reshape(-1, self.n_g, self.o_s), sigma.reshape(-1, self.n_g, self.o_s)



