import torch.nn as nn
import torch
import numpy as np
from torch.distributions import Categorical, Normal, MixtureSameFamily, MultivariateNormal
import math


class mdn(nn.Module):
    def __init__(self, input_size, output_size, num_gaussian, num_hidden):
        super(mdn, self).__init__()
        self.i_s = input_size
        self.o_s = output_size
        self.n_g = num_gaussian
        self.n_h = num_hidden

        self.root_layer = nn.Sequential(
            nn.BatchNorm1d(self.i_s),
            nn.Linear(self.i_s, self.n_h),
            nn.SiLU(),
            nn.Dropout(),
            nn.Linear(self.n_h, self.n_h),
            nn.SiLU(),
            nn.Dropout(),
            nn.Linear(self.n_h, self.n_h),
            nn.SiLU(),
            nn.Dropout(),
            nn.Linear(self.n_h, self.n_h),
            nn.SiLU(),
            nn.BatchNorm1d(self.n_h),
        ).double()

        self.pi = nn.Sequential(
            nn.Linear(self.n_h, self.n_h),
            nn.SiLU(),
            nn.Linear(self.n_h, self.n_g)
        ).double()

        self.normal_layer = nn.Sequential(
            nn.Linear(self.n_h, self.n_h),
            nn.SiLU(),
            nn.Linear(self.n_h, 2 * (self.o_s * self.n_g))
        ).double()

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
        parameters = self.root_layer(x).double()

        pi = torch.log_softmax(self.pi(parameters), dim=-1)

        mu_sigma = self.normal_layer(parameters)
        mu = mu_sigma[..., :self.o_s * self.n_g]

        sigma = mu_sigma[..., self.o_s * self.n_g:]
        sigma = torch.exp(sigma + eps)

        return pi, mu.reshape(-1, self.n_g, self.o_s), sigma.reshape(-1, self.n_g, self.o_s)


class NLLLoss(nn.Module):
    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, pi, mu, sigma, y):
        z_score = (torch.unsqueeze(y, dim=1) - mu) / sigma

        normal_loglik = (-0.5 * torch.einsum("bij, bij->bi", z_score, z_score)) - torch.sum(torch.log(sigma), dim=-1)

        loglik = -torch.logsumexp(pi + normal_loglik, dim=-1)

        # Construct distribution model
        # mix = Categorical(logits=pi)
        # comp = MultivariateNormal(mu, covariance_matrix=torch.diag_embed(sigma))
        #
        # # Mix
        # mixture_model = MixtureSameFamily(mix, comp)
        #
        # # Calculate
        # loss = -mixture_model.log_prob(y).mean()

        return loglik.mean()


class R2_Evaluation(nn.Module):
    def __init__(self):
        super(R2_Evaluation, self).__init__()

    def forward(self, y_ture, y_pred):
        """
            R-square Calculating
            Formular:
                1 - (SSR(Sum of Squares of Residuals)/SST(Sum of Squares Total))
                SSR = Sum(Pow(y_ture - y_pred), 2)
                SST = Sum(Pow(y_ture - mean(y_ture), 2))
        """
        # SSR
        ssr = torch.sum(torch.pow((y_ture - y_pred), 2), dim=-1)
        # SST
        sst = torch.sum(torch.pow((y_ture - torch.mean(y_ture)), 2))
        # R2
        r2 = 1 - (ssr / sst)

        return r2.mean()


class Sample(nn.Module):
    def __init__(self):
        super(Sample, self).__init__()

    def forward(self, pi, mu, sigma):
        select_idx = torch.multinomial(torch.exp(pi), num_samples=1, replacement=True).squeeze()

        # Advance Indexing
        mu_selected = mu[torch.arange(mu.shape[0]), select_idx, :]
        sigma_selected = sigma[torch.arange(sigma.shape[0]), select_idx, :]

        samples = torch.normal(mean=mu_selected, std=sigma_selected)

        return samples


class RelativeError(nn.Module):
    def __init__(self):
        super(RelativeError, self).__init__()

    def forward(self, y_ture, samples, eps=1e-6):
        relative_error = np.divide((np.abs(samples - y_ture)), np.abs(y_ture + eps)).mean()

        return relative_error








