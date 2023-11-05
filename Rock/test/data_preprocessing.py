import pandas as pd
import torch
import numpy as np
from torch.distributions import Categorical, Normal, MixtureSameFamily, MultivariateNormal, Independent
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from Rock.Model.MDN import mdn, NLLLoss
from sklearn.metrics import r2_score
from Rock.Model.MDN_From_Kaggle import mdn as mdn_advance
from Rock.Train.mdnTraining import mdnTraining
import seaborn as sns
import matplotlib.pyplot as plt


class GenerateData:
    def __init__(self):
        # Data Normalization
        self.data = pd.read_excel('D:\\Resource\\Gas_Giants_Core_Earth20W.xlsx')
        self.data['M_total (M_E)'] = self.data['Mcore (M_J/10^3)'] + self.data['Menv (M_E)']
        self.scaler_x = MinMaxScaler(feature_range=[0, 1])
        self.scaler_y = MinMaxScaler(feature_range=[0, 1])

    def __getitem__(self, item):
        # Generate train data, validation data, and test data (6:2:2)
        train_set = self.data.iloc[:int(len(self.data) * 0.6), :]
        train_x = torch.from_numpy(self.scaler_x.fit_transform(train_set.iloc[:, [0, 1, 5]]).astype('float32'))
        train_y = torch.from_numpy(self.scaler_y.fit_transform(train_set.iloc[:, [6, 8, 9, 11]]).astype('float32'))

        val_set = self.data.iloc[int(len(self.data) * 0.6):int(len(self.data) * 0.8), :]
        val_x = torch.from_numpy(self.scaler_x.fit_transform(val_set.iloc[:, [0, 1, 5]]).astype('float32'))
        val_y = torch.from_numpy(self.scaler_y.fit_transform(val_set.iloc[:, [6, 8, 9, 11]]).astype('float32'))

        return (train_x, train_y), (val_x, val_y)


t_v_set = GenerateData()
# train_loader = DataLoader(t_set)

data = pd.read_excel('D:\\Resource\\Gas_Giants_Core_Earth20W.xlsx')
data['M_total (M_E)'] = data['Mcore (M_J/10^3)'] + data['Menv (M_E)']
scaler_x = MinMaxScaler(feature_range=[-1, 1])
scaler_y = MinMaxScaler(feature_range=[-1, 1])
train_set = data.iloc[:int(len(data) * 0.6), :]
train_x = torch.from_numpy(scaler_x.fit_transform(train_set.iloc[:, [0, 1, 5]])).double()
train_y = torch.from_numpy(scaler_y.fit_transform(train_set.iloc[:, [6, 8, 9, 11]])).double()

val_set = data.iloc[int(len(data) * 0.6):int(len(data) * 0.8), :]
val_x = torch.from_numpy(scaler_x.fit_transform(val_set.iloc[:, [0, 1, 5]])).double()
val_y = torch.from_numpy(scaler_y.fit_transform(val_set.iloc[:, [6, 8, 9, 11]])).double()

# Original Model
train_set = data.iloc[:int(len(data) * 0.6), :]
train_x = torch.from_numpy(scaler_x.fit_transform(train_set.iloc[:, [0, 1, 5]]).astype('float32'))
train_y = torch.from_numpy(scaler_y.fit_transform(train_set.iloc[:, [6, 8, 9, 11]]).astype('float32'))

val_set = data.iloc[int(len(data) * 0.6):int(len(data) * 0.8), :]
val_x = torch.from_numpy(scaler_x.fit_transform(val_set.iloc[:, [0, 1, 5]]).astype('float32'))
val_y = torch.from_numpy(scaler_y.fit_transform(val_set.iloc[:, [6, 8, 9, 11]]).astype('float32'))

t_set = torch.utils.data.TensorDataset(train_x, train_y)
train_loader = DataLoader(t_set, batch_size=256, shuffle=True, num_workers=8)
len(train_loader.sampler)

model = mdn_advance(train_x.shape[1], train_y.shape[1], 3, 256)
pi, mu, sigma = model(train_x[:64])

c = NLLLoss()
loss_2 = c(pi, mu, sigma, train_y[:64])

# select_idx = torch.argmax(pi, dim=-1)

# Advance Indexing
# mu_1 = mu[torch.arange(mu.shape[0]), select_idx, :]

y_ture = train_y[:64]

"""
    Sampling
"""
len_y = len(y_ture)
n_mix = pi.shape[-1]

select_idx = torch.multinomial(torch.exp(pi), num_samples=1, replacement=True).squeeze()
mu_selected = mu[torch.arange(mu.shape[0]), select_idx, :]
sigma_selected = sigma[torch.arange(sigma.shape[0]), select_idx, :]

samples = torch.normal(mean=mu_selected, std=sigma_selected)
samples = samples.detach().numpy()
y_ture = y_ture.detach().numpy()

relative_error = torch.div((torch.abs(samples - y_ture)), torch.abs(y_ture)).mean()

relative_error_1 = np.divide((np.abs(samples - y_ture)), np.abs(y_ture)).mean()
type(relative_error.item())


# samples = np.zeros((len_y, 1))
# to_choose_from = np.arange(n_mix)
# pi_numpy = torch.exp(pi).detach().numpy()
# select_idx = np.random.choice(n_mix, p=pi_numpy[0])


"""
    R-square Calculating
    Formular: 
        1 - (SSR(Sum of Squares of Residuals)/SST(Sum of Squares Total))
        SSR = Sum(Pow(y_ture - y_pred), 2)
        SST = Sum(Pow(y_ture - mean(y_ture), 2))    
"""
# SSR
ssr = torch.sum(torch.pow((y_ture - samples), 2), dim=-1)
# SST
sst = torch.sum(torch.pow((y_ture - torch.mean(y_ture)), 2))
# R2
r2 = 1 - (ssr / sst)
r2.mean()

r2_sk = r2_score(y_ture.detach().numpy(), samples.detach().numpy())
r2_sk
# tss = torch.sum(torch.pow((mu_1 - pred_y), 2), dim=1)
# mean_ture = torch.mean(pred_y, dim=1)
# r2 = 1 - tss / (mean_ture * 4)
# r2.mean()

view_pi = pi.detach().numpy()
view_exp_pi = torch.exp(pi).detach().numpy()
view_select_idx = select_idx.detach().numpy()
view_mu_1 = mu_1.detach().numpy()
view_mu = mu.detach().numpy()
view_pred_y = pred_y.detach().numpy()


"""
    negative-likelihood loss
    Formular:
        Mean(Sum(Log(pi) + Pow((y_ture-mu), 2) / 2 * sigma ** 2 + 1/2 * log(2 * pi * sigma ** 2)))
"""
# nllloss = torch.mean(torch.sum(pi + torch.pow((torch.unsqueeze(y_ture, dim=1) - mu), 2) / 2 * torch.pow(sigma, 2) + 1/2 * torch.log(2 * pi * torch.pow(sigma, 2))))
# a = torch.pow((torch.unsqueeze(y_ture, dim=1) - mu), 2)
# b = 2 * torch.pow(sigma, 2)
# a / b
# c = 1/2 * torch.log(2 * torch.unsqueeze(pi, dim=2) * torch.pow(sigma, 2))
# d = torch.unsqueeze(torch.exp(pi), dim=2)
# mix = Categorical(logits=pi)

# coll = [MultivariateNormal(loc=loc, scale_tril=scale) for loc, scale in zip(mu, sigma)]

# comp = Normal(mu, sigma)
# batch_size = pi.shape[0]
# n_gaussian = pi.shape[1]
# out_put_size = mu.shape[2]

# likelihood = coll.log_prob(train_y[:64].unsqueeze(1).expand(batch_size, n_gaussian, out_put_size))
# loss = cat.log_prob()
#
# a = cat.log_prob(torch.zeros_like(train_y[:64]))

# test_comp = MultivariateNormal(mu, covariance_matrix=torch.diag_embed(sigma))
# m_d = MixtureSameFamily(mix, test_comp)
# loss = -m_d.log_prob(train_y[:64]).mean()

# tra_test = mdnTraining()
# a, b, s_x, s_y = tra_test.load_data()
# len(a.sampler)
# len(b.sampler)
#
# data['M_total (M_E)'] = data['Mcore (M_J/10^3)'] + data['Menv (M_E)']
#
# test_a = data.iloc[:, [0, 1, 5]]
# test_b = data.iloc[:, [6, 8, 9, 11]]
#
# d = data.sample(frac=0.8).reset_index(drop=True)
# e = data.drop(d.index).sample(frac=0.5).reset_index(drop=True)





