import os
import pandas as pd
import torch
import numpy as np
from torch.distributions import Categorical, Normal, MixtureSameFamily, MultivariateNormal, Independent
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from Rock.Model.MDN import mdn, NLLLoss
from sklearn.metrics import r2_score
from Rock.Model.MDN_From_Kaggle import mdn as mdn_advance
from Rock.Train.TrainLoopOrigin import mdnTraining
import seaborn as sns
import matplotlib.pyplot as plt
import math


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

input_parameters = [
    'Mass (M_J)',
    'Radius (R_E)',
    'T_sur (K)',
]

output_parameters = [
    'M_total (M_E)',
    'T_int (K)',
    'P_CEB (Mbar)',
    'T_CEB (K)'
]

scaler_x = MinMaxScaler(feature_range=[0, 1])
scaler_y = MinMaxScaler(feature_range=[0, 1])

data_x_nor = pd.DataFrame(scaler_x.fit_transform(data.loc[:, input_parameters]))
data_y_nor = pd.DataFrame(scaler_y.fit_transform(data.loc[:, output_parameters]))

train_x = data_x_nor.sample(frac=0.9)
train_y = data_y_nor.iloc[train_x.index]

val_x = data_x_nor.drop(train_x.index).sample(frac=0.5)
val_y = data_y_nor.iloc[val_x.index]

test_x = data_x_nor.drop(train_x.index).drop(val_x.index)
test_y = data_y_nor.iloc[test_x.index]

train_x.reset_index(inplace=True, drop=True)
train_y.reset_index(inplace=True, drop=True)
val_x.reset_index(inplace=True, drop=True)
val_y.reset_index(inplace=True, drop=True)
test_x.reset_index(inplace=True, drop=True)
test_y.reset_index(inplace=True, drop=True)

train_x = torch.from_numpy(train_x.to_numpy()).double()
train_y = torch.from_numpy(train_y.to_numpy()).double()
val_x = torch.from_numpy(val_x.to_numpy()).double()
val_y = torch.from_numpy(val_y.to_numpy()).double()
test_x = torch.from_numpy(test_x.to_numpy()).double()
test_y = torch.from_numpy(test_y.to_numpy()).double()

data_x = data.loc[:, input_parameters]

a = data_x.iloc[int(len(data) * 0.9):, :]

# train_x = torch.from_numpy(scaler_x.fit_transform(train_set.loc[:, input_parameters])).double()
# train_y = torch.from_numpy(scaler_y.fit_transform(train_set.loc[:, output_parameters])).double()
#
# val_x = torch.from_numpy(scaler_x.fit_transform(val_set.loc[:, input_parameters])).double()
# val_y = torch.from_numpy(scaler_y.fit_transform(val_set.loc[:, output_parameters])).double()
#
#
#
# # Original Model
# train_set = data.iloc[:int(len(data) * 0.6), :]
# train_x = torch.from_numpy(scaler_x.fit_transform(train_set.iloc[:, [0, 1, 5]]).astype('float32'))
# train_y = torch.from_numpy(scaler_y.fit_transform(train_set.iloc[:, [6, 8, 9, 11]]).astype('float32'))
#
# val_set = data.iloc[int(len(data) * 0.6):int(len(data) * 0.8), :]
# val_x = torch.from_numpy(scaler_x.fit_transform(val_set.iloc[:, [0, 1, 5]]).astype('float32'))
# val_y = torch.from_numpy(scaler_y.fit_transform(val_set.iloc[:, [6, 8, 9, 11]]).astype('float32'))

t_set = torch.utils.data.TensorDataset(train_x, train_y)
train_loader = DataLoader(t_set, batch_size=256, shuffle=True, num_workers=8)
# len(train_loader.sampler)

model = mdn_advance(train_x.shape[1], train_y.shape[1], 3, 256)
pi, mu, sigma = model(train_x[:64])
y_ture = train_y[:64]
optimizer = eval("torch.optim.{}({}, lr={}, weight_decay={})".format('Adam', 'model.parameters()', '0.001', '0.01'))

sigma_exp = torch.exp(sigma)

a = torch.tensor(np.array([[0.1, 0.2], [0.3, 0.4], [-0.2, -0.5]]))
torch.nn.functional.elu(a) + 1 + (1e-6)

c = NLLLoss()
loss_2 = c(pi, mu, sigma, train_y[:64])
"""
    Calculating NLLLoss by probability density function
"""
y_ture_ex = y_ture.unsqueeze(1).expand_as(sigma)
# pdf = 1.0 / torch.sqrt(2 * torch.pi * (sigma ** 2)) * torch.exp(-0.5 * ((y_ture_ex - mu) / sigma) ** 2)

scalar = 1.0 / math.sqrt(2 * math.pi)   # 不使用torch.sqrt(2 * torch.pi * (sigma ** 2))的原因是torch的sqrt是对张量中的数据逐个开根
ret = scalar * torch.exp(-0.5 * ((y_ture_ex - mu) / sigma) ** 2) / sigma
ret_mul = torch.prod(ret, 2)

pdf = pi * ret_mul
nll = -torch.log(torch.sum(pdf, dim=1))
nll.mean()
ret_np = ret.detach().numpy()

a_np = np.array([[1, 2], [3, 4], [5, 6], [3, 1]])
np.prod(a_np, 1)
"""
    Other Sampling Function
"""
# Change Shape (batch size, n_gaussian, output_size) -> (batch size, output_size) for every gaussian
cat = Categorical(logits=pi)
cat_sample = cat.sample()
changed_mus = mu[torch.arange(mu.shape[0]), cat_sample, :]
changed_sigmas = sigma[torch.arange(sigma.shape[0]), cat_sample, :]
coll = [Normal(loc=m, scale=s) for m, s in zip(changed_mus, changed_sigmas)]

mixture = MixtureSameFamily(cat, coll)
mixture_idx = cat.sample()
# samples = [coll[i].sample() for i in mixture_idx]
s = coll[0].rsample(sample_shape=(1, 4))

np_mu = mu.detach().numpy().reshape((64, 12))


"""
    Construct Mixture Distribution Model
"""
len_y = len(y_ture)
n_mix = pi.shape[-1]

select_idx = torch.multinomial(torch.exp(pi), num_samples=1, replacement=True).squeeze()
mu_selected = mu[torch.arange(mu.shape[0]), select_idx, :]
sigma_selected = sigma[torch.arange(sigma.shape[0]), select_idx, :]
pdf = Normal(loc=mu_selected, scale=sigma_selected)
pdf_mul = MultivariateNormal(loc=mu_selected, scale_tril=sigma_selected)

# Sampling
samples = pdf.sample()
samples_1 = torch.normal(mean=mu_selected, std=sigma_selected)

r2_1 = r2_score(y_ture, samples)
r2_2 = r2_score(y_ture, samples_1.detach().numpy())
# Loss
log_prob = pdf.log_prob(y_ture)
log_prob = log_prob.negative()
log_prob = log_prob.mean()

# samples = torch.normal(mean=mu_selected, std=sigma_selected)
samples = samples.detach().numpy()
y_ture = y_ture.detach().numpy()

relative_error = torch.div((torch.abs(samples - y_ture)), torch.abs(y_ture)).mean()

relative_error_1 = np.divide((np.abs(samples - y_ture)), np.abs(y_ture)).mean()
type(relative_error.item())

cat = Categorical(logits=pi)
for m, s in zip(mu, sigma):
    print(m, s)
    break

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

test_train = mdnTraining(learning_rate=0.001984, batch_size=512, hidden_size=512, n_gaussian=20, is_gpu=True, epoch=200, if_shuffle=True, is_normal=True)


file_path = 'D:\\Resource\\MRCK_2\\'
nowater_data_frames = []

file_nowater = os.listdir(file_path + 'nowater')
for f in file_nowater:
    tmp = pd.read_table(file_path + '\\' + 'nowater\\' + f, delimiter="\s+", header=None)
    nowater_data_frames.append(tmp)

df_nowater = pd.concat(nowater_data_frames)


file_water = os.listdir(file_path + 'water')
water_data_frames = []
for f in file_water:
    tmp = pd.read_table(file_path + '\\' + 'water\\' + f, delimiter="\s+", header=None)
    water_data_frames.append(tmp)

df_water = pd.concat(water_data_frames)


# combine merged nowater and water data
df_all = pd.concat([df_nowater, df_water])

# rename columns
df_all.columns = ['Mass', 'Radius', 'WMF',
                  'CMF', 'WRF', 'CRF', 'PRS_WMB',
                  'TEP_WMB', 'PRS_CMB', 'TEP_CMB', 'PRS_CEN', 'TEP_CEN',
                  'k2', 'FeMg_mantle', 'SiMg_mantle', 'FeO_mantle']

# reset index
df_all = df_all.reset_index(drop=True)

df_all = df_all.astype(float)

CaMg = 0.06
AlMg = 0.08
wt_frac_S_core = 0.0695     # by mass

mFe = 55.845
mMg = 24.306
mSi = 28.0867
mO = 15.9994
mS = 32.0650
mCa = 40.078
mAl = 26.981

# you can check the FeO_mantle results from the mantle molar ratios FeMg, SiMg, CaMg, AlMg
# The results should be same as the column FeO_mantle
reduced_mantle_mass = df_all['FeMg_mantle'] * (mFe+mO) + df_all['SiMg_mantle'] * (mSi+2.0*mO) + CaMg * (mCa+mO) + AlMg * (mAl+1.5*mO) + (mMg+mO)
FeO_mantle_cal = df_all['FeMg_mantle'] * (mFe+mO) / reduced_mantle_mass

# number of Fe atoms in the core
nFe_core = df_all['CMF'] * (1.0 - wt_frac_S_core) / mFe

# number of Fe atoms in the mantle
nFe_mantle = (1.0 - df_all['CMF'] - df_all['WMF']) * df_all['FeO_mantle'] / (mFe + mO)

# number of Mg atoms in the mantle
nMg_mantle = nFe_mantle / df_all['FeMg_mantle']

# bulk FeMg
FeMg = (nFe_core + nFe_mantle) / nMg_mantle
df_all['FeMg'] = FeMg

# bulk SiMg: there is no Si & Mg in the core
df_all['SiMg'] = df_all['SiMg_mantle']

df_all = df_all[df_all['FeMg']<=40]

df_all["Fe_(Mg+Si)"] = df_all["FeMg"]/(df_all["SiMg"] + 1)

df_all["MRF"] = 1 - df_all["WRF"] - df_all["CRF"]

input_parameters = [
    'Mass',
    'Radius',
    'Fe_(Mg+Si)',
    'k2',
]


output_parameters = [
    'WRF',
    'MRF',
    'CRF',
    'CMF',
    'PRS_CMB',
    'TEP_CMB',
]

X = df_all[input_parameters]
x = df_all.loc[:, input_parameters]

y = df_all.loc[:, output_parameters]

