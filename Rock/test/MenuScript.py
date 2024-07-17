import os

import joblib
import pandas as pd
import torch
import numpy as np
import tensorflow as tf
from matplotlib.ticker import AutoMinorLocator
from sklearn.model_selection import train_test_split
from tensorflow import keras
from torch.distributions import Categorical, Normal, MixtureSameFamily, MultivariateNormal, Independent
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import mdn
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tortreinador.utils.View import init_weights
from Rock.Model.MDN_by_Pytorch import mdn as mdn_advance, Mixture, NLLLoss, NLLLoss_Version_2
from Rock.Train.TrainLoopOrigin import mdnTraining
from Rock.Model.EnsembleMDN import EnsembleMDN
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


def test_optim(b: torch.nn.Module):
    print("OK")


t_set = torch.utils.data.TensorDataset(train_x, train_y)
train_loader = DataLoader(t_set, batch_size=256, shuffle=True, num_workers=8)
# len(train_loader.sampler)

model = mdn_advance(3, 4, 3, 256)
# model.to('cuda')
# train_x = train_x.to('cuda')

pi, mu, sigma = model(train_x[:64])

criterion_straight = NLLLoss()
criterion_sampling = NLLLoss_Version_2()

for i in range(0, 64 * 5, 64):
    pi, mu, sigma = model(train_x[i: i + 64])
    y_ture = train_y[i: i + 64]
    # y_ture = y_ture.to('cuda')
    mix = Mixture()

    pdf = mix(pi, mu, sigma)

    loss_1 = criterion_straight(pi, mu, sigma, y_ture)
    loss_2 = criterion_sampling(pdf, y_ture)

    print("Loss Calculated Straightly: {}, Loss Calculated by Sampling: {}".format(loss_1, loss_2))



y_pred = pdf.sample()

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

df_sample = df_all.sample(frac=0.1, replace=False)
df_sample.to_csv("D:\\Study\\澳科大\\Final_2023\\Data Mining\\sample.csv", index=False, header=True)
"""
    r2 score implementation by pytorch
"""
a = np.random.rand(2, 3)
b = np.random.rand(2, 3)

a_t = torch.from_numpy(a).to('cuda')
b_t = torch.from_numpy(b).to('cuda')

torch.sum(((a_t - b_t) ** 2), dim=0, dtype=torch.float64)
b_t.device == a_t.device
ss_tot = torch.sum(((y_ture - torch.mean(y_ture, dim=0)) ** 2), dim=0, dtype=torch.float64)
ss_res = torch.sum(((y_ture - y_pred) ** 2), dim=0, dtype=torch.float64)
r2 = 1 - ss_res / ss_tot

print("pytorch r2 score : {}".format(torch.mean(r2)))
print("sklean r2 score : {}".format(r2_score(y_ture.cpu(), y_pred.cpu())))


torch.mean(r2)
# test_optim(torch.optim.Adam(model.parameters()))

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

trainer = train.TorchTrainer()

t_loader, v_loader, test_x, test_y, s_x, s_y = trainer.load_data(data=data, input_parameters=input_parameters, output_parameters=output_parameters,
                           if_normal=True, if_shuffle=True)


model = mdn(len(input_parameters), len(output_parameters), 10, 256)
criterion = NLLLoss()
pdf = Mixture()
optim = torch.optim.Adam(model.parameters(), lr=0.0001984)


t_l, v_l, val_r2, train_r2, mse = trainer.fit_for_MDN(t_loader, v_loader, criterion, model=model, mixture=pdf, model_save_path='D:\\Resource\\MDN\\', optim=optim, best_r2=0.5, xavier_init=False)


from numpy.random import seed
seed(123)

a = np.array([1, 2, 3, 5, 6])

X = data[input_parameters]

y = data.loc[:, output_parameters]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y, test_size=0.1)

c = np.random.choice(a, 3)

import matplotlib.pyplot as plt
import numpy as np

# 使用交互式后端
plt.ion()

# 创建一个初始的空图表
fig, ax = plt.subplots()
line, = ax.plot([], [])  # 创建一个空的曲线

# 更新图表的函数
def update_plot(x, y):
    line.set_xdata(x)
    line.set_ydata(y)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()

# 模拟实时更新数据并绘制图表
x = np.linspace(0, 10, 100)
for i in range(100):
    y = np.sin(x - i)  # 模拟新的数据
    update_plot(x, y)  # 更新图表

import torch.nn as nn

# 声明model对象，此时input size为3
model = mdn_advance(3, 4, 3, 256)

# 声明一个input size为2的Linear对象
linear_2 = nn.Linear(in_features=2, out_features=256, bias=True)

# 在root_layer中替换
model.root_layer[0] = linear_2

for name, parameters in model.named_parameters():
    print(name, ':', parameters.size())

w_list = [1, 2, 3, 4, 5, 6, 7]
window = []
stride = 2
windows_size = 3

for i in range(0, len(w_list) - windows_size + stride, stride):
    curr_size = len(w_list) - i
    if curr_size < windows_size:
        tmp = np.array(w_list[i: i + curr_size])

    else:
        tmp = np.array(w_list[i: i + windows_size])

    mu = np.mean(tmp)
    sig = np.std(tmp)
    z_score = np.abs((tmp - mu) / sig)

    print(z_score)

    window.append(tmp)

w_test = np.array([])

i = 2
g = [2, 4]
h = [5]

type(np.array(g)[0].item())

"""
    V3 Testing
"""

df_chunk_0 = pd.read_parquet("D:\\Resource\\rockyExoplanetV3\\data_chunk_0.parquet")
df_chunk_1 = pd.read_parquet("D:\\Resource\\rockyExoplanetV3\\data_chunk_1.parquet")

df_all = pd.concat([df_chunk_0, df_chunk_1])
input_parameters = [
    'Mass',
    'Radius',
    'FeMg',
    'SiMg',
    'Mass_Noise',
    'Radius_Noise',
    'FeMg_Noise',
    'SiMg_Noise'
]

output_parameters = [
    'WRF',
    'MRF',
    'CRF',
    'WMF',
    'CMF',
    'CPS',
    'CTP',
    'k2'
]

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(scaler_x.fit_transform(df_all[input_parameters[:4]].to_numpy()), scaler_y.fit_transform(df_all[output_parameters].to_numpy()), test_size=0.1, random_state=9887)

OUTPUT_DIMS = len(output_parameters)
N_MIXES = 20

m_2 = keras.models.load_model('D:\\Resource\\MDN\\rockyExoplanetV3\\model.h5', custom_objects={'MDN': mdn.MDN, 'mdn_loss_func': mdn.get_mixture_loss_func(OUTPUT_DIMS, N_MIXES)})

# Set the random seed for reproducibility
np.random.seed(125)


# Define the function to generate samples and plot histogram
def generate_samples_and_plot(mean, err, title, ax):
    samples = np.random.normal(loc=mean, scale=err, size=1000)
    ax.hist(samples, bins=50, color='steelblue', edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.axvline(mean, color='red', linestyle='dashed', linewidth=2)  # Mark the mean value
    ax.fill_between([mean - err, mean + err], [0, 0], [100, 100], alpha=0.2, color='green')  # Highlight the error range
    return samples


def calculate_GMM(p, m, s, y_label):
    if len(y_label.shape) == 1:
        y_label = y_label.reshape(-1, 1)

    y_label_ = y_label[:, np.newaxis, np.newaxis, :]

    mu_sub_T = np.transpose(m, (0, 2, 1))
    sigma_sub_T = np.transpose(s, (0, 2, 1))

    exponent = np.exp(
        -1 / 2 * np.square(np.transpose((y_label_ - mu_sub_T), (1, 2, 0, 3)) / sigma_sub_T[:, :, np.newaxis, :]))
    factors = 1 / math.sqrt(2 * math.pi) / sigma_sub_T[:, :, np.newaxis, :]

    # Shape (number of data, number of y_label, types) e.g.(1000, 10, 8)
    GMM_PDF = np.sum(p[:, np.newaxis, np.newaxis, :] * factors * exponent, axis=-1).transpose((0, -1, 1))
    # GMM_PDF = GMM_PDF.reshape(GMM_PDF.shape[0] * GMM_PDF.shape[1], GMM_PDF.shape[-1]).transpose((-1, 0))
    # f = e.transpose((-1, 0))
    return GMM_PDF


# 设置随机数种子以便重复实验
np.random.seed(125)


# 从kepler 78b的均值和误差范围中生成1000个随机样本
mass_mean = 1.77
mass_err = 0.25
mass_err_ori = 0
mass_samples = np.random.normal(loc=mass_mean, scale=mass_err, size=1000)
mass_samples_ori = np.random.normal(loc=mass_mean, scale=0, size=1000)

radius_mean = 1.228
radius_err = 0.019
radius_err_ori = 0
radius_samples = np.random.normal(loc=radius_mean, scale=radius_err, size=1000)
radius_samples_ori = np.random.normal(loc=radius_mean, scale=0, size=1000)

femg_mean = 0.813
femg_err = 0.248
femg_err_ori = 0
femg_samples = np.random.normal(loc=femg_mean, scale=femg_err, size=1000)
femg_samples_ori = np.random.normal(loc=femg_mean, scale=0, size=1000)

simg_mean = 0.933
simg_err = 0.281
simg_err_ori = 0
simg_samples = np.random.normal(loc=simg_mean, scale=simg_err, size=1000)
simg_samples_ori = np.random.normal(loc=simg_mean, scale=0, size=1000)

# Kepler 10b
# 设置随机数种子以便重复实验
np.random.seed(125)

# 从kepler 78b的均值和误差范围中生成1000个随机样本
mass_mean = 3.72
mass_err = 0.42
mass_samples = np.random.normal(loc=mass_mean, scale=mass_err, size=1000)
mass_samples_ori = np.random.normal(loc=mass_mean, scale=mass_err_ori, size=1000)

radius_mean = 1.47
radius_err = 0.03
radius_samples = np.random.normal(loc=radius_mean, scale=radius_err, size=1000)
radius_samples_ori = np.random.normal(loc=radius_mean, scale=radius_err_ori, size=1000)

femg_mean = 0.589
femg_err = 0.073
femg_samples = np.random.normal(loc=femg_mean, scale=femg_err, size=1000)
femg_samples_ori = np.random.normal(loc=femg_mean, scale=femg_err_ori, size=1000)

simg_mean = 0.661
simg_err = 0.098
simg_samples = np.random.normal(loc=simg_mean, scale=simg_err, size=1000)
simg_samples_ori = np.random.normal(loc=simg_mean, scale=simg_err_ori, size=1000)


# Combine all samples into an input matrix X
X = np.stack([mass_samples, radius_samples, femg_samples, simg_samples], axis=1)
X_ori = np.stack([mass_samples_ori, radius_samples_ori, femg_samples_ori, simg_samples_ori], axis=1)


pred = m_2.predict(scaler_x.transform(X))

# Extract the Gaussian mixture parameters
mus = pred[0, :N_MIXES*OUTPUT_DIMS]
sigs = pred[0, N_MIXES*OUTPUT_DIMS:2*N_MIXES*OUTPUT_DIMS]
pis = mdn.softmax(pred[:, -N_MIXES:])

# Prepare for the sampling
y_label = np.arange(0, 1, 0.001).reshape(-1, 1)

# The mu and sigma of first column
mus_ = mus[0::OUTPUT_DIMS]
sigs_ = sigs[0::OUTPUT_DIMS]
factors = 1 / math.sqrt(2 * math.pi) / sigs_
exponent = np.exp(-0.5 * ((y_label - mus_) / sigs_)**2)
GMM_PDF = np.sum(pis[0] * factors * exponent, axis=1)   # Using the first input to test, so pis[0]
GMM_PDF = GMM_PDF.reshape(-1, 1)

# Define the density storage structure for the outputs
density_x = {param: [] for param in output_parameters}
density_y = {param: [] for param in output_parameters}
pdf_nonzero = np.count_nonzero(GMM_PDF)

"""
    Testing found that the below codes are unnecessary
"""
if GMM_PDF.sum() == 0:
    index = np.random.choice(y_label[:, 0], size=20, replace=True)
else:
    size = min(20, pdf_nonzero)  # 确保采样大小不超过非零元素数量
    index = np.random.choice(y_label[:, 0], size=size, replace=False, p=GMM_PDF/GMM_PDF.sum())
bins = np.concatenate(([y_label[0, 0]], y_label[:, 0]))
indices = np.searchsorted(bins, index) - 1
density_x['WRF'] = np.concatenate([density_x['WRF'], y_label[:, 0][indices]])
density_y['WRF'] = np.concatenate([density_y['WRF'], GMM_PDF[indices]])

mu = np.apply_along_axis((lambda a: a[:N_MIXES * OUTPUT_DIMS]), 1, pred)
sig = np.apply_along_axis((lambda a: a[N_MIXES * OUTPUT_DIMS:2 * N_MIXES * OUTPUT_DIMS]), 1, pred)
pis_t = mdn.softmax(pred[:, -N_MIXES:])
mu = mu.reshape(mu.shape[0], N_MIXES, int(mu.shape[1] / N_MIXES))
sig = sig.reshape(sig.shape[0], N_MIXES, int(sig.shape[1] / N_MIXES))


"""
    Attn-EnsembleMDN Testing
"""
model = EnsembleMDN(int(len(input_parameters) / 2), len(output_parameters), 10, 256, kernel_size=3)
init_weights(model)
model = nn.DataParallel(model)
model.load_state_dict(torch.load("D:\\Resource\\MDN\\rockyExoplanetV3\\NoiseADD\\best_model_colabe.pth"))

m_y = joblib.load("D:\\Resource\\MDN\\rockyExoplanetV3\\NoiseADD\\testData\\MDN_v3_yscaler_20240630.save")
m_x = joblib.load("D:\\Resource\\MDN\\rockyExoplanetV3\\NoiseADD\\testData\\MDN_v3_xscaler_20240630.save")
X_com = np.concatenate((X_ori, X), axis=1)
X_scaled = m_x.transform(X_com)
tensor_x = torch.from_numpy(X_scaled)
x_o, x_n = tensor_x.chunk(2, dim=1)

pred = model(x_o, x_n)

mus_ = pred[1][0, :].ravel().detach().cpu().numpy()
sigs_ = pred[2][0, :].ravel().detach().cpu().numpy()
pis = pred[0][0, :].detach().cpu().numpy()
mus_ = mus_[0::OUTPUT_DIMS]
sigs_ = sigs_[0::OUTPUT_DIMS]
factors = 1 / math.sqrt(2 * math.pi) / sigs_
exponent = np.exp(-0.5 * ((y_label - mus_) / sigs_)**2)
GMM_PDF = np.sum(np.exp(pis) * factors * exponent, axis=1)   # Using the first input to test, so pis[0]
GMM_PDF = GMM_PDF.reshape(-1, 1)

y_label_ = y_label[:, np.newaxis, np.newaxis, :]

mu_sub_T = np.transpose(pred[1].detach().cpu().numpy(), (0, 2, 1))
sigma_sub_T = np.transpose(pred[2].detach().cpu().numpy(), (0, 2, 1))

exponent_t = np.exp(
    -1 / 2 * np.square(np.transpose((y_label_ - mu_sub_T), (1, 2, 0, 3)) / sigma_sub_T[:, :, np.newaxis, :]))
factors_t = 1 / math.sqrt(2 * math.pi) / sigma_sub_T[:, :, np.newaxis, :]

GMM_PDF_Tor = np.sum(pred[0].detach().cpu().numpy()[:, np.newaxis, np.newaxis, :] * factors_t * exponent_t, axis=-1).transpose((0, -1, 1))

factors_t_ = factors_t[0, :]
factors_t_ = factors_t_.reshape((factors_t_.shape[0] * factors_t_.shape[-1], ))

exponent_t_ = exponent_t[0, 0, :]
pi_softmax = torch.exp(pred[0]).detach().cpu().numpy()[0]
pi_softmax.sum()

GMM_PDF_Tor = calculate_GMM(torch.exp(pred[0]).detach().cpu().numpy(), pred[1].detach().cpu().numpy(), pred[2].detach().cpu().numpy(), y_label)
bins = np.concatenate(([y_label[0, 0]], y_label[:, 0]))
density_x = {param: [] for param in output_parameters}
density_y = {param: [] for param in output_parameters}
# Loop start here
for out_param in range(len(output_parameters)):
    GMM_PDF_Tor_sub = GMM_PDF_Tor[:, :, out_param]

    GMM_PDF_Tor_sub_sum = GMM_PDF_Tor_sub.sum(axis=1)[:, np.newaxis]

    prob = np.divide(GMM_PDF_Tor_sub, GMM_PDF_Tor_sub_sum)
    non_zero = np.count_nonzero(GMM_PDF_Tor_sub, axis=1)

    cond = np.where(GMM_PDF_Tor_sub_sum == 0, True, False)
    p = np.where(cond, 1 / len(y_label), prob)
    size_cond = np.where((non_zero < 20) & (non_zero > 0), non_zero, np.where(non_zero == 0, 20, 20))

    idx = [np.random.choice(y_label[:, 0], size=size_cond[i], replace=cond[i, 0], p=p[i, :]) for i in range(len(GMM_PDF_Tor_sub))]
    indices = [np.searchsorted(bins, idx[i]) - 1 for i in range(len(idx))]
    flattened_indices = np.concatenate(indices)
    density_x[output_parameters[out_param]] = np.concatenate([density_x[output_parameters[out_param]], y_label[:, 0][flattened_indices].ravel()])

# density_y['WRF'] = np.concatenate([density_y['WRF'], GMM_PDF_Tor_sub[indices[1]]])      # Waiting for update

"""
    Tensorflow plotting test
"""
# Plot settings for MDN and MCMC results
x_max = [0.15, 1, 1, 0.1, 1, 2500, 6000, 1.5]
x_locator = [0.05, 0.2, 0.2, 0.02, 0.2, 500, 2000, 0.5]
colors = ["steelblue"] * len(output_parameters)


# Convert density dictionaries into DataFrames for easier manipulation
df_density_x = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in density_x.items()]))
# df_density_y = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in density_y.items()]))


# Ensure that the original_x DataFrame has the same index as df_density_y after dropping NaNs
original_x = m_y.inverse_transform(df_density_x)

df_k_samples = pd.DataFrame(original_x, columns=output_parameters)

df_k = pd.read_csv("D:\\Resource\\rockyExoplanetV3\\Kepler10b_feo.csv")
df_k["MRF"] = 1 - df_k["WRF"] - df_k["CRF"]

# Plotting comparison histograms for MCMC and MDN results
# Assuming output_parameters, x_max, x_locator are defined as before
# Assuming df_k is the DataFrame containing the MCMC results, loaded as before

x_labels = ['Water Radial Fraction', 'Mantle Radial Fraction', 'Core Radial Fraction',
            'Water Mass Fraction', 'Core Mass Fraction', 'CMB Pressure (TPa)',
            'CMB Temperature (10^3K)', 'Love number k2']

fig, axs = plt.subplots(2, 4, figsize=(16, 8))
axs = axs.flatten()

for i, param in enumerate(output_parameters):
    ax = axs[i]
    # plot MCMC results
    x = df_k[param]
    ax.hist(x, density=True, bins=15, histtype='step', color='#4682b4', linewidth=2, label='MCMC inference')
    median = np.median(x)
    ax.axvline(median, color='#4682b4', linestyle='--', lw=2)

    # plot MDN results
    params_x = original_x[:, i]
    params_x = params_x[~np.isnan(params_x)]
    counts, bins, _ = ax.hist(params_x, density=True, bins=15, histtype='step', color='red', linewidth=2,
                              label='MDN inference')
    median = np.median(params_x)
    ax.axvline(median, color='r', linestyle='--', lw=2)

    # Set x-axis label from the provided list
    ax.set_xlabel(x_labels[i])

    ax.set_xlim(0, x_max[i])
    ax.set_yticks([])
    ax.xaxis.set_major_locator(plt.MultipleLocator(x_locator[i]))
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    # Add legend to the first subplot
    if i == 0:
        ax.legend()

# Adjust the layout
plt.tight_layout()

# Display the plot
plt.show()


