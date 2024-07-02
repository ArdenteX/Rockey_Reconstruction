import os
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.distributions import Categorical, Normal, MixtureSameFamily, MultivariateNormal, Independent
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from Rock.Model.MDN import mdn, NLLLoss
from sklearn.metrics import r2_score
from Rock.Model.MDN_by_Pytorch import mdn as mdn_advance, Mixture, NLLLoss, NLLLoss_Version_2
from Rock.Train.TrainLoopOrigin import mdnTraining
import seaborn as sns
import matplotlib.pyplot as plt
import math
from tortreinador import train
from tortreinador.models.MDN import mdn, Mixture, NLLLoss


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



