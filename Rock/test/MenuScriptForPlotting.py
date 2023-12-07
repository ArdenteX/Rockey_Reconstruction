import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from Rock.Model.MDN_by_Pytorch import mdn, Mixture
from Rock.Utils.View import init_weights
from sklearn.metrics import r2_score
import math
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

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
OUTPUT_DIMS = len(output_parameters)
scaler_x = MinMaxScaler(feature_range=[0, 1])
scaler_y = MinMaxScaler(feature_range=[0, 1])

data_x_nor = pd.DataFrame(scaler_x.fit_transform(data.loc[:, input_parameters]))
data_y_nor = pd.DataFrame(scaler_y.fit_transform(data.loc[:, output_parameters]))

input_size = 3
output_size = 4
n_gaussian = 10
hidden_size = 128

mixture = Mixture()

model = mdn(input_size, output_size, n_gaussian, hidden_size)
init_weights(model)
model = nn.DataParallel(model)
model.to("cuda")
model.load_state_dict(torch.load("D:\\Resource\\MDN\\model_best_mdn_normalization_4060Ti_0.97_0.0008.pth"))

test_x = np.load("D:\\Resource\\MDN\\TestData\\test_x.npy")
test_y = np.load("D:\\Resource\\MDN\\TestData\\test_y.npy")
y_label = np.arange(0, 1, 0.001).reshape(-1, 1)

model.eval()
# Predict
pi, mu, sigma = model(torch.from_numpy(test_x))

# Construct Distribution Function
normal, pi_idx, mu_selected, sigma_selected = mixture(pi, mu, sigma, 'test')

# Sample from Distribution Function
samples = normal.sample()

# Calculate R2 Score
print(r2_score(test_y, samples.cpu().numpy()))

test_y_inverse = scaler_y.inverse_transform(test_y)

y_max = max(test_y_inverse[:, 0])
y_min = min(test_y_inverse[:, 0])

pi = pi.detach().cpu().numpy()
mu_sub = mu.detach().cpu().numpy()[:, :, 0]
sigma_sub = sigma.detach().cpu().numpy()[:, :, 0]

test_divide = y_label - mu_sub[0]

test_y_label = y_label[:, np.newaxis]

sigma_sub_1 = sigma_sub[0, :]
factors_test = 1 / math.sqrt(2 * math.pi) / sigma_sub_1

mu_sub_1 = mu_sub[0, :]
miner_test = y_label - mu_sub_1

y_label_test = y_label[:, np.newaxis, :]
miner_test_2 = y_label_test - mu_sub

a = miner_test_2[:, 0]

subtract_test = miner_test - sigma_sub_1
subtract_test_2 = miner_test_2 - sigma_sub

square_test = np.square(subtract_test)
square_test_2 = np.square(subtract_test_2)

factors_1 = 1 / math.sqrt(2 * math.pi) / sigma_sub_1
exponent_1 = np.exp(-1 / 2 * np.square((y_label - mu_sub_1) / sigma_sub_1))

GMM_PDF = np.sum(pi[0] * factors_1 * exponent_1, axis=1)  # 对多个高斯分布求和

multiply_test = pi[0] * factors_1 * exponent_1

sum_test_1 = np.sum(multiply_test, axis=1)

# New Test
scalar = 1.0 / math.sqrt(2 * math.pi)
test_y_sub = test_y[:, 0]
y_true_expanded = test_y_sub[:, np.newaxis]
# y_true_expanded = np.tile(y_true_expanded, sigma_sub.shape)
ret = scalar * np.exp(-0.5 * ((y_true_expanded - mu_sub) / sigma_sub) ** 2) / sigma_sub

prob = np.sum(pi * ret, axis=-1)


# MM = MinMaxScaler()
#
# GMM_PDF_scaled_test_1 = MM.fit_transform(GMM_PDF.reshape(-1, 1))
#
# reshape_test_1 = GMM_PDF_scaled_test_1.T
# reshape_test_2 = GMM_PDF_scaled_test_2.T
#
# cmap = plt.cm.hot_r
# norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(2, 2, 0 + 1)
#
# ax.imshow(
#         GMM_PDF_scaled_test_2,
#         cmap=cmap,
#         norm=norm,
#         origin='lower',
#         extent=(0, 1, 0, y_max)
#     )
"""
    TODO: 修改老代码
"""
# ax.plot([y_min, y_max], [y_min, y_max], c='cornflowerblue', ls='--')
# plt.axis('square')
# ax.set_xlim(y_min, y_max)
# ax.set_ylim(y_min, y_max)
# ax.set_title('M_total (M_E)')
# plt.show()

test_y_sub = test_y[:, 0]

pis = pi.detach().cpu().numpy()
mus = mu.detach().cpu().numpy()
sigmas = sigma.detach().cpu().numpy()

mus = mus.reshape((mus.shape[0], mus.shape[1] * mus.shape[-1]))
sigmas = sigmas.reshape((sigmas.shape[0], sigmas.shape[1] * sigmas.shape[-1]))

y_label = np.arange(0,1,0.001).reshape(-1,1)
cmap = plt.cm.hot_r
norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
fig = plt.figure(figsize=(10, 10))

for img_num in range(OUTPUT_DIMS):
    ax = fig.add_subplot(2, 2, img_num+1)
    y_max = max(test_y_inverse[:, img_num])
    y_min = min(test_y_inverse[:, img_num])
    for c in range(len(test_y_inverse)):
        for m in range(OUTPUT_DIMS):
            locals()['mus'+str(m)] = []
            locals()['sigs'+str(m)] = []
            for n in range(10):
                locals()['mus'+str(m)].append(mus[c][n*OUTPUT_DIMS + m])
                locals()['sigs'+str(m)].append(sigmas[c][n*OUTPUT_DIMS + m])
        i = img_num
        mus_ = np.array(locals()['mus'+str(i)])
        sigs_ = np.array(locals()['sigs'+str(i)])
        factors = 1 / math.sqrt(2*math.pi) / sigs_
        exponent = np.exp(-1/2*np.square((y_label-mus_)/sigs_))
        GMM_PDF = np.sum(pi[c]*factors*exponent, axis=1) # 对多个高斯分布求和
        MM = MinMaxScaler()
        GMM_PDF_scaled = MM.fit_transform(GMM_PDF.reshape(-1, 1))
        tx, ty = [test_y_inverse[c, i], y_min]
        ax.imshow(
            GMM_PDF_scaled.reshape(-1, 1),
            cmap=cmap,
            norm=norm,
            origin='lower',
            extent=(tx, tx + 0.0001, ty, y_max)
        )
    ax.plot([y_min, y_max], [y_min, y_max], c='cornflowerblue', ls='--')
    plt.axis('square')
    ax.set_xlim(y_min, y_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(output_parameters[img_num])

plt.show()

sample = samples.cpu().numpy()
fig = plt.figure(figsize=(10, 10))
"""
    Plot Scatter
"""
for i in range(OUTPUT_DIMS):
    x = sample[:, i]
    y = test_y[:, i]

    y_max_2 = max(y)
    y_min_2 = min(y)

    ax = fig.add_subplot(2, 2, i + 1)

    ax.scatter(x, y, alpha=0.7, s=10)

    ax.plot([y_min_2, y_max_2], [y_min_2, y_max_2], c='cornflowerblue', ls='--')
    plt.axis('equal')
    ax.set_xlim(y_min_2, y_max_2)
    ax.set_ylim(y_min_2, y_max_2)
    ax.set_title(output_parameters[i])

plt.show()


pis = pi[torch.arange(pi.shape[0]), pi_idx].detach().cpu().numpy()

mus = mu_selected.detach().cpu().numpy()
sigmas = sigma_selected.detach().cpu().numpy()


y_label = np.arange(0, 1, 0.001).reshape(-1, 1)
test_y_inverse = scaler_y.inverse_transform(test_y)

cmap = plt.cm.hot_r
norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
fig = plt.figure(figsize=(10, 10))

for o in range(OUTPUT_DIMS):
    # Fixed Begin
    factors = 1 / math.sqrt(2 * math.pi) / sigmas[:, o]
    exponent = np.exp(-1 / 2 * np.square((y_label - mus[:, o]) / sigmas[:, o]))
    GMM_PDF = pis * factors * exponent
    GMM_PDF_reshape = GMM_PDF.T
    MM = MinMaxScaler(feature_range=[-1, 1])
    GMM_PDF_scaled = MM.fit_transform(GMM_PDF_reshape)
    # Fix End

    y = test_y_inverse[:, o]
    y_max_2 = max(y)
    y_min_2 = min(y)
    ax = fig.add_subplot(2, 2, o + 1)
    for i in range(len(test_y_inverse)):
        tx = test_y_inverse[i, o]
        ax.imshow(
            GMM_PDF_scaled[i, :].reshape(-1, 1),
            cmap=cmap,
            norm=norm,
            origin='lower',
            extent=(tx, tx + 5, y_min_2, y_max_2)
        )
    ax.plot([y_min_2, y_max_2], [y_min_2, y_max_2], c='cornflowerblue', ls='--')

    plt.axis('square')
    ax.set_xlim(y_min_2, y_max_2)
    ax.set_ylim(y_min_2, y_max_2)
    ax.set_title(output_parameters[o])

plt.show()

# 创建一些样本数据
x = sample[:, 0]
y = test_y[:, 0]

# 绘制双变量的核密度估计曲线图
sns.kdeplot(x, y, shade=True, linewidth=2.5, label='True dist')

# 设置图表的标题和轴标签
plt.title('Kernel Density Estimation')
plt.xlabel('X')
plt.ylabel('Y')

plt.show()

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

input_size = 4
output_size = 6
n_gaussian = 10
hidden_size = 256

mixture = Mixture()

model = mdn(input_size, output_size, n_gaussian, hidden_size)
init_weights(model)
model = nn.DataParallel(model)
model.to("cuda")
model.load_state_dict(torch.load("D:\\Resource\\MDN\\model_best_mdn_0.9918_-30_0.0004.pth"))

test_x = np.load("D:\\Resource\\MDN\\TestDataMRCk2\\test_x.npy")
test_y = np.load("D:\\Resource\\MDN\\TestDataMRCk2\\test_y.npy")
y_label = np.arange(0, 1, 0.001).reshape(-1, 1)

model.eval()
# Predict
pi, mu, sigma = model(torch.from_numpy(test_x))

# Construct Distribution Function
normal, pi_idx, mu_selected, sigma_selected = mixture(pi, mu, sigma, 'test')
samples = normal.sample()
sample = samples.cpu().numpy()


fig = plt.figure(figsize=(10, 10))
"""
    Plot Scatter
"""
for i in range(len(output_parameters)):
    x = sample[:, i]
    y = test_y[:, i]

    y_max_2 = max(y)
    y_min_2 = min(y)

    ax = fig.add_subplot(3, 3, i + 1)

    ax.scatter(x, y, alpha=0.7, s=10)

    ax.plot([y_min_2, y_max_2], [y_min_2, y_max_2], c='cornflowerblue', ls='--')
    plt.axis('equal')
    ax.set_xlim(y_min_2, y_max_2)
    ax.set_ylim(y_min_2, y_max_2)
    ax.set_title(output_parameters[i])

plt.show()

fig.savefig("D:\\PythonProject\\RebuildProject\\Rock\\imgs\\MRCk2_MDN20231129_Scatter.png")