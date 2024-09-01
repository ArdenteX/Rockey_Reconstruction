import joblib
import numpy as np
import torch.nn as nn
import torch
from Rock.Model.MLP import MLP_Attention
from Rock.Model.EnsembleMDN import EnsembleMDN
from tortreinador.utils.View import init_weights
from tortreinador.models.MDN import Mixture
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

"""
    Sampling from hyperellipsoid 
    X_ellipsoid = 5 * L * X_ball + P
        - L = Cholesky decomposition of the covariance matrix of posterior distribution
        - P = Mean of peak of posterior distribution
        - X_ball = X_U ** 1/N * Y
            X_U = Samples which uniform distribution in the interval (0, 1)
            Y = X_N / Euclidean norm(X_N)
"""

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

model_posterior_distribution = EnsembleMDN(int(len(input_parameters) / 2), len(output_parameters), 10, 256, kernel_size=2)
init_weights(model_posterior_distribution)
model_posterior_distribution = nn.DataParallel(model_posterior_distribution)
model_posterior_distribution.load_state_dict(torch.load("D:\\Resource\\MDN\\rockyExoplanetV3\\NoiseADD\\best_model.pth"))

model_cov = MLP_Attention(input_size=len(input_parameters[:4]), output_size=len(output_parameters), layer1_size=256, layer2_size=128, mode='cov')
init_weights(model_cov)
model_cov = nn.DataParallel(model_cov)
model_cov.load_state_dict(torch.load("D:\\Resource\\MDN\\MLPSelfAttention\\Test\\best_model.pth"))

t_x = np.load("D:\\Resource\\MDN\\rockyExoplanetV3\\NoiseADD\\testData\\test_x.npy")
t_y = np.load("D:\\Resource\\MDN\\rockyExoplanetV3\\NoiseADD\\testData\\test_y.npy")
m_y = joblib.load("D:\\Resource\\MDN\\rockyExoplanetV3\\NoiseADD\\testData\\MDN_v3_yscaler_20240630.save")
m_x = joblib.load("D:\\Resource\\MDN\\rockyExoplanetV3\\NoiseADD\\testData\\MDN_v3_Xscaler_20240630.save")

mixture = Mixture()

selected_idx = np.random.choice(t_x.shape[0], size=20000, replace=False)
t_x_sub = t_x[selected_idx, :]
t_y_sub = t_y[selected_idx, :]

t_x_sub = m_x.transform(t_x_sub)
t_x_sub_tensor = torch.from_numpy(t_x_sub)
t_x_sub_ori, t_x_sub_noise = t_x_sub_tensor.chunk(2, dim=-1)

model_posterior_distribution.eval()
p_distribution = model_posterior_distribution(t_x_sub_ori.to('cuda'), t_x_sub_noise.to('cuda'))
cov_matrix = model_cov(t_x_sub_noise.to('cuda'))

exp_test = torch.exp(p_distribution[0])
torch.sum(p_distribution[0])

pdf = mixture(torch.exp(p_distribution[0]), p_distribution[1], p_distribution[2])

y_range = torch.linspace(-2, 2, 20000).reshape(-1, 1)

pdf_values = torch.exp(pdf.log_prob(y_range.to('cuda'))).detach().cpu().numpy()

feature_pdf_peak = []
for f in range(pdf_values.shape[1]):
    feature_pdf_peak.append(y_range[np.argmax(pdf_values[:, f], axis=0)].item())

P = np.array(feature_pdf_peak)
# 绘制概率密度函数
plt.plot(y_range, pdf_values[:, 0])
plt.axvline(feature_pdf_peak[0].item(), color='r', linestyle='--', label=f'Peak value: {feature_pdf_peak[0].item():.2f}')
plt.legend()
plt.show()

L = np.linalg.cholesky(cov_matrix[0].detach().cpu().numpy())
X_U = np.random.random(20000) ** (1.0 / len(output_parameters))
X = np.random.normal(size=(len(output_parameters), 20000))
Y = X / np.linalg.norm(X, axis=0)
X_ball = (X_U * Y).T
x_ellipsoid = P + 5 * L.dot(X_ball)

l_x_ball = L.dot(X_ball[:, :, np.newaxis])

# x_0_ = np.average(data[:, 0])
# x_1_ = np.average(data[:, 1])
#
# cov = (1 / (len(data) - 1)) * np.sum([(data[:, 0] - x_0_) * (data[:, 1] - x_1_)])
#
# a = (data[:, 0] - x_0_) * (data[:, 1] - x_1_)



# attn = nn.MultiheadAttention(5, 5)
# data_tensor = torch.from_numpy(data).to(torch.float32).unsqueeze(0)
# pred = attn(data_tensor, data_tensor, data_tensor)
# pred_0 = pred[0].detach().numpy()
# pred_1 = pred[1].detach().numpy()
#
# pred_0_T = pred_0.transpose(1, 0, -1)
#
# data.reshape((5, 10, 5))


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# 示例输入
np.random.seed(42)
num_features = 10
seq_len = 20

q_processor = nn.Linear(num_features, num_features)
k_processor = nn.Linear(num_features, num_features)
v_processor = nn.Linear(num_features, num_features)

q_weights = q_processor.weight.data.numpy()
k_processor = k_processor.weight.data.numpy()
v_processor = v_processor.weight.data.numpy()

# 随机生成输入数据
X = np.random.rand(seq_len, num_features)

# 随机生成 Query, Key, Value 权重矩阵
W_Q = np.random.rand(num_features, num_features)
W_K = np.random.rand(num_features, num_features)
W_V = np.random.rand(num_features, num_features)

# 生成 Query, Key, Value 矩阵
Q = np.dot(X, q_weights)
K = np.dot(X, k_processor)
V = np.dot(X, v_processor)

Q_troch = q_processor(torch.from_numpy(X).to(torch.float32))
K_troch = q_processor(torch.from_numpy(X).to(torch.float32))
V_troch = q_processor(torch.from_numpy(X).to(torch.float32))

# 计算自注意力输出
# output, attention_weights = self_attention(Q, K, V)

d_k = Q_troch.shape[-1]
scores = np.dot(Q_troch.detach().numpy(), K_troch.T.detach().numpy()) / np.sqrt(d_k)
scores_torch = torch.matmul(Q_troch, K_troch.T) / np.sqrt(d_k)

attention_weights = softmax(scores)
attention_weights_torch = torch.softmax(scores_torch, dim=-1)

output = np.dot(attention_weights, V_troch.detach().numpy())
output_torch = torch.matmul(attention_weights_torch, V_troch)

print("Attention Weights:\n", attention_weights)
print("Output:\n", output)


class SelfAttention(nn.Module):
    def __init__(self, input_feature, output_feature):
        super().__init__()
        self.output_dim = output_feature
        self.input_dim = input_feature
        self.d_k = output_feature ** 0.5
        self.Q_layer = nn.Linear(self.input_dim, self.output_dim * self.output_dim)
        self.K_layer = nn.Linear(self.input_dim, self.output_dim * self.output_dim)
        self.V_layer = nn.Linear(self.input_dim, self.output_dim * self.output_dim)

    def forward(self, x):
        Q = self.Q_layer(x)
        K = self.K_layer(x)
        V = self.V_layer(x)

        scores = torch.matmul(Q, K.T) / self.d_k
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.view(-1, self.output_dim, self.output_dim)

        return attn_output, attn_weights


X = torch.rand(seq_len, num_features, requires_grad=True).to(torch.float64)

model_test = MLP_Attention(num_features, 32, 16, num_features)
# model_test.zero_grad()
mse = nn.MSELoss()
result = model_test(X)

result_cov = torch.mean(result[0], dim=0)

# 使矩阵对称
x = (result[0] + result[0].transpose(1, 2)) / 2
identity = torch.eye(10).to(x.device)
x = x + identity * 1e-3

cov_pred = result[0][:num_features, :num_features]

y = torch.from_numpy(np.cov(X.detach().numpy(), rowvar=False)).to(torch.float32)
y_torch = torch.cov(torch.from_numpy(X))

loss = mse(result[0][:num_features, :num_features], y)


def torch_cov(X):
    X_mean = torch.mean(X, dim=0)
    X_centered = X - X_mean
    n_samples = X.shape[0]
    covariance_matrix = torch.mm(X_centered.T, X_centered) / (n_samples - 1)
    return covariance_matrix




# 超椭球体的参数
mean = [0, 0]
cov = [[1, 0.8], [0.8, 1]]  # 协方差矩阵

# 生成样本
np.random.seed(42)
samples = np.random.multivariate_normal(mean, cov, 1000)

# 绘图
fig, ax = plt.subplots(figsize=(8, 6))

# 样本的散点图
ax.scatter(samples[:, 0], samples[:, 1], s=10, color='gray', alpha=0.5)

# 绘制椭圆的函数
def plot_cov_ellipse(cov, pos, nstd=1, ax=None, **kwargs):
    """
    根据指定的协方差矩阵（cov）和均值（pos）绘制`nstd`倍标准差的误差椭圆。
    """
    if ax is None:
        ax = plt.gca()

    # 计算特征值和对应的特征向量
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # 计算最大特征向量与x轴的角度
    vx, vy = eigvecs[:, 0][0], eigvecs[:, 0][1]
    theta = np.degrees(np.arctan2(vy, vx))

    # 椭圆的宽度和高度
    width, height = 2 * nstd * np.sqrt(eigvals)

    # 绘制椭圆
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_patch(ellip)

# 绘制1σ和2σ椭圆
plot_cov_ellipse(cov, mean, nstd=1, ax=ax, edgecolor='red')
plot_cov_ellipse(cov, mean, nstd=2, ax=ax, edgecolor='red')

# 标签和标题
ax.set_xlabel(r'参数 $\theta_1$')
ax.set_ylabel(r'参数 $\theta_2$')
ax.set_title('超椭球体')
plt.grid(True)

# 显示图形
plt.show()


