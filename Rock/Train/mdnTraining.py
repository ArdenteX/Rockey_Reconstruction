import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from Rock.Model.MDN import mdn, NLLLoss
from tqdm import tqdm
from Rock.Utils.Recorder import Recorder


class mdnTraining:
    def __init__(self, file_path='/Users/xuhongtao/pycharmprojects/resource/Gas_Giants_Core_Earth20W.xlsx', learning_rate=0.1984, batch_size=128, hidden_size=256, n_gaussian=3, is_gpu=True):
        self.f_p = file_path
        self.lr = learning_rate
        self.b_s = batch_size
        self.h_s = hidden_size
        self.n_g = n_gaussian
        self.device = torch.device('cuda' if is_gpu and torch.cuda.is_available() else 'cpu')

    def load_data(self):
        """
        Load Data and Normalization
        :return: Tensor() * 6
        """
        data = pd.read_excel(self.f_p)
        data['M_total (M_E)'] = data['Mcore (M_J/10^3)'] + data['Menv (M_E)']
        # Data Shuffle
        data = data.sample(frac=1).reset_index(drop=True)
        # Data Normalization
        scaler = MinMaxScaler(feature_range=[-1, 1])

        # Generate train data, validation data, and test data (6:2:2)
        train_set = data.iloc[:int(len(data) * 0.6), :]
        train_x = torch.from_numpy(scaler.fit_transform(train_set.iloc[:, [0, 1, 5]]).astype('float32'))
        train_y = torch.from_numpy(np.array(train_set.iloc[:, [6, 8, 9, 11]]).astype('float32'))

        val_set = data.iloc[int(len(data) * 0.6):int(len(data) * 0.8), :]
        val_x = torch.from_numpy(scaler.fit_transform(val_set.iloc[:, [0, 1, 5]]).astype('float32'))
        val_y = torch.from_numpy(np.array(val_set.iloc[:, [6, 8, 9, 11]]).astype('float32'))

        test_set = data.iloc[int(len(data) * 0.8):, :]
        test_x = torch.from_numpy(scaler.fit_transform(test_set.iloc[:, [0, 1, 5]]).astype('float32'))
        test_y = torch.from_numpy(np.array(test_set.iloc[:, [6, 8, 9, 11]]).astype('float32'))

        return train_x, train_y, val_x, val_y, test_x, test_y, scaler

    def fit(self):
        pass
# load data
tra_x, tra_y, v_x, v_y, t_x, t_y, s = mdnTraining().load_data()

# Parameters
input_size = tra_x.shape[1]
output_size = tra_y.shape[1]
num_gaussian = 3
hidden_size = 256

# load model
model = mdn(input_size, output_size, num_gaussian, hidden_size)

# Loss Function
criterion = NLLLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1986)

for name, parameters in model.named_parameters():
    print(name, ':', parameters.size())

pbar = tqdm(range(0, tra_x.shape[0], 256))

model.train()
train_loss = Recorder()
val_loss = Recorder()
for p in pbar:
    mini_batch_x = tra_x[p:p + 256]
    mini_batch_y = tra_y[p:p + 256]

    pi, mu, sigma = model(mini_batch_x)

    pi_anti_nor = s.inverse_transform(pi.detach().numpy())
    mu_anti_nor = s.inverse_transform(mu.detach().numpy())
    sigma_anti_nor = s.inverse_transform(sigma.detach().numpy())



