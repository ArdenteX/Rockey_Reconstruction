import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

data = pd.read_excel('D:\\Resource\\Gas_Giants_Core_Earth20W.xlsx')

test_part = data.iloc[:2000, :]

t_2 = MinMaxScaler(feature_range=[-1, 1])
t_2_ = t_2.fit_transform(data.iloc[:, :-1])

t_2_.min()
t_2_.max()


class mdnTraining:
    def __init__(self, file_path='D:\\Resource\\Gas_Giants_Core_Earth20W.xlsx', learning_rate=0.1984, batch_size=128, hidden_size=256, n_gaussian=3, is_normalization=True, is_gpu=True):
        self.f_p = file_path
        self.lr = learning_rate
        self.b_s = batch_size
        self.h_s = hidden_size
        self.n_g = n_gaussian
        self.is_nor = is_normalization
        self.device = torch.device('cuda' if is_gpu and torch.cuda.is_available() else 'cpu')

    def load_data(self):
        d = pd.read_excel(self.f_p)



