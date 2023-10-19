import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
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
scaler_x = MinMaxScaler(feature_range=[0, 1])
scaler_y = MinMaxScaler(feature_range=[0, 1])
train_set = data.iloc[:int(len(data) * 0.6), :]
train_x = torch.from_numpy(scaler_x.fit_transform(train_set.iloc[:, [0, 1, 5]]).astype('float32'))
train_y = torch.from_numpy(scaler_y.fit_transform(train_set.iloc[:, [6, 8, 9, 11]]).astype('float32'))

val_set = data.iloc[int(len(data) * 0.6):int(len(data) * 0.8), :]
val_x = torch.from_numpy(scaler_x.fit_transform(val_set.iloc[:, [0, 1, 5]]).astype('float32'))
val_y = torch.from_numpy(scaler_y.fit_transform(val_set.iloc[:, [6, 8, 9, 11]]).astype('float32'))

t_set = torch.utils.data.TensorDataset(train_x, train_y)
train_loader = DataLoader(t_set, batch_size=256, shuffle=True, num_workers=8)
len(train_loader.sampler)

a = [1, 2, 3]
b = [2, 2, 2]
y = [1, 2, 3]


