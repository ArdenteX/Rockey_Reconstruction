import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from Rock.Model.MDN import mdn, NLLLoss
from tqdm import tqdm
from Rock.Utils.Recorder import Recorder
from Rock.Utils.WarmUpLR import WarmUpLR


# Develop in MAC: /Users/xuhongtao/pycharmprojects/resource/Gas_Giants_Core_Earth20W.xlsx
"""
    数据处理穷举：
        1.数据归一化
        2.不做任何数据处理
        3.数据归一化+反归一化
"""


class mdnTraining:
    def __init__(self, file_path='D:\\Resource\\Gas_Giants_Core_Earth20W.xlsx', learning_rate=0.001984, batch_size=512, hidden_size=256, n_gaussian=3, is_gpu=True, epoch=150):
        self.f_p = file_path
        self.lr = learning_rate
        self.b_s = batch_size
        self.h_s = hidden_size
        self.n_g = n_gaussian
        self.epoch = epoch
        self.device = torch.device('cuda' if is_gpu and torch.cuda.is_available() else 'cpu')

    def load_data(self):
        """
        Load Data and Normalization
        :return: Tensor() * 6
        """
        data = pd.read_excel(self.f_p)
        data['M_total (M_E)'] = data['Mcore (M_J/10^3)'] + data['Menv (M_E)']
        # Data Shuffle
        # data = data.sample(frac=1).reset_index(drop=True)

        # Data Normalization
        scaler_x = MinMaxScaler(feature_range=[0, 1])
        scaler_y = MinMaxScaler(feature_range=[0, 1])

        # Generate train data, validation data, and test data (6:2:2)
        train_set = data.iloc[:int(len(data) * 0.6), :]
        train_x = torch.from_numpy(scaler_x.fit_transform(train_set.iloc[:, [0, 1, 5]]).astype('float32'))
        train_y = torch.from_numpy(scaler_y.fit_transform(train_set.iloc[:, [6, 8, 9, 11]]).astype('float32'))

        t_set = TensorDataset(train_x, train_y)
        train_loader = DataLoader(t_set, batch_size=self.b_s, shuffle=True, num_workers=8)

        val_set = data.iloc[int(len(data) * 0.6):int(len(data) * 0.8), :]
        val_x = torch.from_numpy(scaler_x.fit_transform(val_set.iloc[:, [0, 1, 5]]).astype('float32'))
        val_y = torch.from_numpy(scaler_y.fit_transform(val_set.iloc[:, [6, 8, 9, 11]]).astype('float32'))

        v_set = TensorDataset(val_x, val_y)
        validation_loader = DataLoader(v_set, batch_size=self.b_s, shuffle=True, num_workers=8)

        # test_set = data.iloc[int(len(data) * 0.8):, :]
        # test_x = torch.from_numpy(scaler_x.fit_transform(test_set.iloc[:, [0, 1, 5]]).astype('float32'))
        # test_y = torch.from_numpy(scaler_y.fit_transform(test_set.iloc[:, [6, 8, 9, 11]]).astype('float32'))

        return train_loader, validation_loader, scaler_x, scaler_y

    def plot_loss(self, t_l, v_l):
        result_pd = pd.DataFrame()
        result_pd['epoch'] = range(self.epoch)
        result_pd['train_loss_avg'] = t_l
        result_pd['validation_loss_avg'] = v_l

        sns.lineplot(x='epoch', y='train_loss_avg', data=result_pd)
        sns.lineplot(x='epoch', y='validation_loss_avg', data=result_pd)
        plt.show()

    def fit(self):
        # load data
        t_l, v_l, s_x, s_y = mdnTraining().load_data()

        # Parameters
        input_size = 3
        output_size = 4

        # load model
        model = mdn(input_size, output_size, self.n_g, self.h_s).to(self.device)

        # Loss Function
        criterion = NLLLoss()

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        # Schedular
        warmup = WarmUpLR(optimizer, len(t_l) * 5)

        for name, parameters in model.named_parameters():
            print(name, ':', parameters.size())

        epoch_train_loss = []
        epoch_val_loss = []
        best_loss = -2

        # Epoch
        for e in range(self.epoch):
            model.train()

            train_loss = Recorder()
            val_loss = Recorder()

            with tqdm(t_l, unit='batch') as t_epoch:
                for x, y in t_epoch:
                    if e < 5:
                        warmup.step()
                    t_epoch.set_description(f"Epoch {e + 1} Training")

                    mini_batch_x = x.to(self.device)
                    mini_batch_y = y.to(self.device)
                    optimizer.zero_grad()

                    pi, mu, sigma = model(mini_batch_x)

                    # pi_anti_nor = s.inverse_transform(pi.detach().numpy())
                    # mu_anti_nor = s.inverse_transform(mu.detach().numpy())
                    # sigma_anti_nor = s.inverse_transform(sigma.detach().numpy())

                    loss = criterion(pi, mu, sigma, mini_batch_y)
                    train_loss.update(loss.item())

                    loss.backward()

                    optimizer.step()

                    t_epoch.set_postfix(loss="{:.4f}".format(train_loss.val), lr="{:.4f}".format(optimizer.state_dict()['param_groups'][0]['lr']), loss_avg=train_loss.avg)

                epoch_train_loss.append(train_loss.avg)

            with torch.no_grad():
                model.eval()

                with tqdm(v_l, unit='batch') as v_epoch:
                    v_epoch.set_description(f"Epoch {e + 1} Validating")

                    for v_x, v_y in v_epoch:

                        val_batch_x = v_x.to(self.device)
                        val_batch_y = v_y.to(self.device)

                        pi_val, mu_val, sigma_val = model(val_batch_x)

                        loss = criterion(pi_val, mu_val, sigma_val, val_batch_y)
                        val_loss.update(loss.item())

                        v_epoch.set_postfix(loss_val="{:.4f}".format(val_loss.val), loss_avg="{:.4f}".format(val_loss.avg))

                    epoch_val_loss.append(val_loss.avg)

                    if e >= 50:
                        if val_loss.avg < best_loss:
                            torch.save(model.state_dict(), 'D:\\Resource\\MDN\\model_best_mdn_normalization.pth')
                            best_loss = val_loss.avg
                            print("Save Best model: Loss:{:.4f}, Loss Avg:{:.4f}".format(val_loss.val, val_loss.avg))

                        else:
                            print(" ")
                    else:
                        print(" ")



        return epoch_train_loss, epoch_val_loss


# test_train = mdnTraining()
# test_train.fit()
# tra_x, tra_y, v_x, v_y, t_x, t_y, s_x, s_y = test_train.load_data()
# model = mdn(tra_x.shape[1], tra_y.shape[1], 3, 256).to(test_train.device)
# pi, mu, sigma = model(tra_x[:64])
#
# z_score = (torch.unsqueeze(tra_y[:64], dim=1) - mu) / sigma
#
# normal_loglik = (-0.5 * torch.einsum("bij, bij->bi", z_score, z_score)) - torch.sum(torch.log(sigma), dim=-1)
#
# loglik = torch.logsumexp(pi + normal_loglik, dim=-1)
#
# avg_abs = torch.abs(torch.mean(loglik))
# p_test = tqdm(range(0, 6400, 64))
# p_test_1 = tqdm(range(0, 6400, 64))
# for p_v in p_test:
#
#     p_test.set_postfix(loss=p_v)
#
# for p_1 in p_test_1:
#     p_test_1.set_postfix(loss=p_1)
#
# a = 12000
# b = 256
# with tqdm(total=a / b) as e:
#     for i in range(a, b):
#         e.set_postfix(loss=1)
#         e.update(b)
