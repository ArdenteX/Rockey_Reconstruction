import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from Rock.Model.MDN_by_Pytorch import mdn as mdn_advance, RelativeError, Mixture, NLLLoss
from tqdm import tqdm
from Rock.Utils.Recorder import Recorder
from Rock.Utils.WarmUpLR import WarmUpLR

from Rock.Utils.View import init_weights, split_weights, visualize_lastlayer, visualize_train_loss, visualize_test_loss
from tensorboardX import SummaryWriter


# Develop in MAC: /Users/xuhongtao/pycharmprojects/resource/Gas_Giants_Core_Earth20W.xlsx
# Develop in Windows: D:\\Resource\\Gas_Giants_Core_Earth20W.xlsx
"""
    数据处理穷举：
        1.数据归一化
        2.不做任何数据处理
        3.数据归一化+反归一化
"""


class mdnTraining:
    def __init__(self, batch_size, file_path='D:\\Resource\\Gas_Giants_Core_Earth20W.xlsx', learning_rate=0.001984, hidden_size=256, n_gaussian=5, is_gpu=True, epoch=150, if_shuffle=True, is_normal=True, weight_decay=0.1):
        self.f_p = file_path
        self.lr = learning_rate
        self.b_s = batch_size
        self.h_s = hidden_size
        self.n_g = n_gaussian
        self.epoch = epoch
        self.if_shuffle = if_shuffle
        self.if_normal = is_normal
        self.w_d = weight_decay
        self.device = torch.device('cuda' if is_gpu and torch.cuda.is_available() else 'cpu')
        self.writer = SummaryWriter(log_dir="D:\\Resource\\MDN\\Log\\")

        print("Batch size: {}, Learning rate:{}, Hidden size:{}, is GPU: {}, is Normal: {}".format(self.b_s, self.lr,
                                                                                               self.h_s, is_gpu,
                                                                                               self.if_normal))

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

        train_x = None
        train_y = None
        val_x = None
        val_y = None
        test_x = None
        test_y = None

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

        data_x = data.loc[:, input_parameters]
        data_y = data.loc[:, output_parameters]

        # Generate train data, validation data, and test data default: (9:0.5:0.5)
        if self.if_normal:
            data_x_nor = pd.DataFrame(scaler_x.fit_transform(data_x))
            data_y_nor = pd.DataFrame(scaler_y.fit_transform(data_y))

            if self.if_shuffle:
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

            elif not self.if_shuffle:
                train_x = data_x_nor.iloc[:int(len(data) * 0.8), :]
                train_y = data_y_nor.iloc[:int(len(data) * 0.8), :]

                val_x = data_x_nor.iloc[int(len(data) * 0.8):int(len(data) * 0.9), :]
                val_y = data_y_nor.iloc[int(len(data) * 0.8):int(len(data) * 0.9), :]

                test_x = data_x_nor.iloc[int(len(data) * 0.9):, :]
                test_y = data_y_nor.iloc[int(len(data) * 0.9):, :]

        else:
            if self.if_shuffle:
                train_x = data_x.sample(frac=0.9)
                train_y = data_y.iloc[train_x.index]

                val_x = data_x.drop(train_x.index).sample(frac=0.5)
                val_y = data_y.iloc[val_x.index]

                test_x = data_x.drop(train_x.index).drop(val_x.index)
                test_y = data_y.iloc[test_x.index]

                train_x.reset_index(inplace=True, drop=True)
                train_y.reset_index(inplace=True, drop=True)
                val_x.reset_index(inplace=True, drop=True)
                val_y.reset_index(inplace=True, drop=True)
                test_x.reset_index(inplace=True, drop=True)
                test_y.reset_index(inplace=True, drop=True)

            elif not self.if_shuffle:
                train_x = data_x.iloc[:int(len(data) * 0.8), :]
                train_y = data_y.iloc[:int(len(data) * 0.8), :]

                val_x = data_x.iloc[int(len(data) * 0.8):int(len(data) * 0.9), :]
                val_y = data_y.iloc[int(len(data) * 0.8):int(len(data) * 0.9), :]

                test_x = data_x.iloc[int(len(data) * 0.9):, :]
                test_y = data_y.iloc[int(len(data) * 0.9):, :]

        train_x = torch.from_numpy(train_x.to_numpy()).double()
        train_y = torch.from_numpy(train_y.to_numpy()).double()
        val_x = torch.from_numpy(val_x.to_numpy()).double()
        val_y = torch.from_numpy(val_y.to_numpy()).double()
        test_x = torch.from_numpy(test_x.to_numpy()).double()
        test_y = torch.from_numpy(test_y.to_numpy()).double()

        t_set = TensorDataset(train_x, train_y)
        train_loader = DataLoader(t_set, batch_size=self.b_s, shuffle=False, num_workers=8)

        v_set = TensorDataset(val_x, val_y)
        validation_loader = DataLoader(v_set, batch_size=self.b_s, shuffle=False, num_workers=8)

        # test_set = data.iloc[int(len(data) * 0.8):, :]
        # test_x = torch.from_numpy(scaler_x.fit_transform(test_set.iloc[:, [0, 1, 5]]).astype('float32'))
        # test_y = torch.from_numpy(scaler_y.fit_transform(test_set.iloc[:, [6, 8, 9, 11]]).astype('float32'))

        return train_loader, validation_loader, test_x, test_y, scaler_x, scaler_y

    def plot_line_2(self, y_1: str, y_2: str, df: pd.DataFrame, output_path: str, fig_size: tuple = (10, 6), dpi: int = 300):
        """
        Plot Merge Line (2 Lines) using Seaborn
        :param y_1: Name of Line 1
        :param y_2: Name of Line 2
        :param df: Dataframe
        :param fig_size:
        :param output_path:
        :param dpi:
        :return: Show Line picture and save to the specific location
        """
        fig = plt.figure(figsize=fig_size)
        sns.lineplot(x='epoch', y=y_1, data=df)
        sns.lineplot(x='epoch', y=y_2, data=df)
        plt.show()
        fig.savefig(output_path, dpi=dpi)

    def fit(self, t_l, v_l):
        # load data
        # t_l, v_l, test_x, test_y, s_x, s_y = self.load_data()

        # Parameters
        input_size = 3
        output_size = 4

        # load model
        model = mdn_advance(input_size, output_size, self.n_g, self.h_s)

        # Xavier Init
        init_weights(model)
        model = nn.DataParallel(model)
        model.to(self.device)

        # Loss Function
        criterion = NLLLoss()

        # MSE
        mse = nn.MSELoss()

        # Sample Function
        pdf = Mixture()

        # Relative Error
        r_e = RelativeError()

        # Optimizer
        optimizer = torch.optim.Adam(split_weights(model), lr=self.lr, weight_decay=self.w_d)
        # optimizer = torch.optim.SGD(split_weights(model), lr=self.lr, weight_decay=0.1, momentum=0.9, nesterov=True)

        # Schedular 1
        warmup = WarmUpLR(optimizer, len(t_l) * 5)

        # Schedular 2
        # lr_schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones=[115], gamma=0.7)

        for name, parameters in model.named_parameters():
            print(name, ':', parameters.size())



        epoch_train_loss = []
        epoch_val_loss = []
        epoch_train_r2 = []
        epoch_val_r2 = []
        epoch_r_e = []
        epoch_mse = []
        best_r2 = 0.80

        # Epoch
        for e in range(self.epoch):
            model.train()

            train_loss = Recorder()
            val_loss = Recorder()
            train_r2_recorder = Recorder()
            val_r2_recorder = Recorder()
            r_e_recorder = Recorder()
            mse_recorder = Recorder()

            # if e > 5:
            #     lr_schedular.step()

            # lr_schedular.step()

            i = 0

            with tqdm(t_l, unit='batch') as t_epoch:
                for x, y in t_epoch:
                    if e <= 5:
                        warmup.step()
                    t_epoch.set_description(f"Epoch {e + 1} Training")

                    mini_batch_x = x.to(self.device)
                    mini_batch_y = y.to(self.device)
                    optimizer.zero_grad()

                    pi, mu, sigma = model(mini_batch_x)

                    mixture = pdf(pi, mu, sigma)

                    y_pred = mixture.sample()

                    y_pred = y_pred.cpu().numpy()
                    val_batch_y = mini_batch_y.cpu().numpy()

                    r2_per = r2_score(val_batch_y, y_pred)

                    # pi_anti_nor = s.inverse_transform(pi.detach().numpy())
                    # mu_anti_nor = s.inverse_transform(mu.detach().numpy())
                    # sigma_anti_nor = s.inverse_transform(sigma.detach().numpy())

                    # t_y_pred = sample(pi, mu, sigma)          # Sampling for MSE Loss Function

                    # loss = criterion(mini_batch_y, pi, mu, sigma)     # NLLLoss Function
                    loss = criterion(pi, mu, sigma, mini_batch_y)     # NLLLoss Function
                    # loss = criterion(t_y_pred, mini_batch_y)              # MSE

                    train_loss.update(loss.item())
                    train_r2_recorder.update(r2_per.astype('float32'))

                    loss.backward()

                    optimizer.step()


                    n_iter = (e - 1) * len(t_l) + i + 1
                    visualize_lastlayer(self.writer, model, n_iter)
                    visualize_train_loss(self.writer, loss.item(), n_iter)

                    t_epoch.set_postfix(loss="{:.4f}".format(train_loss.val), lr="{:.6f}".format(optimizer.state_dict()['param_groups'][0]['lr']), loss_avg="{:.4f}".format(train_loss.avg), r2="{:.4f}".format(train_r2_recorder.avg))

                epoch_train_r2.append(train_r2_recorder.avg)
                epoch_train_loss.append(train_loss.avg)

            with torch.no_grad():
                model.eval()

                with tqdm(v_l, unit='batch') as v_epoch:
                    v_epoch.set_description(f"Epoch {e + 1} Validating")

                    for v_x, v_y in v_epoch:

                        val_batch_x = v_x.to(self.device)
                        val_batch_y = v_y.to(self.device)

                        pi_val, mu_val, sigma_val = model(val_batch_x)

                        mixture = pdf(pi_val, mu_val, sigma_val)

                        # loss = criterion(val_batch_y, pi_val, mu_val, sigma_val)         # NLLLoss Function Probability Density Function
                        loss = criterion(pi_val, mu_val, sigma_val, val_batch_y)         # NLLLoss Function Sampling

                        y_pred = mixture.sample()

                        # loss = criterion(y_pred, val_batch_y)       # MSE
                        mse_per = mse(y_pred, val_batch_y)

                        y_pred = y_pred.cpu().numpy()
                        val_batch_y = val_batch_y.cpu().numpy()

                        r2_per = r2_score(val_batch_y, y_pred)
                        r_e_per = r_e(val_batch_y, y_pred)

                        val_loss.update(loss.item())
                        mse_recorder.update(mse_per.item())
                        val_r2_recorder.update(r2_per.astype('float32'))
                        r_e_recorder.update(r_e_per.astype('float32'))

                        v_epoch.set_postfix(loss_val="{:.4f}".format(val_loss.val), loss_avg="{:.4f}".format(val_loss.avg), r2="{:.4f}".format(val_r2_recorder.avg), relative_error="{:.4f}".format(r_e_recorder.avg), mse="{:.4f}".format(mse_recorder.avg))

                epoch_val_loss.append(val_loss.avg)
                epoch_val_r2.append(val_r2_recorder.avg)
                epoch_r_e.append(r_e_recorder.avg)
                epoch_mse.append(mse_recorder.avg)

                visualize_test_loss(self.writer, epoch_val_loss[-1], e)

                if e >= 15:
                    if val_r2_recorder.avg > best_r2:
                        torch.save(model.state_dict(), 'D:\\Resource\\MDN\\model_best_mdn_normalization.pth')
                        best_r2 = val_r2_recorder.avg
                        print("Save Best model: R2:{:.4f}, Loss Avg:{:.4f}".format(val_r2_recorder.avg, val_loss.avg))

                    else:
                        print(" ")
                else:
                    print(" ")

        return epoch_train_loss, epoch_val_loss, epoch_val_r2, epoch_train_r2, epoch_r_e, epoch_mse
