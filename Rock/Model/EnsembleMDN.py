import torch.nn as nn
import torch
from Rock.Model.MDN_by_Pytorch import mdn


class EnsembleMDN(torch.nn.Module):
    def __init__(self, input_size, output_size, num_gaussian, num_hidden, kernel_size=3):
        super().__init__()
        self.i_s = input_size
        self.o_s = output_size
        self.n_g = num_gaussian
        self.n_h = num_hidden
        self.mdn_original = mdn(self.i_s, self.o_s, self.n_g, self.n_h, mode='ensemble')
        self.mdn_noise = mdn(self.i_s, self.o_s, self.n_g, self.n_h, mode='ensemble')

        # ECA Module
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.attn = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False).to(torch.float64)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, x_n):
        p, m, s = self.mdn_original(x)
        p_n, m_n, s_n = self.mdn_noise(x_n)

        p_integrate = torch.stack([p, p_n], dim=1).unsqueeze(-1)
        m_integrate = torch.stack([m, m_n], dim=1)
        s_integrate = torch.stack([s, s_n], dim=1)

        attn_weight = self.avg_pool(p_integrate)
        attn_weight = self.sigmoid(self.attn(attn_weight.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1))

        p_integrate = torch.mean(p_integrate * attn_weight.expand_as(p_integrate), dim=1).squeeze(-1)
        m_integrate = torch.mean(m_integrate * attn_weight.expand_as(m_integrate), dim=1)
        s_integrate = torch.mean(s_integrate * attn_weight.expand_as(s_integrate), dim=1)

        p_integrate = torch.log_softmax(p_integrate, dim=-1)

        return p_integrate.double(), m_integrate.double(), s_integrate.double()
