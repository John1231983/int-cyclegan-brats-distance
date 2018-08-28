import torch
from torch import nn
from torch.autograd import Function
import numpy as np

str_device = 'cpu'

class DistanceLossTorch(nn.Module):
    def __init__(self, fea_shape, cell_range = 64, crop_size = 4, pair_num = 2, threshold = 3):
    #def __init__(self, fea_shape, cell_range = 16, crop_size = 16, pair_num = 5, threshold = 3):
        super(DistanceLossTorch, self).__init__()
        self.cell_range = cell_range
        self.crop_size = crop_size
        self.pair_num = pair_num
        self.threshold = threshold
        self.batch, self.channel, self.height, self.width = fea_shape
        self.h_crop_num = (self.height-1) / self.crop_size +1
        self.w_crop_num = (self.width-1) / self.crop_size +1
        self.hist_num = self.h_crop_num * self.w_crop_num
        self.hist_dim = 256 / cell_range

    def forward(self, fea_in, fea_out, hist_pair):
        hist_fea_in = torch.zeros(self.batch, self.channel, self.hist_num, self.hist_dim, device=torch.device(str_device)).double()
        hist_fea_out = torch.zeros(self.batch, self.channel, self.hist_num, self.hist_dim, device=torch.device(str_device)).double()
        hist_diff_in = torch.zeros(self.batch, self.channel, self.hist_num, self.pair_num, self.hist_dim, device=torch.device(str_device)).double()
        hist_diff_out = torch.zeros(self.batch, self.channel, self.hist_num, self.pair_num, self.hist_dim, device=torch.device(str_device)).double()
        #hist_pair = torch.randint(self.hist_num, (self.batch, self.channel, self.hist_num, self.pair_num), device=torch.device(str_device)).long()
        rgb_in = ((fea_in +1) *127.5) / self.cell_range
        rgb_out = ((fea_out +1) *127.5) / self.cell_range
        count = 0
        for b_idx in range(self.batch):
            for c_idx in range(self.channel):
                for h_idx in range(self.h_crop_num):
                    for w_idx in range(self.w_crop_num):
                        hist_idx = h_idx * self.w_crop_num + w_idx
                        crop_fea_in = rgb_in[b_idx, c_idx, h_idx*self.crop_size:(h_idx+1)*self.crop_size, w_idx*self.crop_size:(w_idx+1)*self.crop_size]
                        crop_fea_out = rgb_out[b_idx, c_idx, h_idx*self.crop_size:(h_idx+1)*self.crop_size, w_idx*self.crop_size:(w_idx+1)*self.crop_size]
                        for v_idx in range(self.hist_dim):
                            soft_hist = 1 - torch.abs(crop_fea_in - v_idx)
                            soft_hist = torch.max(soft_hist, torch.zeros_like(soft_hist))
                            hist_fea_in[b_idx, c_idx, hist_idx, v_idx] = torch.sum(soft_hist)
                            soft_hist = 1 - torch.abs(crop_fea_out - v_idx)
                            soft_hist = torch.max(soft_hist, torch.zeros_like(soft_hist))
                            hist_fea_out[b_idx, c_idx, hist_idx, v_idx] = torch.sum(soft_hist)
        for b_idx in range(self.batch):
            for c_idx in range(self.channel):
                for h_idx in range(self.hist_num):
                    pair_selected_in = hist_fea_in[b_idx, c_idx, hist_pair[b_idx, c_idx, h_idx], :]
                    hist_diff_in[b_idx, c_idx, h_idx, ...] = torch.abs(hist_fea_in[b_idx, c_idx, h_idx] - pair_selected_in)
                    pair_selected_out = hist_fea_out[b_idx, c_idx, hist_pair[b_idx, c_idx, h_idx], :]
                    hist_diff_out[b_idx, c_idx, h_idx, ...] = torch.abs(hist_fea_out[b_idx, c_idx, h_idx] - pair_selected_out)
        diff_in_mean = torch.mean(hist_diff_in, dim=-1, keepdim=True)
        hist_diff_bool = diff_in_mean <= self.threshold
        hist_diff_out = hist_diff_bool.double() * hist_diff_out
        return hist_diff_out

class CustomDistance(nn.Module):
    def __init__(self, fea_shape=(1,1,1,1), cell_range=64, crop_size=4, pair_num=2, threshold=3):
        super(CustomDistance, self).__init__()
        self.cell_range = cell_range
        self.crop_size = crop_size
        self.pair_num = pair_num
        self.threshold = threshold
        self.batch, self.channel, self.height, self.width = fea_shape
        self.h_crop_num = (self.height-1) / self.crop_size +1
        self.w_crop_num = (self.width-1) / self.crop_size +1
        self.hist_num = self.h_crop_num * self.w_crop_num
        self.hist_dim = 256 / cell_range
        self.hist_mu = torch.tensor([[[[range(self.hist_dim)]]]], device=torch.device(str_device)).double()
    def im2col(self, fea):
        tmp = fea.unfold(2, self.crop_size, self.crop_size)
        tmp1 = tmp.unfold(3, self.crop_size, self.crop_size)
        tmp2 = tmp1.reshape((tmp1.shape[0], tmp1.shape[1], tmp1.shape[2]*tmp1.shape[3], tmp1.shape[4]*tmp1.shape[5]))
        return tmp2
    def softhist(self, fea):
        fea_tmp = fea.unsqueeze(-1)
        fea_tmp1 = fea_tmp.repeat(1,1,1,1,self.hist_dim)
        tmp_mu = self.hist_mu.repeat(fea_tmp1.shape[0], fea_tmp1.shape[1], fea_tmp1.shape[2], fea_tmp1.shape[3], 1)
        tmp = 1 - torch.abs(fea_tmp1 - tmp_mu)
        tmp1 = torch.max(tmp, torch.zeros_like(tmp))
        tmp_sum = torch.sum(tmp1, dim=3)
        return tmp_sum
    def gather_abs(self, fea, idx):
        tmp_fea = fea.unsqueeze(3)
        tmp_fea_gather = tmp_fea.repeat(1,1,1,fea.shape[2],1)
        tmp_fea_gather = tmp_fea_gather.permute(0,1,3,2,4)
        tmp_idx = idx.unsqueeze(-1)
        tmp_idx = tmp_idx.repeat(1,1,1,1, fea.shape[3])
        tmp_gather = tmp_fea_gather.gather(-2, tmp_idx)
        tmp_fea_repeat = tmp_fea.repeat(1,1,1,idx.shape[3],1)
        tmp_abs = torch.abs(tmp_gather - tmp_fea_repeat)
        return tmp_abs
    def forward(self, fea_in, fea_out, idx_in):
        rgb_in = ((fea_in +1) *127.5) / self.cell_range
        rgb_out = ((fea_out +1) *127.5) / self.cell_range
        col_in = self.im2col(rgb_in)
        col_out = self.im2col(rgb_out)
        hist_fea_in = self.softhist(col_in)
        hist_fea_out = self.softhist(col_out)
        hist_diff_in = self.gather_abs(hist_fea_in, idx_in)
        hist_diff_out = self.gather_abs(hist_fea_out, idx_in)
        diff_in_mean = torch.mean(hist_diff_in, dim=-1, keepdim=True)
        hist_diff_bool = diff_in_mean <= self.threshold
        hist_diff_out = hist_diff_bool.double() * hist_diff_out
        return hist_diff_out


mode = DistanceLossTorch([1, 1, 8, 8])
m_in = torch.rand(1, 1, 8, 8, dtype=torch.double, device=torch.device(str_device)) *2 -1
m_out = torch.rand(1, 1, 8, 8, dtype=torch.double, device=torch.device(str_device), requires_grad=True) *2 -1
idx_in = torch.randint(4, (1, 1, 4, 2), device=torch.device(str_device)).long()
out = mode(m_in, m_out, idx_in)
print m_in
print m_out
print out
print torch.mean(out)
grad = torch.rand_like(out).double()
print out.backward(grad)

mode2 = CustomDistance()
out2 = mode2(m_in, m_out, idx_in)
print out2
print out2.backward(grad)

'''
inputfea = (m_in, m_out)
from torch.autograd.gradcheck import gradcheck
test = gradcheck(mode, inputfea, eps=1e-2, raise_exception=True)
print test


class Customabs(nn.Module):
    def __init__(self):
        super(Customabs, self).__init__()
    def forward(self, fea):
        tmp = 1 - torch.abs(fea)
        tmp = torch.max(tmp, torch.zeros_like(tmp))
        return torch.sum(tmp)

mode = Customabs()
inputfea = torch.rand(1, 1, 5, 5, dtype=torch.double, device=torch.device(str_device), requires_grad=True)
inputfea = (inputfea)
test = gradcheck(mode, inputfea, eps=1e-6, raise_exception=True)
print test
'''
