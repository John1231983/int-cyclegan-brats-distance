import torch
from torch import nn
from torch.autograd import Function
import numpy as np
from torch.autograd.gradcheck import gradcheck

str_device = 'cuda:0'

class CustomDistance(nn.Module):
    def __init__(self, cell_range=64, crop_size=4, pair_num=2, threshold=3):
        super(CustomDistance, self).__init__()
        self.cell_range = cell_range
        self.crop_size = crop_size
        self.pair_num = pair_num
        self.threshold = threshold
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
    def get_idx_in(self, fea):
        fea_batch, fea_channel, fea_height, fea_width = fea.shape
        h_crop_num = (fea_height-1) / self.crop_size +1
        w_crop_num = (fea_width-1) / self.crop_size +1
        hist_num = h_crop_num * w_crop_num
        hist_pair = torch.randint(hist_num, (fea_batch, fea_channel, hist_num, self.pair_num), device=torch.device(str_device)).long()
        return hist_pair
    def forward(self, fea_in, fea_out):
        idx_in = self.get_idx_in(fea_in)
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

mode = CustomDistance()
inputfea = torch.rand(1, 1, 8, 8, dtype=torch.double, device=torch.device(str_device), requires_grad=True)*2-1
outputfea = torch.rand(1, 1, 8, 8, dtype=torch.double, device=torch.device(str_device), requires_grad=True)*2-1
idx_in = torch.randint(4, (1, 1, 4, 2), device=torch.device(str_device)).long()
print inputfea, outputfea
infea = (inputfea, outputfea)
print mode(inputfea, outputfea)
#test = gradcheck(mode, infea, eps=1e-2, raise_exception=True)
#print test

mode = CustomDistance(cell_range=16, crop_size=16, pair_num=30, threshold=3)
inputfea = torch.rand(1, 1, 256, 256, dtype=torch.double, device=torch.device(str_device), requires_grad=True)*2-1
outputfea = torch.rand(1, 1, 256, 256, dtype=torch.double, device=torch.device(str_device), requires_grad=True)*2-1
out = mode(inputfea, outputfea)
print out
print torch.mean(out)
grad = torch.rand_like(out).double()
print out.backward(grad)
