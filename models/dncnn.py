import os
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel

import numpy as np

class dncnn(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(n_frames=5)
        
        return parser
    
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        
        self.model_names = ['mcdncnn']
        self.n_frames = opt.n_frames
        
        self.net = create_model(opt).to(self.device)
        
        if self.is_train:
            self.loss_criterion = nn.MSELoss()

            self.optimizer_names = ['optimizerQ']
            self.optimizerQ = torch.optim.Adam(self.cnn.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
            self.optimizers.append(self.optimizerQ)

        # for calculating PSNR
        self.mse_loss_criterion = nn.MSELoss()

    def set_input(self, input):
        self.x = input['lr'].to(self.device)
        if 'hr' in input:
            self.target = input['hr'].to(self.device)[:, :, self.n_frames-1]
        
        from thop import profile
        print("########flops")
        flops, _ = profile(self.net, inputs=(self.x,))
        print(f"FLOPS: {flops}")

        

    def forward(self):
        self.out = self.net(self.x)
        
    def backward(self):
        self.loss.backward()
        
    def optimize_parameters(self):
        self.optimizerQ.zero_grad()
        self.forward()
        self.calc_loss()
        self.backward()
        self.optimizerQ.step()
        
    def calc_loss(self):
        self.loss = self.loss_criterion(self.target, self.out)
        mse_loss = self.mse_loss_criterion(self.out.detach(), self.target.detach())
        self.psnr = 10 * torch.log10(1 / mse_loss)
        
    def get_logs(self):
        log_dict = {
            'loss': '{:.8f}'.format(self.loss),
            'psnr': '{:.8f}'.format(self.psnr)
        }
        return log_dict
    
    def get_batch_measure(self):
        return self.loss.detach(), self.psnr.detach()

    def predict(self, batch):
        n_frames = self.n_frames
        x = batch['lr']
        _, c, n, h, w = x.shape # here n is the number of whole images in the video

        predicted_video = []
        predicted_idxs = []

        for d in range(n):
            predicted_idx = d + 1
            
            if d < n_frames-1:
                n_dummy_vids = n_frames-1 - d
                dummy_x =  x.repeat(1, 1, n_dummy_vids, 1, 1)
                xd = x[:, :, :d + n_frames]
                xd = torch.cat((dummy_x, xd), dim=2)
            elif d >= n - n_frames-1:
                n_dummy_vids = n_frames - (n - d)
                xd = x[:, :, d - n_frames-1:]
                dummy_x =  x.repeat(1, 1, n_dummy_vids, 1, 1)
                xd = torch.cat((xd, dummy_x), dim=2)
            else:
                xd = x[:, :, d-n_frames-1:d+n_frames]
            
            tensors_input = {
                "lr": xd,
            }

            with torch.no_grad():
                self.set_input(tensors_input)
                self.test()
            
            out = self.out.detach()
            out = out.unsqueeze(2)
            print('predicted file {:03d}'.format(predicted_idx))
            predicted_video.append(out)
            predicted_idxs.append(predicted_idx)

        predicted_video = torch.cat(predicted_video, dim=2)
        return predicted_video, predicted_idxs

def create_model(opt):
    return DenseDnCNN(opt)

class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.bn(self.conv3(self.conv1(x))))], 1)


class DenseBlock(nn.Module):
    def __init__(self, opt, in_channels, num_layers=4):
        super(DenseBlock, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels, in_channels) for i in range(num_layers)])

    def forward(self, x):
        return self.layers(x)
    

class DenseDnCNN(nn.Module):
    def __init__(self, opt):
        super(DenseDnCNN, self).__init__()
        self.n_frames = opt.n_frames
        n_channels = opt.n_channels
        
        self.db = DenseBlock(n_channels)
        self.tail = nn.Conv2d(n_channels*self.n_frames, n_channels, 3, padding=3//2)

    def forward(self, x):
        x_res = x[:, :, self.n_frames-1]

        x_in = x[:, :, 0]
        x_out = self.db(x_in)
        
        for i in range(1, self.n_frames):
            x_in = x[:, :, i]
            x_out.append(self.db(x_in))

        xs_out = torch.cat(x_out, dim=1)
        out = self.tail(xs_out) + x_res

        return out
