import os
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.loss.perceptual_loss import parse_perceptual_loss, PerceptualLoss
from .base_model import BaseModel

import numpy as np

class MFCNN(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # Network parameters
        # n_inputs is set in base options
        # n_channels is set in dataset options
        parser.add_argument('--ms_channels', type=int, default=32,
            help='number of features of output in multi-scale convolution')
        parser.add_argument('--growth_rate', type=int, default=32,
            help='growth rate of each layer in dense block')
        parser.add_argument('--n_denselayers', type=int, default=5,
            help='number of layers in dense block')
        # n_denseblocks is currently is not used
        # parser.add_argument('--n_denseblocks', type=int, default=8,
        #     help='number of layers of dense blocks')

        # n_denseblocks = opt.n_denseblocks # Righit now, we use one dense block

        parser.add_argument('--perceptual_loss', type=str, default=None,
            choices=['srgan', 'wavelet_transfer', 'perceptual_loss'],
            help='specity loss_type')

        if is_train:
            parser = parse_perceptual_loss(parser)
        else:
            parser.set_defaults(test_patches=False)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        if opt.perceptual_loss is not None:
            self.perceptual_loss = True
            self.loss_type = opt.perceptual_loss
        else:
            self.perceptual_loss = False

        self.model_names = ['qenet']
        self.n_frames = opt.n_frames

        # Create model
        self.qenet = create_model(opt).to(self.device)
        
        # Define losses and optimizers
        if self.is_train:
            if opt.content_loss == 'l1':
                self.loss_criterion = nn.L1Loss()
            elif opt.content_loss == 'l2':
                self.loss_criterion = nn.MSELoss()

            if self.perceptual_loss:
                self.loss_criterion = PerceptualLoss(opt)

            self.optimizer_names = ['optimizerQ']
            self.optimizerQ = torch.optim.Adam(self.qenet.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
            self.optimizers.append(self.optimizerQ)

        # for calculating PSNR
        self.mse_loss_criterion = nn.MSELoss()

    def set_input(self, input):
        # assert len(input) == 1

        self.x = input['lr'].to(self.device)
        # b, c, n, h, w = self.x.shape
        # self.x = self.x.view(b, c*n, h, w)
        if 'hr' in input:
            self.target = input['hr'].to(self.device)[:, :, self.n_frames//2]

    def forward(self):
        self.out = self.qenet(self.x)

    def backward(self):
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizerQ.zero_grad()
        self.forward()
        self.calc_loss()
        self.backward()
        self.optimizerQ.step()

    def calc_loss(self):
        if self.perceptual_loss:
            print(self.target.shape)
            print(self.out.ndim)
            self.content_loss, self.style_loss = self.loss_criterion(self.target, self.out)
            self.loss = self.content_loss + self.style_loss
        else:
            self.loss = self.loss_criterion(self.target, self.out)
        mse_loss = self.mse_loss_criterion(self.out.detach(), self.target.detach())
        self.psnr = 10 * torch.log10(1 / mse_loss)

    def get_logs(self):
        if self.perceptual_loss:
            log_dict = {
            'loss': '{:.8f}'.format(self.loss),
            'content_loss': '{:.8f}'.format(self.content_loss),
            'style_loss': '{:.8f}'.format(self.style_loss),
            # 'mse_loss': '{:.8f}'.format(mse_loss),
            'psnr': '{:.8f}'.format(self.psnr)
        }
        else:
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
            # print('d: {}, predicted_idx: {}'.format(d, predicted_idx))
            if predicted_idx % 10 != 0:
                continue

            if d < n_frames//2:
                n_dummy_vids = n_frames//2 - d
                dummy_x =  x.repeat(1, 1, n_dummy_vids, 1, 1)
                xd = x[:, :, :d + n_frames//2 + 1]
                xd = torch.cat((dummy_x, xd), dim=2)
            elif d >= n - n_frames//2:
                n_dummy_vids = n_frames//2 - (n - d) + 1
                xd = x[:, :, d - n_frames//2:]
                dummy_x =  x.repeat(1, 1, n_dummy_vids, 1, 1)
                xd = torch.cat((xd, dummy_x), dim=2)
            else:
                xd = x[:, :, d-n_frames//2:d+n_frames//2+1]
            
            tensors_input = {
                "lr": xd,
            }

            with torch.no_grad():
                self.set_input(tensors_input)
                self.test()

            out = self.out.detach()
            out = out.permute(0, 2, 3, 1).squeeze().to('cpu').numpy()

            out[out > 1.0] = 1.0
            out[out < 0.0] = 0.
            out = out * 256
            out = np.rint(out)
            out  = out.astype(np.uint8)

            print('predicted file {:03d}'.format(predicted_idx))
            predicted_video.append(out)
            predicted_idxs.append(predicted_idx)

        return predicted_video, predicted_idxs

def create_model(opt):
    return QENet(opt)

class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv, self).__init__()
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=3//2)
        self.conv5 = nn.Conv2d(in_channels, out_channels, 5, padding=5//2)
        self.conv7 = nn.Conv2d(in_channels, out_channels, 7, padding=7//2)

    def forward(self, x):
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)

        out = torch.cat((x3, x5, x7), dim=1)
        return out

class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)

class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        # print('x.shape:', x.shape)
        # print('self.layers(x).shape:', self.layers(x).shape)
        # print('self.lff(self.layers(x)).shape', self.lff(self.layers(x)).shape)
        # return x + self.lff(self.layers(x))  # local residual learning
        return self.lff(self.layers(x))


class RDN(nn.Module):
    def __init__(self, num_channels, num_features, growth_rate, num_blocks, num_layers):
    # def __init__(self, opt):
        super(RDN, self).__init__()

        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )

        self.output = nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)

    def forward(self, x):
        global_res = x
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        # x = self.upscale(x)
        x = self.output(x) + global_res
        return x


class QENet(nn.Module):
    def __init__(self, opt):
        super(QENet, self).__init__()
        self.n_frames = opt.n_frames
        n_channels = opt.n_channels
        self.nc = n_channels
        ms_channels = opt.ms_channels # number of channels of output multi-scale conv

        dense_in_channels = ms_channels * 3 * self.n_frames

        n_denselayers = opt.n_denselayers
        # n_denseblocks = opt.n_denseblocks # Righit now, we use one dense block
        growth_rate = opt.growth_rate

        multiscale_conv = [MultiScaleConv(n_channels, ms_channels) for _ in range(self.n_frames)]

        self.multiscale_conv = nn.ModuleList(multiscale_conv)
        self.rdn = RDB(dense_in_channels, growth_rate, n_denselayers)
        self.tail = nn.Conv2d(growth_rate, n_channels, 3, padding=3//2)
        
    def forward(self, x):
        # assert (x.size(1) // self.nc) == self.n_frames

        # if x-ray has statics images, we will take mean of images
        # x_mean = torch.zeros(x[:,:,0].shape, dtype=x.dtype, device=x.device)
        # for i in range(self.n_frames):
        #     x_mean = x_mean + x[:, :, i]
        # x_mean = x_mean / self.n_frames
        # print('x_mean.shape:', x_mean.shape)
        x_mean = x[:, :, self.n_frames//2]
        
        # multi-scale network
        ms_out = []
        for i in range(self.n_frames):
            x_in  = x[:, :, i]
            ms_conv = self.multiscale_conv[i]
            ms_out.append(ms_conv(x_in))

        # densely connected network
        ms_out = torch.cat(ms_out, dim=1)
        out = self.rdn(ms_out)
        out = self.tail(out) + x_mean

        return out
