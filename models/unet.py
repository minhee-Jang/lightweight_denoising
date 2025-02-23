"""
U-Net: Convolutional Networks for Biomedical Image Segmentation
U-Net architecture is slightly different depending on each implementation.
I implemented U-Net based of the following paper:
'A performance comparison of convolutional neural network-based imagedenoising methods: The effect of loss functions on low-dose CT images'
Byeongjoon Kim, Minah Han, Hyunjung Shima), and Jongduk Baek
"""
# import os
# import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.convs import common
from models.common.unet import create_unet

from .base_model import BaseModel

class UNet(BaseModel):
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
        parser.add_argument('--bilinear', type=str, default='bilinear',
            help='up convolution type (bilineaer or transposed2d)')
        if is_train:
            parser.add_argument('--content_loss', type=str, choices=['l1', 'l2'], default='l2',
                help='loss function (l1, l2)')

        parser.set_defaults(n_frames=1)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.model_names = ['net']

        # Create model
        self.net = create_unet(opt).to(self.device)
        self.mse_loss_criterion = nn.MSELoss()
        
        # Define losses and optimizers
        if self.is_train:
            if opt.content_loss == 'l1':
                self.loss_criterion = nn.L1Loss()
            elif opt.content_loss == 'l2':
                self.loss_criterion = nn.MSELoss()

            self.optimizer_names = ['optimizer']
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
            self.optimizers.append(self.optimizer)

    def set_input(self, input):
        self.x = input['lr'].to(self.device)
        assert self.x.shape[2] == 1, 'please set --n_frames 1'

        bs, c, n, h, w = self.x.shape
        self.x = self.x.view(bs, c*n, h, w)
        if 'hr' in input:
            self.target = input['hr'].to(self.device).view(bs, c*n, h, w)


    def forward(self):
        self.out = self.net(self.x)

    def backward(self):
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        self.calc_loss()
        self.backward()
        self.optimizer.step()

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
        x = batch['lr']
        _, c, n, h, w = x.shape # here n is the number of whole images in the video

        predicted_imgs = []
        predicted_idxs = []

        for d in range(n):
            predicted_idx = d + 1
            # print('d: {}, predicted_idx: {}'.format(d, predicted_idx))
            xd = x[:, :, d:d+1]
            
            tensors_input = {
                "lr": xd,
            }

            # print('xd.shape:', xd.shape)
            with torch.no_grad():
                self.set_input(tensors_input)
                self.test()

            out = self.out.unsqueeze(2)
            # print('out.shape:', out.shape)
            predicted_imgs.append(out)
            predicted_idxs.append(predicted_idx)

        predicted_imgs = torch.cat(predicted_imgs, dim=2)
        return predicted_imgs, predicted_idxs