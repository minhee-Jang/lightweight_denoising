##### BNN with GRADIENT #####

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .base_model import BaseModel
from models.convs.common import MeanShift
# from models.bilateral_filter_layer import *

# import bilateralfilter_cpu_lib
# import bilateralfilter_gpu_lib

# from .bfgradient import Gradient

class CBDNet(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--sigma_x', type=float, default=0.5)
        parser.add_argument('--sigma_y', type=float, default=0.5)
        parser.add_argument('--sigma_z', type=float, default=0.5)
        parser.add_argument('--color_sigma', type=float, default=0.01)
        parser.add_argument('--downsample_factor', type=int, default=2)

        return parser

      

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.model_names = ['net']
        self.loss_name = ['loss']
        self.var_name = ['x', 'out', 'target', 'noise_level_est']

        self.n_frames = opt.n_frames

        # Create model
        self.net = create_model(opt).to(self.device)
        print("parameter 갯수 : ", sum(p.numel() for p in self.net.parameters() if p.requires_grad))

        # Define losses and optimizers
        if self.is_train:
            self.mse_loss_criterion = nn.MSELoss()
            self.optimizer_names = ['optimizer']
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
            self.optimizers.append(self.optimizer)


    def set_input(self, input):
        self.x = input['lr'].to(self.device)
        self.x = self.x.squeeze(dim=2)
        #print(self.x.shape)
        if 'hr' in input:
            self.target = input['hr'].to(self.device)  #[bs, c, n, h, w]
            self.target = self.target.squeeze(dim=2)
        
    def forward(self):
        self.noise_level_est, self.out = self.net(self.x)

    def backward(self):
        self.loss.backward()
        

    def calc_loss(self):
        f_loss = fixed_loss()
        # self.loss_image = self.mse_loss_criterion(self.target, self.out)
        self.loss = f_loss(self.out, self.target, self.noise_level_est, 0, 0)
        self.psnr = 10 * torch.log10(1 / self.loss)
        # self.psnr_image = 10 * torch.log10(1 / self.loss_image)

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        self.calc_loss()
        self.backward()
        self.optimizer.step()

    def get_logs(self):
        log_dict = {
            'loss': '{:.8f}'.format(self.loss),
            'psnr': '{:.8f}'.format(self.psnr),
            # 'psnr_image': '{:.8f}'.format(self.psnr)
        }
        return log_dict

    def get_batch_measure(self):
        return self.loss.detach(), self.psnr.detach()


    def predict(self, batch):
        n_frames = self.n_frames
        x = batch['lr']
        b, c, n, h, w = x.shape # here n is the number of whole images in the video

        predicted_video = []
        predicted_idxs = []

        for d in range(n):
            predicted_idx = d

            if n-d < n_frames:
                xd = x[:, :, -1*n_frames:]
            else:
                xd = x[:, :, d:d+n_frames]
            
            tensors_input = {
                "lr": xd,
            }

            with torch.no_grad():
                self.set_input(tensors_input)
                self.test()
            
            ##########
            #bo, co, ho, wo, no = self.out.shape
            # print('self.out.shape: ', self.out.shape)
            #out = self.out.view(bo, co, no, ho, wo)
            # print('out.shape1: ', out.shape)
            out = self.out.detach()
            out = out.unsqueeze(2)
            #print('out.shape2: ', out.shape)
            #rint('predicted file {:03d}'.format(predicted_idx))
            predicted_video.append(out)
            predicted_idxs.append(predicted_idx)

        predicted_video = torch.cat(predicted_video, dim=2)
        return predicted_video, predicted_idxs

def create_model(opt):
        return CBDNetModel(opt)

class fixed_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, out_image, gt_image, est_noise, gt_noise, if_asym):
        h_x = est_noise.size()[2]
        w_x = est_noise.size()[3]
        count_h = self._tensor_size(est_noise[:, :, 1:, :])
        count_w = self._tensor_size(est_noise[:, :, : ,1:])
        h_tv = torch.pow((est_noise[:, :, 1:, :] - est_noise[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((est_noise[:, :, :, 1:] - est_noise[:, :, :, :w_x-1]), 2).sum()
        tvloss = h_tv / count_h + w_tv / count_w

        loss = torch.mean(torch.pow((out_image - gt_image), 2)) + \
                if_asym * 0.5 * torch.mean(torch.mul(torch.abs(0.3 - F.relu(gt_noise - est_noise)), torch.pow(est_noise - gt_noise, 2))) + \
                0.05 * tvloss
        return loss

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]



class CBDNetModel(nn.Module):
    def __init__(self, opt):
        super(CBDNetModel, self).__init__()
        n_channels = opt.n_channels
        self.fcn = FCN(opt)
        self.unet = UNet(opt)
        self.sub_mean = MeanShift(1.0, n_channels=n_channels)
        self.add_mean = MeanShift(1.0, n_channels=n_channels, sign=1)

    
    def forward(self, x):
        x = self.sub_mean(x)
        noise_level = self.fcn(x)
        concat_img = torch.cat([x, noise_level], dim=1)
        out = self.unet(concat_img) + x
        out = self.add_mean(out)
        return noise_level, out


class FCN(nn.Module):
    def __init__(self, opt):
        super(FCN, self).__init__()
        c = opt.n_channels
        self.inc = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.outc = nn.Sequential(
            nn.Conv2d(32, c, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        conv1 = self.inc(x)
        conv2 = self.conv(conv1)
        conv3 = self.conv(conv2)
        conv4 = self.conv(conv3)
        conv5 = self.outc(conv4)
        return conv5


class UNet(nn.Module):
    def __init__(self, opt):
        super(UNet, self).__init__()
        c = opt.n_channels
        
        self.inc = nn.Sequential(
            single_conv(c*2, 64),
            single_conv(64, 64)
        )

        self.down1 = nn.AvgPool2d(2)
        self.conv1 = nn.Sequential(
            single_conv(64, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.down2 = nn.AvgPool2d(2)
        self.conv2 = nn.Sequential(
            single_conv(128, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256)
        )

        self.up1 = up(256)
        self.conv3 = nn.Sequential(
            single_conv(128, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.up2 = up(128)
        self.conv4 = nn.Sequential(
            single_conv(64, 64),
            single_conv(64, 64)
        )

        self.outc = outconv(64, c)

    def forward(self, x):
        inx = self.inc(x)

        down1 = self.down1(inx)
        conv1 = self.conv1(down1)

        down2 = self.down2(conv1)
        conv2 = self.conv2(down2)

        up1 = self.up1(conv2, conv1)
        conv3 = self.conv3(up1)

        up2 = self.up2(conv3, inx)
        conv4 = self.conv4(up2)

        out = self.outc(conv4)
        return out


class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = x2 + x1
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
