##### BNN with GRADIENT #####

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init
from .base_model import BaseModel
from models.convs.common import MeanShift
from models.common.unet import create_unet
from torchvision.models import vgg19

# from models.bilateral_filter_layer import *

# import bilateralfilter_cpu_lib
# import bilateralfilter_gpu_lib

# from .bfgradient import Gradient

class WganVGG(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--n_d_train', type=int, default=4,
            help='number of discriminator training')
        parser.add_argument('--generator', type=str, default='unet',
            help='generator model [unet | original]')
        parser.add_argument('--perceptual', dest='perceptual', action='store_true',
            help='use perceptual loss')
        parser.add_argument('--mse', dest='perceptual', action='store_false',
            help='use MSE loss')
        parser.add_argument('--bilinear', type=str, default='bilinear')

        if is_train:
            parser.set_defaults(perceptual=True)
            parser.set_defaults(lr=1e-4)
            parser.set_defaults(b1=0.5)
            parser.set_defaults(b2=0.999)
        return parser

      

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_name = ['loss_d', 'loss_g', 'loss_p', 'loss']
        self.var_name = ['x', 'out', 'target']
        self.n_frames = opt.n_frames
        # Create model
        # self. = create_model(opt).to(self.device)
        self.net_G = create_G(opt).to(self.device)
        self.mse_loss_criterion = nn.MSELoss()
        
        # Define losses and optimizers
        if self.is_train:
            self.net_D = create_D(opt).to(self.device)
            self.feature_extractor = create_vgg().to(self.device)
            self.model_names = ['net_G', 'net_D', 'feature_extractor']
            
            self.perceptual_criterion = nn.MSELoss()
            self.mse_loss_criterion = nn.MSELoss()
            self.optimizer_names = ['optimizer_G', 'optimizer_D']
            self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
            self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.n_d_train = opt.n_d_train
            self.gp = True
            self.perceptual = opt.perceptual
        else:
            self.model_names = ['net_G']
        
        # print("parameter 갯수 : ", sum(p.numel() for p in self.net_G.parameters() if p.requires_grad))
        # print("parameter 갯수 : ", sum(p.numel() for p in self.net_D.parameters() if p.requires_grad))
        # print("parameter 갯수 : ", sum(p.numel() for p in self.feature_extractor.parameters() if p.requires_grad))



    def set_input(self, input):
        self.x = input['lr'].to(self.device)
        self.x = self.x.squeeze(dim=2)
        #print(self.x.shape)
        if 'hr' in input:
            self.target = input['hr'].to(self.device)  #[bs, c, n, h, w]
            self.target = self.target.squeeze(dim=2)
        


    def forward(self):
        self.out = self.net_G(self.x)

    def gp_loss(self, y, fake, lambda_=10):
        assert y.size() == fake.size()
        a = torch.FloatTensor(np.random.random((y.size(0), 1, 1, 1))).to(self.device)
        interp = (a*y + ((1-a)*fake)).requires_grad_(True)
        d_interp = self.net_D(interp)
        fake_ = torch.FloatTensor(y.shape[0], 1).fill_(1.0).requires_grad_(False).to(self.device)
        gradients = torch.autograd.grad(
            outputs=d_interp, inputs=interp, grad_outputs=fake_,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) -1)**2).mean() * lambda_
        return gradient_penalty
        
    def p_loss(self, x, y):
        fake = x.repeat(1, 3, 1, 1)
        real = y.repeat(1, 3, 1, 1)
        fake_feature = self.feature_extractor(fake)
        real_feature = self.feature_extractor(real)
        loss = self.perceptual_criterion(fake_feature, real_feature)
        return loss



    # def backward(self):
    #     self.loss.backward()
    def backward_D(self):
        fake = self.out
        d_real = self.net_D(self.target)
        d_fake = self.net_D(fake)
        loss = -torch.mean(d_real) + torch.mean(d_fake)
        loss_gp = self.gp_loss(self.target, fake) if self.gp else 0

        self.loss_d = loss + loss_gp
        self.loss_d.backward()  

        mse_loss = self.mse_loss_criterion(self.out.detach(), self.target.detach())
        self.psnr = 10 * torch.log10(1 / mse_loss)
        self.loss = mse_loss
        
    def backward_G(self):
        fake = self.out
        d_fake = self.net_D(fake)
        loss = -torch.mean(d_fake)
        # loss = 0
        loss_p = self.p_loss(self.out, self.target) if self.perceptual else self.mse_loss_criterion(self.out, self.target)

        self.loss_g = loss + loss_p
        self.loss_g.backward()

    def calc_loss(self):
        mse_loss = self.mse_loss_criterion(self.out.detach(), self.target.detach())
        self.psnr = 10 * torch.log10(1 / mse_loss)
        self.loss = mse_loss

    def optimize_parameters(self):
        self.set_requires_grad([self.net_D], True)
        for _ in range(self.n_d_train):
            self.optimizer_D.zero_grad()
            # self.net_D.zero_grad()
            self.forward()
            self.backward_D()
            self.optimizer_D.step()

        self.set_requires_grad([self.net_D], False)
        self.optimizer_G.zero_grad()
        # self.net_G.zero_grad()
        self.forward()
        self.backward_G()
        self.optimizer_G.step()

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

def create_G(opt):
    if opt.generator == 'unet':
        generator = create_unet(opt)
    elif opt.generator == 'original':
        generator = WGAN_VGG_generator(opt)
    return generator

def create_D(opt):
    return WGAN_VGG_discriminator(opt)

def create_vgg():
    return WGAN_VGG_FeatureExtractor()

class WGAN_VGG_generator(nn.Module):
    def __init__(self, opt):
        super(WGAN_VGG_generator, self).__init__()
        n_channels = opt.n_channels
        layers = [nn.Conv2d(n_channels,32,3,1,1), nn.ReLU()]
        for i in range(2, 8):
            layers.append(nn.Conv2d(32,32,3,1,1))
            layers.append(nn.ReLU())
        layers.extend([nn.Conv2d(32,n_channels,3,1,1), nn.ReLU()])
        self.net = nn.Sequential(*layers)

        self.sub_mean = common.MeanShift(1.0, n_channels=n_channels)
        self.add_mean = common.MeanShift(1.0, n_channels=n_channels, sign=1)

    def forward(self, x):
        x = self.sub_mean(x)
        out = self.net(x)
        out = self.add_mean(out)
        return out


class WGAN_VGG_discriminator(nn.Module):
    def __init__(self, opt):
        super(WGAN_VGG_discriminator, self).__init__()
        input_size = opt.patch_size
        n_channels = opt.n_channels
        def conv_output_size(input_size, kernel_size_list, stride_list):
            n = (input_size - kernel_size_list[0]) // stride_list[0] + 1
            for k, s in zip(kernel_size_list[1:], stride_list[1:]):
                n = (n - k) // s + 1
            return n

        def add_block(layers, ch_in, ch_out, stride):
            layers.append(nn.Conv2d(ch_in, ch_out, 3, stride, 0))
            layers.append(nn.LeakyReLU())
            return layers

        layers = []
        ch_stride_set = [(n_channels,64,1),(64,64,2),(64,128,1),(128,128,2),(128,256,1),(256,256,2)]
        for ch_in, ch_out, stride in ch_stride_set:
            add_block(layers, ch_in, ch_out, stride)

        self.output_size = conv_output_size(input_size, [3]*6, [1,2]*3)
        self.net = nn.Sequential(*layers)
        self.fc1 = nn.Linear(256*self.output_size*self.output_size, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        out = self.net(x)
        out = out.view(-1, 256*self.output_size*self.output_size)
        out = self.lrelu(self.fc1(out))
        out = self.fc2(out)
        return out


class WGAN_VGG_FeatureExtractor(nn.Module):
    def __init__(self):
        super(WGAN_VGG_FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        network = nn.Sequential(*list(vgg19_model.features.children())[:35]).eval()
        for param in network.parameters():
            param.requires_grad = False

        self.feature_extractor = network

    def forward(self, x):
        out = self.feature_extractor(x)
        return out


class WGAN_VGG(nn.Module):
    # referred from https://github.com/kuc2477/pytorch-wgan-gp
    def __init__(self, opt):
        super(WGAN_VGG, self).__init__()
        input_size = opt.patch_size
        self.generator = WGAN_VGG_generator(opt)
        self.discriminator = WGAN_VGG_discriminator(opt)
        self.feature_extractor = WGAN_VGG_FeatureExtractor()
        self.p_criterion = nn.L1Loss()
        

    def d_loss(self, x, y, gp=True, return_gp=False):
        fake = self.generator(x)
        d_real = self.discriminator(y)
        d_fake = self.discriminator(fake)
        d_loss = -torch.mean(d_real) + torch.mean(d_fake)
        if gp:
            gp_loss = self.gp(y, fake)
            loss = d_loss + gp_loss
        else:
            gp_loss = None
            loss = d_loss
        return (loss, gp_loss) if return_gp else loss

    def g_loss(self, x, y, perceptual=True, return_p=False):
        fake = self.generator(x)
        d_fake = self.discriminator(fake)
        g_loss = -torch.mean(d_fake)
        if perceptual:
            p_loss = self.p_loss(x, y)
            loss = g_loss + (0.1 * p_loss)
        else:
            p_loss = None
            loss = g_loss
        return (loss, p_loss) if return_p else loss

    def p_loss(self, x, y):
        fake = self.generator(x).repeat(1,3,1,1)
        real = y.repeat(1,3,1,1)
        fake_feature = self.feature_extractor(fake)
        real_feature = self.feature_extractor(real)
        loss = self.p_criterion(fake_feature, real_feature)
        return loss

    def gp(self, y, fake, lambda_=10):
        assert y.size() == fake.size()
        a = torch.cuda.FloatTensor(np.random.random((y.size(0), 1, 1, 1)))
        interp = (a*y + ((1-a)*fake)).requires_grad_(True)
        d_interp = self.discriminator(interp)
        fake_ = torch.cuda.FloatTensor(y.shape[0], 1).fill_(1.0).requires_grad_(False)
        gradients = torch.autograd.grad(
            outputs=d_interp, inputs=interp, grad_outputs=fake_,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) -1)**2).mean() * lambda_
        return gradient_penalty



def create_model(opt):
    return UNetModel(opt)