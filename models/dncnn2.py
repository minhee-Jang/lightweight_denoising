"""
Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising
"""
import os
import datetime
import torch
import torch.nn as nn

from .base_model import BaseModel


class DnCNN2(BaseModel):
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
        parser.add_argument('--n_feats', type=int, default=64,
            help='number of feature maps')
        parser.add_argument('--kernel_size', type=int, default=3,
            help='kernel size of convolution layer')
        parser.add_argument('--n_layers', type=int, default=17,
            help='number of dncnn layers')
        parser.set_defaults(n_frames=1)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.n_frames = opt.n_frames
        self.model_names = ['net']
        self.net = create_model(opt).to(self.device)
        
        # Define losses and optimizers
        if self.is_train:
            self.loss_criterion = nn.MSELoss()
            self.mse_loss_criterion = nn.MSELoss()
            self.optimizer_names = ['optimizer']
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
            self.optimizers.append(self.optimizer)

    def set_input(self, input):
        self.x = input['lr'].to(self.device)
        
        bs, c, n, h, w = self.x.shape
        self.x = self.x.view(bs, c*n, h, w)
        
        if 'hr' in input:
            self.target = input['hr'].to(self.device).view(bs, c*n, h, w)
        from thop import profile
        print("########flops")
        flops, _ = profile(self.net, inputs=(self.x,))
        print(f"FLOPS: {flops}")


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
        mse_loss = self.mse_loss_criterion(self.target.detach(), self.out.detach())
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

        b, c, n, h, w = x.shape # here n is the number of whole images in the video

        predicted_video = []

        for d in range(n):
            xd = x[:, :, d:d+n_frames]

            tensors_input = {
                "lr": xd,
            }
     
            with torch.no_grad():
                self.set_input(tensors_input)
                self.test()
            out = self.out
            predicted_video.append(out.unsqueeze(dim=2))

        predicted_video = torch.cat(predicted_video, dim=2)
        return predicted_video

def create_model(opt):
    return DnCNNModel(opt)


class DnCNNModel(nn.Module):
    def __init__(self, opt):
        super(DnCNNModel,self).__init__()
        
        kernel_size = opt.kernel_size
        padding = kernel_size // 2
        features = opt.n_feats
        num_of_layers = opt.n_layers
        layers = []

        n_channels = opt.n_channels

        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        
        res = self.dncnn(x)
        return x - res
