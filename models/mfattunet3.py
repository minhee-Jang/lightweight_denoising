from audioop import mul
import os
import datetime
from reprlib import recursive_repr

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.loss.perceptual_loss import parse_perceptual_loss, PerceptualLoss
from .base_model import BaseModel
from models.convs import common
from models.common.unet import create_unet
url = {
    # 'n5m32g32n5': 'http://gofile.me/4u1bp/95jMJyKDt',
    'n3m32g32n5': 'https://www.dropbox.com/s/9gxcjy706ho4qqt/epoch_best_n0003_loss0.00032845_psnr34.8401.pth?dl=1'
}


class MFATTUNET3(BaseModel):
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
        

        parser.add_argument('--bilinear', type=str, default='bilinear',
            help='up convolution type (bilineaer or transposed2d)')

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

    @staticmethod
    def set_savedir(opt):
        dt = datetime.datetime.now()
        date = dt.strftime("%Y%m%d-%H%M")
        dataset_name = ''
        for d in opt.datasets:
            dataset_name = dataset_name + d

        model_opt = dataset_name  + "-" + date + "-" + opt.model
        model_opt = model_opt + "-n_inputs" + str(opt.n_inputs)
        model_opt = model_opt + "-ms_channels" + str(opt.ms_channels)
        model_opt = model_opt + "-growth_rate" + str(opt.growth_rate)
        model_opt = model_opt + "-n_denselayers" + str(opt.n_denselayers)
        # model_opt = model_opt + "-n_denseblocks" + str(opt.n_denseblocks)
        if opt.perceptual_loss is not None:
            model_opt = model_opt + '-perceptual_loss' + '-' + opt.perceptual_loss

        if opt.prefix != '': model_opt = opt.prefix + "-" + model_opt
        if opt.suffix != '': model_opt = model_opt + "-" + opt.suffix
        
        savedir = os.path.join(opt.checkpoints_dir, model_opt)
        return savedir

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        if opt.perceptual_loss is not None:
            self.perceptual_loss = True
            self.loss_type = opt.perceptual_loss
        else:
            self.perceptual_loss = False

        if self.perceptual_loss and self.is_train:
            self.loss_name = ['content_loss', 'style_loss']
        else:
            self.loss_name = ['content_loss']

        self.model_names = ['mfattunet3']
        self.var_name = ['x', 'out', 'target']

        # Create model
        self.mfattunet3 = create_unet2(opt).to(self.device)
        
        # Define losses and optimizers
        if self.is_train:
            if opt.content_loss == 'l1':
                #print("opt.content_loss using.....")
                self.content_loss_criterion = nn.L1Loss()
            elif opt.content_loss == 'l2':
                self.content_loss_criterion = nn.MSELoss()

            if self.perceptual_loss:
                self.perceptual_loss_criterion = PerceptualLoss(opt)

            self.optimizer_names = ['optimizerQ']
            self.optimizerQ = torch.optim.Adam(self.mfattunet3.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
            self.optimizers.append(self.optimizerQ)

        # for calculating PSNR
        self.mse_loss_criterion = nn.MSELoss()
        self.n_inputs = opt.n_inputs

        url_name = 'n{}m{}g{}n{}'.format(opt.n_inputs, opt.ms_channels, opt.growth_rate, opt.n_denselayers)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None

    def set_input(self, input):
        self.x = input['x'].to(self.device)
        self.ct =self.x[:, self.n_inputs//2:self.n_inputs//2+1]
        self.ix = torch.cat((self.x[:,:self.n_inputs//2], self.x[:,self.n_inputs//2+1:]), dim=1)
        # test_dir = r'D:/data'
        # import os
        # from skimage.io import imsave
        # recursive_dir = os.path.join(test_dir, 'test_')
        # ix1 = self.x[:,0:1].detach().to('cpu').numpy().transpose(0,2,3,1)#.squeeze()
        # ix2 = self.x[:,1:2].detach().to('cpu').numpy().transpose(0,2,3,1)#.squeeze()
        # ix3 = self.x[:,2:3].detach().to('cpu').numpy().transpose(0,2,3,1)#.squeeze()
        # ix4 = self.x[:,3:4].detach().to('cpu').numpy().transpose(0,2,3,1)#.squeeze()
        # ix5 = self.x[:,4:5].detach().to('cpu').numpy().transpose(0,2,3,1)#.squeeze()
        # recursive_path = os.path.join(recursive_dir + 'ix1.tiff')
        # recursive_path2 = os.path.join(recursive_dir + 'ix2.tiff')
        # recursive_path3 = os.path.join(recursive_dir + 'ix3.tiff')
        # recursive_path4 = os.path.join(recursive_dir + 'ix4.tiff')
        # recursive_path5 = os.path.join(recursive_dir + 'ix5.tiff')
        # imsave(recursive_path, ix1)
        # imsave(recursive_path2, ix2)
        # imsave(recursive_path3, ix3)
        # imsave(recursive_path4, ix4)
        # imsave(recursive_path5, ix5)
        # if input['target'] is not None:
        #     self.target = input['target'].to(self.device)
        #     self.ct =self.x[:, self.n_inputs//2:self.n_inputs//2+1]
        #     #self.target = input['target'].to(self.device)[:,self.n_inputs//2]
        # #print("self.target.shape: ",self.target.shape)
        # t1 = self.target[:,0:1].detach().to('cpu').numpy().transpose(0,2,3,1)#.squeeze()
        # t2 = self.target[:,1:2].detach().to('cpu').numpy().transpose(0,2,3,1)#.squeeze()
        # t3 = self.target[:,2:3].detach().to('cpu').numpy().transpose(0,2,3,1)#.squeeze()
        # t4 = self.target[:,3:4].detach().to('cpu').numpy().transpose(0,2,3,1)#.squeeze()
        # t5 = self.target[:,4:5].detach().to('cpu').numpy().transpose(0,2,3,1)#.squeeze()
        # recursive_path = os.path.join(recursive_dir + 't1.tiff')
        # recursive_path2 = os.path.join(recursive_dir + 't2.tiff')
        # recursive_path3 = os.path.join(recursive_dir + 't3.tiff')
        # recursive_path4 = os.path.join(recursive_dir + 'ix4.tiff')
        # recursive_path5 = os.path.join(recursive_dir + 'ix5.tiff')
        # imsave(recursive_path, ix1)
        # imsave(recursive_path2, ix2)
        # imsave(recursive_path3, ix3)
        # imsave(recursive_path4, ix4)
        # imsave(recursive_path5, ix5)
        print("self.ct.shape : ",self.ct.shape)
        bs, n, h, w = self.x.shape
        
           
    def forward(self):
        self.out = self.mfattunet3(self.ct.detach())          # [bs, 1, h, w]
        #print("self.out.shape :", self.out.shape)
        #test_dir = r'D:/data'
        #import os
        #from skimage.io import imsave
        #recursive_dir = os.path.join(test_dir, 'fluoroscopy_denoising_')
        #out = self.out.detach().to('cpu').numpy().transpose(0,2,3,1)#.squeeze()
        #out2 = self.x[:,1:2].detach().to('cpu').numpy().transpose(0,2,3,1)
        #out3 = self.x[:,2:3].detach().to('cpu').numpy().transpose(0,2,3,1)
        #ct = self.ct.detach().to('cpu').numpy().transpose(0,2,3,1)#.squeeze()
        #recursive_path = os.path.join(recursive_dir + 'self.tattunet3_out_ct.tiff')
        #recursive_path2 = os.path.join(recursive_dir + 'self.i_ct2.tiff')
        #recursive_path3 = os.path.join(recursive_dir + 'self.i_ct3.tiff')
        #recursive_path2 = os.path.join(recursive_dir + 'self.ct.tiff')
        #print("0000 image save----")
        #imsave(recursive_path, out)
        #imsave(recursive_path2, out2)
        #imsave(recursive_path3, out3)
        #imsave(recursive_path2, ct)
        #print("tt",t.shape)
      
    def backward(self):
        #self.artif_loss = self.content_loss_criterion(torch.mul(self.pixel_weight, self.out), torch.mul(self.pixel_weight, self.recur_out))
        if self.perceptual_loss:

            self.content_loss, self.style_loss = self.perceptual_loss_criterion(self.ct, self.out)
            print(self.content_loss)
            print("self.style loss : ", self.style_loss)
            self.loss = self.content_loss + self.style_loss #+ 1000*self.artif_loss
        else:
            self.loss = self.content_loss_criterion(self.ct, self.out)

        self.loss.backward()

        mse_loss = self.mse_loss_criterion(self.out.detach(), self.ct.detach())
        self.psnr = 10 * torch.log10(1 / mse_loss)

    def optimize_parameters(self):
        self.optimizerQ.zero_grad()
        self.forward()
        
        self.backward()
        self.optimizerQ.step()
    

    def log_loss(self, opt, phase, batch_time, iter, n_iter):
        if self.perceptual_loss:
            #print("Content Loss: {:.8f}, Style Loss: {:.8f}, Artifact Loss: {:.11f}".format(
            #    self.content_loss, self.style_loss, self.artif_loss)
            #)
            print("Content Loss: {:.8f}, Style Loss: {:.8f}".format(
                self.content_loss, self.style_loss)
            )
        print("{} {:.3f}s => Epoch[{}/{}]({}/{}): Loss: {:.8f}, PSNR: {:.5f}".format(
            phase, batch_time, opt.epoch, opt.n_epochs, iter, n_iter,
            self.loss.item(), self.psnr.item())
        )


def create_model(opt):
    return MFattUnet_model(opt)

class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv, self).__init__()
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=3//2)
        self.conv5 = nn.Conv2d(in_channels, out_channels, 5, padding=5//2)
        self.conv7 = nn.Conv2d(in_channels, out_channels, 7, padding=7//2)

    def forward(self, x):
        x3 = self.conv3(x)
        #print("x3.shape?",x3.shape)
    
        x5 = self.conv5(x)
        #print("x5.shape?",x5.shape)
    
        x7 = self.conv7(x)
        #print("x7.shape?",x7.shape)
    

        out = torch.cat((x3, x5, x7), dim=1)
        return out

class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=False):
        super(single_conv, self).__init__()
        m_body = []
        m_body.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
        if bn: m_body.append(nn.BatchNorm2d(out_ch))
        m_body.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*m_body)

        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch, 3, padding=1),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, x):
        x = self.conv(x)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            single_conv(in_channels, out_channels, bn=bn),
            single_conv(out_channels, out_channels, bn=bn)
        )
        # self.double_conv = nn.Sequential(
        #     nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(mid_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x
class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class AttU_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        print("--------------------x.shape?", x.shape)
        # encoding path
        x1 = self.Conv1(x)
        print("==================")
        print("x1.shape", x1.shape)
        x2 = self.Maxpool(x1)
        print("maxppooooo -> ", x2.shape)
        x2 = self.Conv2(x2)
        print("max-> conv2 -> x2.shape ", x2.shape)
        print("==================")
        x3 = self.Maxpool(x2)
        print("maxppooooo ->x3 ", x3.shape)
        x3 = self.Conv3(x3)
        print("max-> conv3 -> x3.shape ", x3.shape)
        print("==================")
        x4 = self.Maxpool(x3)
        print("maxppooooo ->x4 ", x4.shape)
        x4 = self.Conv4(x4)
        print("max-> conv4 -> x4.shape ", x4.shape)
        print("==================")
        print(" up start  d d fd ")
        d4 = self.Up4(x4)
        print("up x4 -> d4.shape ", d4.shape)
        print("x3 and d4 atte -> x3.shape ?", x3.shape)
        x3 = self.Att4(g=d4,x=x3)
        print("attention start ---> x3.shae", x3.shape)
        d4 = torch.cat((x3,d4),dim=1)
        print("torch cat  x3, d4, : d4.shape", d4.shape)
        d4 = self.Up_conv4(d4)
        print("self.up conv 4 d4, ", d4.shape)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x


class MFattUnet_model(nn.Module):
    def __init__(self, opt):
        super(MFattUnet_model, self).__init__()
        self.n_inputs2 = opt.n_inputs - 4 # 4개
        n_channels = opt.n_channels
        self.nc = n_channels
        bilinear = opt.bilinear
        ms_channels = opt.ms_channels
        dense_in_channels = n_channels * ms_channels * 3 * self.n_inputs2
        n_denselayers = opt.n_denselayers
        growth_rate = opt.growth_rate
        multiscale_conv = [MultiScaleConv(n_channels, ms_channels) for _ in range(self.n_inputs2)]
        self.multiscale_conv = nn.ModuleList(multiscale_conv)
        self.unet = AttU_Net(dense_in_channels, growth_rate)
        self.outc = OutConv(growth_rate, n_channels)

    def forward(self, x):
        
        assert (x.size(1) // self.nc) == self.n_inputs2
        x_mean = torch.zeros(x[:,:self.nc].shape, dtype=x.dtype, device=x.device)
        for i in range(1):
            x_mean = x_mean + x[:, i * self.nc: i * self.nc + self.nc]
        x_mean = x_mean / self.n_inputs2
        # print('x_mean.shape:', x_mean.shape)
        
        # multi-scale network
        ms_out = []
        for i in range(self.n_inputs2):
            x_in  = x[:, i * self.nc: i * self.nc + self.nc]
            ms_conv = self.multiscale_conv[i]
            ms_out.append(ms_conv(x_in))
            #print("input number : ",i)

        # densely connected network
        ms_out = torch.cat(ms_out, dim=1)
        print("ms_out.shape :", ms_out.shape)
        out = self.unet(ms_out)
        out = self.outc(out) + x_mean

        return out

def create_unet2(opt):
    return UNetModel(opt)

class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=False):
        super(single_conv, self).__init__()
        m_body = []
        m_body.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
        if bn: m_body.append(nn.BatchNorm2d(out_ch))
        m_body.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*m_body)

        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch, 3, padding=1),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, x):
        x = self.conv(x)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            single_conv(in_channels, out_channels, bn=bn),
            single_conv(out_channels, out_channels, bn=bn)
        )
        # self.double_conv = nn.Sequential(
        #     nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(mid_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetModel(nn.Module):
    def __init__(self, opt):
        super(UNetModel, self).__init__()
        n_channels = opt.n_channels
        self.nc = n_channels
        bilinear = opt.bilinear
        self.n_inputs = opt.n_inputs-1
        self.sub_mean = common.MeanShift(1.0, n_channels=n_channels)
        

        ms_channels = opt.ms_channels
        dense_in_channels = n_channels * ms_channels * 3 * self.n_inputs
        self.inc = DoubleConv(dense_in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        # self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        # self.down4 = Down(512, 1024 // factor)
        # self.up1 = Up(1024, 512 // factor, bilinear)
        # self.up2 = Up(512, 256 // factor, bilinear)
        multiscale_conv = [MultiScaleConv(n_channels, ms_channels) for _ in range(self.n_inputs)]
        #print("len(multiscale_conv):",len(multiscale_conv))
        self.multiscale_conv = nn.ModuleList(multiscale_conv)
        self.convs = nn.Sequential(
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 128),
        )

        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_channels)

        self.add_mean = common.MeanShift(1.0, n_channels=n_channels, sign=1)

    def forward(self, x):
        x_mean = torch.zeros(x[:,:self.nc].shape, dtype=x.dtype, device=x.device)
        for i in range(self.n_inputs):
            x_mean = x_mean + x[:, i * self.nc: i * self.nc + self.nc]
        x_mean = x_mean / self.n_inputs
        # print('x_mean.shape:', x_mean.shape)
        
        # multi-scale network
        ms_out = []
        for i in range(self.n_inputs):
            x_in  = x[:, i * self.nc: i * self.nc + self.nc]
            ms_conv = self.multiscale_conv[i]
            ms_out.append(ms_conv(x_in))
            #print("input number : ",i)

        # densely connected network
        ms_out = torch.cat(ms_out, dim=1)
        #x = self.sub_mean(ms_out)
        #res = x
        x1 = self.inc(ms_out)
        #print('x1.shape:', x1.shape) # 1,64, 680, 680
        x2 = self.down1(x1)
        #print('x2.shape:', x2.shape) # 1,128, 680, 680
        x3 = self.down2(x2)
        #print('x3.shape:', x3.shape)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        x = self.convs(x3)
        #print('x.shape:', x.shape)

        x = self.up1(x, x2)
        #print('up1 x.shape:', x.shape)
        x = self.up2(x, x1)
        #print('up2 x.shape:', x.shape)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        x = self.outc(x)
        #print('outc x.shape:', x.shape)
        out = x + x_mean
        #out = self.add_mean(out)
        return out
