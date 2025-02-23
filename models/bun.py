##### BNN and UNet with GRADIENT #####

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from models.convs import common

import bilateralfilter_cpu_lib
import bilateralfilter_gpu_lib

from .gdmasking import GradientMask

class BUN(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--sigma_x', type=float, default=0.5)
        parser.add_argument('--sigma_y', type=float, default=0.5)
        parser.add_argument('--sigma_z', type=float, default=0.5)
        parser.add_argument('--color_sigma', type=float, default=0.01)
        parser.add_argument('--downsample_factor', type=int, default=2)
        
        parser.add_argument('--bilinear', type=str, default='bilinear',
            help='up convolution type (bilineaer or transposed2d)')

        if is_train:
            parser.set_defaults(n_frames=5)
        else:
            parser.set_defaults(test_patches=False)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.model_names = ['bf3d', 'unet']
        self.n_frames = opt.n_frames
 
        # Create model
        self.bf3d = create_model1(opt).to(self.device)
        self.unet = create_model2(opt).to(self.device)
        
        # Define losses and optimizers
        if self.is_train:
            self.loss_criterion = nn.MSELoss()

            # self.optimizer_names = ['optimizer_B', 'optimizer_U']
            self.optimizer_names = ['optimizer_spat', 'optimizer_color', 'optimizer_U']
            
            # self.optimizer_B = torch.optim.Adam(self.bf3d.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
            self.optimizer_U = torch.optim.Adam(self.unet.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
            
            self.optimizer_spat = torch.optim.Adam([parameter for name, parameter in self.bf3d.named_parameters()
                                                if name.endswith(".sigma_x") or name.endswith(".sigma_y") or name.endswith(".sigma_z")], lr=1e-2)
            self.optimizer_color = torch.optim.Adam([parameter for name, parameter in self.bf3d.named_parameters()
                                                 if name.endswith(".color_sigma")], lr=5e-3)

            
            self.optimizer = []
            # self.optimizers.append(self.optimizer_B)
            self.optimizers.append(self.optimizer_spat)
            self.optimizers.append(self.optimizer_color)
            self.optimizers.append(self.optimizer_U)

        # for calculating PSNR
        self.mse_loss_criterion = nn.MSELoss()

    def set_input(self, input):
        self.x = input['lr'].to(self.device)
        b, c, n, h, w = self.x.shape

        self.x_B = self.x.view(b, c, h, w, n)    #[32, 1, 120, 120, 5]
        self.x_U = self.x[:, :, 0, :, :]    #[32, 1, 120, 120]
        
        if 'hr' in input:
            self.target = input['hr'].to(self.device)
            self.target_B = self.target.view(b, c, h, w, n)    #[32, 1, 120, 120, 5]
            self.target_U = self.target[:, :, 0, :, :]    #[32, 1, 120, 120]

    def forward(self):
        self.out_B = self.bf3d(self.x_B)
        self.out_U = self.unet(self.x_U)
        
        for name, param in self.bf3d.named_parameters():
            print(name, param)

    def backward_B(self):
        self.loss_B.backward()
        
    def backward_U(self):
        self.loss_U.backward()

    def optimize_parameters(self):
        with torch.autograd.set_detect_anomaly(True):
            # self.optimizer_B.zero_grad()
            self.optimizer_spat.zero_grad()
            self.optimizer_color.zero_grad()
            self.optimizer_U.zero_grad()
            self.forward()
            self.calc_loss()
            self.backward_B()
            # self.optimizer_B.step()
            self.optimizer_spat.step()
            self.optimizer_color.step()
            self.backward_U()
            self.optimizer_U.step()

    def masking(self):
        gradient_lapE = GradientMask().to('cuda')
        bx, cx, hx, wx, nx = self.x_B.shape
        self.x2 = self.x_B.view(bx, cx, nx, hx, wx).clone() #[32, 1, 5, 120, 120]
        self.mask_B, self.mask_U = gradient_lapE(self.x2) #[32, 1, 5, 120, 120]
        
        self.ob = self.out_B.view(bx, cx, nx, hx, wx).clone()    #[32, 1, 5, 120, 120]
        self.out2b = self.ob.clone()
        self.out2b.masked_fill_(self.mask_B, 0.0) #[32, 1, 5, 120, 120]

        self.ou = self.out_U.clone()   #[32, 1, 120, 120]      
        self.mask_U = self.mask_U[:, :, 0, :, :].clone()
        self.out2u = self.ou.clone()
        self.out2u.masked_fill_(self.mask_U, 0.0)
        
        self.oB1 = self.out2b[:, :, 0, :, :].clone()
        self.oU1 = self.out2u.clone()
        self.out = self.oB1 + self.oU1
        
        # import os
        # import imageio
        # pppath = "./hahahaha01"
        # os.makedirs(pppath, exist_ok=True)
        # fmt = '.tiff'
        # filename = "1"
        # out_fn_pathm2 = os.path.join(pppath, filename + "_m2_" + fmt)
        # out_fn_pathm1 = os.path.join(pppath, filename + "_m1_" + fmt)
        # mkm = self.mask_B[0, :, :, 0].squeeze(dim=1)
        # mkm2 = self.mask_U[0, :, :, :].squeeze(dim=1)
        # mkm = mkm.detach().to('cpu').numpy().transpose((1, 2, 0)).squeeze()
        # mkm2 = mkm2.detach().to('cpu').numpy().transpose((1, 2, 0)).squeeze()
        # imageio.imwrite(out_fn_pathm1, mkm)
        # imageio.imwrite(out_fn_pathm2, mkm2)

    def calc_loss(self):
        # import os
        # import imageio
        # pppath = "./hahahaha10"
        # os.makedirs(pppath, exist_ok=True)
        # fmt = '.tiff'
        # filename = "8"
        # out_fn_patho = os.path.join(pppath, filename + "_o_" + fmt)
        # out_fn_path1 = os.path.join(pppath, filename + "_t_" + fmt)
        # out_fn_path2 = os.path.join(pppath, filename + "_1_" + fmt)
        # out_fn_path3 = os.path.join(pppath, filename + "_2_" + fmt)
        # out_fn_pathm = os.path.join(pppath, filename + "_m_" + fmt)
        
        self.masking()

        bt, ct, ht, wt, nt = self.target_B.shape
        self.tb = self.target_B.view(bt, ct, nt, ht, wt).clone()    #[32, 1, 5, 120, 120]
        # self.ob = self.out_B.view(bt, ct, nt, ht, wt).clone()    #[32, 1, 5, 120, 120]

        self.target2b = self.tb.clone()
        self.target2b.masked_fill_(self.mask_B, 0.0) #[32, 1, 5, 120, 120]
        # self.out2b = self.ob.clone()
        # self.out2b.masked_fill_(self.mask_B, 0.0) #[32, 1, 5, 120, 120]

        self.tu = self.target_U.clone()    #[32, 1, 120, 120]
        # self.ou = self.out_U.clone()   #[32, 1, 120, 120]
        
        # self.mask_U = self.mask_U[:, :, 0, :, :].clone()
        self.target2u = self.tu.clone()
        self.target2u.masked_fill_(self.mask_U, 0.0)
        # self.out2u = self.ou.clone()
        # self.out2u.masked_fill_(self.mask_U, 0.0)
        
        # self.oB1 = self.out2b[:, :, 0, :, :].clone()
        # self.oU1 = self.out2u.clone()
        # self.out = self.oB1 + self.oU1
        self.tx = self.target_U.clone()
        
        self.loss_B = self.loss_criterion(self.target2b, self.out2b)
        mse_loss_B = self.mse_loss_criterion(self.out2b.detach(), self.target2b.detach())
        self.psnr_B = 10 * torch.log10(1 / mse_loss_B)
        
        self.loss_U = self.loss_criterion(self.target2u, self.out2u)
        mse_loss_U = self.mse_loss_criterion(self.out2u.detach(), self.target2u.detach())
        self.psnr_U = 10 * torch.log10(1 / mse_loss_U)

        self.loss = self.loss_criterion(self.tx, self.out)
        mse_loss = self.mse_loss_criterion(self.tx.detach(), self.out.detach())
        self.psnr = 10 * torch.log10(1 / mse_loss)
        
        # mk = self.out[0, :, :, :].squeeze(dim=1)
        # mkk = self.tx[0, :, :, :].squeeze(dim=1)
        # mk1 = self.out11[0, :, :, :].squeeze(dim=1)
        # mk2 = self.out22[0, :, :, :].squeeze(dim=1)
        # mk = mk.detach().to('cpu').numpy().transpose((1, 2, 0)).squeeze()
        # mkk = mkk.detach().to('cpu').numpy().transpose((1, 2, 0)).squeeze()
        # mk1 = mk1.detach().to('cpu').numpy().transpose((1, 2, 0)).squeeze()
        # mk2 = mk2.detach().to('cpu').numpy().transpose((1, 2, 0)).squeeze()
        # imageio.imwrite(out_fn_patho, mk)
        # imageio.imwrite(out_fn_path1, mkk)
        # imageio.imwrite(out_fn_path2, mk1)
        # imageio.imwrite(out_fn_path3, mk2)
        # print("Writing {}".format(os.path.abspath(out_fn_patho)))

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
                self.masking()  ## !! out3의 경우에만
            
            # ########## out1
            # bo, co, ho, wo, no = self.out_B.shape
            # out = self.out_B.view(bo, co, no, ho, wo)
            # out = out[:, :, 0, :, :].detach()
            # out = out.unsqueeze(2)
            # predicted_video.append(out)
            # predicted_idxs.append(predicted_idx)
            
            # ########## out2
            # bo, co, ho, wo = self.out_U.shape
            # out = self.out_U.unsqueeze(dim=1).detach()
            # predicted_video.append(out)
            # predicted_idxs.append(predicted_idx)
            
            ########## out3 (위에 self.masking 주석 처리 해제하기)
            bo, co, ho, wo = self.out.shape
            out = self.out.unsqueeze(dim=1).detach()
            predicted_video.append(out)
            predicted_idxs.append(predicted_idx)
            
            # ########## mask1 (위에 self.masking 주석 처리 해제하기)
            # bo, co, no, ho, wo = self.mask_B.shape
            # out = self.mask_B[:, :, 0, :, :].int().detach()
            # out = out.unsqueeze(2)
            # predicted_video.append(out)
            # predicted_idxs.append(predicted_idx)
            
            # ########## mask2 (위에 self.masking 주석 처리 해제하기)
            # bo, co, ho, wo = self.mask_U.shape
            # out = self.mask_U.unsqueeze(dim=1).detach()
            # predicted_video.append(out)
            # predicted_idxs.append(predicted_idx)
            
            # ########## masked2 (위에 self.masking 주석 처리 해제하기)
            # # bo, co, ho, wo = self.mask_U.shape
            # out = self.oU1.unsqueeze(dim=1).detach()
            # predicted_video.append(out)
            # predicted_idxs.append(predicted_idx)

        predicted_video = torch.cat(predicted_video, dim=2)
        return predicted_video, predicted_idxs

def create_model1(opt):
    return BilateralFilterLayer(opt)

def create_model2(opt):
    return UNetModel(opt)


####################################################
############### Bilater Filter Layer ###############
####################################################

class BilateralFilterFunction3dCPU(torch.autograd.Function):
    """
    3D Differentiable bilateral filter to remove noise while preserving edges. C++ accelerated layer (CPU).
    See:
        Paris, S. (2007). A gentle introduction to bilateral filtering and its applications: https://dl.acm.org/doi/pdf/10.1145/1281500.1281604
    Args:
        input_img: input tensor: [B, C, X, Y, Z]
        sigma_x: standard deviation of the spatial blur in x direction.
        sigma_y: standard deviation of the spatial blur in y direction.
        sigma_z: standard deviation of the spatial blur in z direction.
        color_sigma: standard deviation of the range kernel.
    Returns:
        output (torch.Tensor): Filtered tensor.
    """

    @staticmethod
    def forward(ctx, input_img, sigma_x, sigma_y, sigma_z, color_sigma):
        assert len(input_img.shape) == 5, "Input shape of 3d bilateral filter layer must equal [B, C, X, Y, Z]."
        assert input_img.shape[1] == 1, "Currently channel dimensions >1 are not supported."

        # Use c++ implementation for better performance.
        outputTensor, outputWeightsTensor, dO_dx_ki, dO_dsig_r, dO_dsig_x, dO_dsig_y, dO_dsig_z = bilateralfilter_cpu_lib.forward_3d_cpu(input_img, sigma_x, sigma_y, sigma_z, color_sigma)

        ctx.save_for_backward(input_img,
                              sigma_x,
                              sigma_y,
                              sigma_z,
                              color_sigma,
                              outputTensor,
                              outputWeightsTensor,
                              dO_dx_ki,
                              dO_dsig_r,
                              dO_dsig_x,
                              dO_dsig_y,
                              dO_dsig_z)  # save for backward

        return outputTensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_sig_x = None
        grad_sig_y = None
        grad_sig_z = None
        grad_color_sigma = None

        input_img = ctx.saved_tensors[0]  # input image
        sigma_x = ctx.saved_tensors[1]
        sigma_y = ctx.saved_tensors[2]
        sigma_z = ctx.saved_tensors[3]
        color_sigma = ctx.saved_tensors[4]
        outputTensor = ctx.saved_tensors[5]  # filtered image
        outputWeightsTensor = ctx.saved_tensors[6]  # weights
        dO_dx_ki = ctx.saved_tensors[7]  # derivative of output with respect to input, while k==i
        dO_dsig_r = ctx.saved_tensors[8]  # derivative of output with respect to range sigma
        dO_dsig_x = ctx.saved_tensors[9]  # derivative of output with respect to sigma x
        dO_dsig_y = ctx.saved_tensors[10]  # derivative of output with respect to sigma y
        dO_dsig_z = ctx.saved_tensors[11]  # derivative of output with respect to sigma z

        # calculate gradient with respect to the sigmas
        grad_color_sigma = torch.sum(grad_output * dO_dsig_r)
        grad_sig_x = torch.sum(grad_output * dO_dsig_x)
        grad_sig_y = torch.sum(grad_output * dO_dsig_y)
        grad_sig_z = torch.sum(grad_output * dO_dsig_z)

        grad_output_tensor = bilateralfilter_cpu_lib.backward_3d_cpu(grad_output,
                                                                     input_img,
                                                                     outputTensor,
                                                                     outputWeightsTensor,
                                                                     dO_dx_ki,
                                                                     sigma_x,
                                                                     sigma_y,
                                                                     sigma_z,
                                                                     color_sigma)

        return grad_output_tensor, grad_sig_x, grad_sig_y, grad_sig_z, grad_color_sigma


class BilateralFilterFunction3dGPU(torch.autograd.Function):
    """
    3D Differentiable bilateral filter to remove noise while preserving edges. CUDA accelerated layer.
    See:
        Paris, S. (2007). A gentle introduction to bilateral filtering and its applications: https://dl.acm.org/doi/pdf/10.1145/1281500.1281604
    Args:
        input_img: input tensor: [B, C, X, Y, Z]
        sigma_x: standard deviation of the spatial blur in x direction.
        sigma_y: standard deviation of the spatial blur in y direction.
        sigma_z: standard deviation of the spatial blur in z direction.
        color_sigma: standard deviation of the range kernel.
    Returns:
        output (torch.Tensor): Filtered tensor.
    """

    @staticmethod
    def forward(ctx, input_img, sigma_x, sigma_y, sigma_z, color_sigma):
        assert len(input_img.shape) == 5, "Input shape of 3d bilateral filter layer must equal [B, C, X, Y, Z]."
        assert input_img.shape[1] == 1, "Currently channel dimensions >1 are not supported."

        # Use c++ implementation for better performance.
        outputTensor, outputWeightsTensor, dO_dx_ki, dO_dsig_r, dO_dsig_x, dO_dsig_y, dO_dsig_z = bilateralfilter_gpu_lib.forward_3d_gpu(input_img, sigma_x, sigma_y, sigma_z, color_sigma)

        ctx.save_for_backward(input_img,
                              sigma_x,
                              sigma_y,
                              sigma_z,
                              color_sigma,
                              outputTensor,
                              outputWeightsTensor,
                              dO_dx_ki,
                              dO_dsig_r,
                              dO_dsig_x,
                              dO_dsig_y,
                              dO_dsig_z)  # save for backward

        return outputTensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_sig_x = None
        grad_sig_y = None
        grad_sig_z = None
        grad_color_sigma = None

        input_img = ctx.saved_tensors[0]  # input image
        sigma_x = ctx.saved_tensors[1]
        sigma_y = ctx.saved_tensors[2]
        sigma_z = ctx.saved_tensors[3]
        color_sigma = ctx.saved_tensors[4]
        outputTensor = ctx.saved_tensors[5]  # filtered image
        outputWeightsTensor = ctx.saved_tensors[6]  # weights
        dO_dx_ki = ctx.saved_tensors[7]  # derivative of output with respect to input, while k==i
        dO_dsig_r = ctx.saved_tensors[8]  # derivative of output with respect to range sigma
        dO_dsig_x = ctx.saved_tensors[9]  # derivative of output with respect to sigma x
        dO_dsig_y = ctx.saved_tensors[10]  # derivative of output with respect to sigma y
        dO_dsig_z = ctx.saved_tensors[11]  # derivative of output with respect to sigma z

        # calculate gradient with respect to the sigmas
        grad_color_sigma = torch.sum(grad_output * dO_dsig_r)
        grad_sig_x = torch.sum(grad_output * dO_dsig_x)
        grad_sig_y = torch.sum(grad_output * dO_dsig_y)
        grad_sig_z = torch.sum(grad_output * dO_dsig_z)

        grad_output_tensor = bilateralfilter_gpu_lib.backward_3d_gpu(grad_output,
                                                                     input_img,
                                                                     outputTensor,
                                                                     outputWeightsTensor,
                                                                     dO_dx_ki,
                                                                     sigma_x,
                                                                     sigma_y,
                                                                     sigma_z,
                                                                     color_sigma)

        return grad_output_tensor, grad_sig_x, grad_sig_y, grad_sig_z, grad_color_sigma


class BilateralFilter3d(nn.Module):
    def __init__(self, sigma_x, sigma_y, sigma_z, color_sigma, use_gpu=True):
        super(BilateralFilter3d, self).__init__()

        self.use_gpu = use_gpu

        # make sigmas trainable parameters
        self.sigma_x = nn.Parameter(torch.tensor(sigma_x))
        self.sigma_y = nn.Parameter(torch.tensor(sigma_y))
        self.sigma_z = nn.Parameter(torch.tensor(sigma_z))
        self.color_sigma = nn.Parameter(torch.tensor(color_sigma))

    def forward(self, input_tensor):

        assert len(input_tensor.shape) == 5, "Input shape of 3d bilateral filter layer must equal [B, C, X, Y, Z]."
        assert input_tensor.shape[1] == 1, "Currently channel dimensions >1 are not supported."

        # Choose between CPU processing and CUDA acceleration.
        if self.use_gpu:
            return BilateralFilterFunction3dGPU.apply(input_tensor,
                                                      self.sigma_x,
                                                      self.sigma_y,
                                                      self.sigma_z,
                                                      self.color_sigma)
        else:
            return BilateralFilterFunction3dCPU.apply(input_tensor,
                                                      self.sigma_x,
                                                      self.sigma_y,
                                                      self.sigma_z,
                                                      self.color_sigma)

    
class BilateralFilterLayer(nn.Module):
    def __init__(self, opt):
        super(BilateralFilterLayer, self).__init__()
        
        self.BF_img_1 = BilateralFilter3d(0.5, 0.5, 0.5, 0.01, True).to("cuda")
        self.BF_img_2 = BilateralFilter3d(0.5, 0.5, 0.5, 0.01, True).to("cuda")
        self.BF_img_3 = BilateralFilter3d(0.5, 0.5, 0.5, 0.01, True).to("cuda")
    
    def forward(self, x):
        out1 = self.BF_img_1(x)
        out2 = self.BF_img_2(out1)
        out3 = self.BF_img_3(out2)
        
        return out3


####################################################
####################### UNet #######################
####################################################

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
        bilinear = opt.bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        # self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        # self.down4 = Down(512, 1024 // factor)
        # self.up1 = Up(1024, 512 // factor, bilinear)
        # self.up2 = Up(512, 256 // factor, bilinear)

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

        self.sub_mean = common.MeanShift(1.0, n_channels=n_channels)
        self.add_mean = common.MeanShift(1.0, n_channels=n_channels, sign=1)

    def forward(self, x):
        x = self.sub_mean(x)
        res = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.convs(x3)

        x = self.up1(x, x2)
        x = self.up2(x, x1)
        x = self.outc(x)
        out = x + res

        out = self.add_mean(out)
        return out