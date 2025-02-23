##### BNN with GRADIENT #####

import torch
import torch.nn as nn
from .base_model import BaseModel
# from models.bilateral_filter_layer import create_bf
# from models.bilateral_filter_layer import *

import bilateralfilter_cpu_lib
import bilateralfilter_gpu_lib

from .bfgradient import Gradient

class BNN(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--sigma_x', type=float, default=0.5)
        parser.add_argument('--sigma_y', type=float, default=0.5)
        parser.add_argument('--sigma_z', type=float, default=0.5)
        parser.add_argument('--color_sigma', type=float, default=0.01)
        parser.add_argument('--downsample_factor', type=int, default=2)

        if is_train:
            parser.set_defaults(n_frames=5)
        else:
            parser.set_defaults(test_patches=False)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.model_names = ['bnn']
        self.n_frames = opt.n_frames

        # Create model
        self.bnn = create_model(opt).to(self.device)
        # self.bf3d = create_bf(1.0,1.0,1.0,0.01,True).to(self.device)
        
        # Define losses and optimizers
        if self.is_train:
            self.loss_criterion = nn.MSELoss()

            self.optimizer_names = ['optimizerQ']
            self.optimizerQ = torch.optim.Adam(self.bnn.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
            self.optimizers.append(self.optimizerQ)

        # for calculating PSNR
        self.mse_loss_criterion = nn.MSELoss()

    def set_input(self, input):
        self.x = input['lr'].to(self.device)
        # print("여기는 인풋, 데이터 사이즈는??", self.x.shape)
        # self.x = self.x.unsqueeze(4)
        b, c, n, h, w = self.x.shape
        # self.x = self.x.view(b, c*n, h, w)
        self.x = self.x.view(b, c, h, w, n)
        # print("여기는 인풋2, 데이터 사이즈는??", self.x.shape)
        if 'hr' in input:
            self.target = input['hr'].to(self.device)
            self.target = self.target.view(b, c, h, w, n)
            # self.target = self.target.unsqueeze(4)

    def forward(self):
        self.out = self.bnn(self.x)

    def backward(self):
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizerQ.zero_grad()
        self.forward()
        self.maskingOutputUponX()
        self.calc_loss()
        self.backward()
        self.optimizerQ.step()

    def maskingOutputUponX(self):
        gradient_lap = Gradient()
        
        # bilateral filter layer에 맞춰져 있던 shape을 gradient 계산 용으로 바꿔줌
        bx, cx, hx, wx, nx = self.x.shape
        self.XB = self.x.view(bx, cx, nx, hx, wx)
        bo, co, ho, wo, no = self.out.shape
        self.OB = self.out.view(bo, co, no, ho, wo)
        bt, ct, ht, wt, nt = self.target.shape
        self.TB = self.target.view(bt, ct, nt, ht, wt)
        
        # gradient 계산과 마스크 뽑는 함수로 넘김
        self.XB = self.XB.to('cpu')
        self.OB = self.OB.to('cpu')
        self.TB = self.TB.to('cpu')
        maskout, masktarget = gradient_lap(self.XB, self.OB, self.TB)
        maskout = maskout.to('cuda')
        masktarget = masktarget.to('cuda')
        self.OM = maskout.clone().detach().requires_grad_(True)
        self.TM = masktarget.clone().detach().requires_grad_(True)

    def calc_loss(self):
        # b, c, h, w, n = self.target.shape
        # self.target2 = self.target.view(b, c, n, h, w)
        # self.out2 = self.out.view(b, c, n, h, w)
        # self.loss = self.loss_criterion(self.target2, self.out2)
        # mse_loss = self.mse_loss_criterion(self.out2.detach(), self.target2.detach())
        # self.psnr = 10 * torch.log10(1 / mse_loss)

        # 마스크만 보고 loss 구함
        self.loss = self.loss_criterion(self.TM, self.OM)
        mse_loss = self.mse_loss_criterion(self.OM.detach(), self.TM.detach())
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
            bo, co, ho, wo, no = self.out.shape
            # print('self.out.shape: ', self.out.shape)
            out = self.out.view(bo, co, no, ho, wo)
            # print('out.shape1: ', out.shape)
            out = out[:, :, 0, :, :].detach()
            out = out.unsqueeze(2)
            #print('out.shape2: ', out.shape)
            #rint('predicted file {:03d}'.format(predicted_idx))
            predicted_video.append(out)
            predicted_idxs.append(predicted_idx)

        predicted_video = torch.cat(predicted_video, dim=2)
        return predicted_video, predicted_idxs

def create_model(opt):
    return BilateralFilter3d(opt)


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
    def __init__(self, opt):
        super(BilateralFilter3d, self).__init__()
        
        self.sigma_x = opt.sigma_x
        self.sigma_y = opt.sigma_y
        self.sigma_z = opt.sigma_z
        self.color_sigma = opt.color_sigma
        # make sigmas trainable parameters
        self.sigma_x = nn.Parameter(torch.tensor(self.sigma_x))
        self.sigma_y = nn.Parameter(torch.tensor(self.sigma_y))
        self.sigma_z = nn.Parameter(torch.tensor(self.sigma_z))
        self.color_sigma = nn.Parameter(torch.tensor(self.color_sigma))
        
        
        # self.sigma_x1 = opt.sigma_x
        # self.sigma_y1 = opt.sigma_y
        # self.sigma_z1 = opt.sigma_z
        # self.color_sigma1 = opt.color_sigma
        # self.sigma_x2 = opt.sigma_x
        # self.sigma_y2 = opt.sigma_y
        # self.sigma_z2 = opt.sigma_z
        # self.color_sigma2 = opt.color_sigma
        # self.sigma_x3 = opt.sigma_x
        # self.sigma_y3 = opt.sigma_y
        # self.sigma_z3 = opt.sigma_z
        # self.color_sigma3 = opt.color_sigma
        # self.sigma_x1 = nn.Parameter(torch.tensor(self.sigma_x1))
        # self.sigma_y1 = nn.Parameter(torch.tensor(self.sigma_y1))
        # self.sigma_z1 = nn.Parameter(torch.tensor(self.sigma_z1))
        # self.color_sigma1 = nn.Parameter(torch.tensor(self.color_sigma1))
        # self.sigma_x2 = nn.Parameter(torch.tensor(self.sigma_x2))
        # self.sigma_y2 = nn.Parameter(torch.tensor(self.sigma_y2))
        # self.sigma_z2 = nn.Parameter(torch.tensor(self.sigma_z2))
        # self.color_sigma2 = nn.Parameter(torch.tensor(self.color_sigma2))
        # self.sigma_x3 = nn.Parameter(torch.tensor(self.sigma_x3))
        # self.sigma_y3 = nn.Parameter(torch.tensor(self.sigma_y3))
        # self.sigma_z3 = nn.Parameter(torch.tensor(self.sigma_z3))
        # self.color_sigma3 = nn.Parameter(torch.tensor(self.color_sigma3))
        
        self.use_gpu = True

    def forward(self, x):

        assert len(x.shape) == 5, "Input shape of 3d bilateral filter layer must equal [B, C, X, Y, Z]."
        assert x.shape[1] == 1, "Currently channel dimensions >1 are not supported."

        # Choose between CPU processing and CUDA acceleration.
        if self.use_gpu:
            # out = BilateralFilterFunction3dGPU.apply(x,
            #                                           self.sigma_x,
            #                                           self.sigma_y,
            #                                           self.sigma_z,
            #                                           self.color_sigma)
            
            out1 = BilateralFilterFunction3dGPU.apply(x,
                                                      self.sigma_x,
                                                      self.sigma_y,
                                                      self.sigma_z,
                                                      self.color_sigma)
            out2 = BilateralFilterFunction3dGPU.apply(out1,
                                                      self.sigma_x,
                                                      self.sigma_y,
                                                      self.sigma_z,
                                                      self.color_sigma)
            out = BilateralFilterFunction3dGPU.apply(out2,
                                                      self.sigma_x,
                                                      self.sigma_y,
                                                      self.sigma_z,
                                                      self.color_sigma)

            
            # out1 = BilateralFilterFunction3dGPU.apply(x,
            #                                           self.sigma_x1,
            #                                           self.sigma_y1,
            #                                           self.sigma_z1,
            #                                           self.color_sigma1)
            # # print("color sigma0: ", self.color_sigma1)
            # # print("x0: ", x[0, 0, 0, 0, 0])
            # out2 = BilateralFilterFunction3dGPU.apply(out1,
            #                                           self.sigma_x2,
            #                                           self.sigma_y2,
            #                                           self.sigma_z2,
            #                                           self.color_sigma2)
            # # print("color sigma2: ", self.color_sigma2)
            # # print("out1: ", out1[0, 0, 0, 0, 0])
            # out = BilateralFilterFunction3dGPU.apply(out2,
            #                                           self.sigma_x3,
            #                                           self.sigma_y3,
            #                                           self.sigma_z3,
            #                                           self.color_sigma3)
            # # print("color sigma3: ", self.color_sigma3)
            # # print("out2: ", out2[0, 0, 0, 0, 0])
        else:
            out = BilateralFilterFunction3dCPU.apply(x,
                                                      self.sigma_x,
                                                      self.sigma_y,
                                                      self.sigma_z,
                                                      self.color_sigma)
            print("**use cpu**")

        return out