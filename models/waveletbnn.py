##### BNNs with GRADIENT #####
import torch
import torch.nn as nn
from .base_model import BaseModel
from models.loss.perceptual_loss_g import parse_wavelet_perceptual_loss, WaveletPerceptualLoss
from models.convs.wavelet import serialize_swt, unserialize_swt
from models.convs.wavelet import SWTForward, SWTInverse

import bilateralfilter_cpu_lib
import bilateralfilter_gpu_lib
# from models.common.bilateral import create_bnn

from .gdmasking import GradientMask
from models.loss.perceptual_loss_g import parse_perceptual_loss, PerceptualLoss

class WaveletBNN(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        
        # Wavelet deep learning model specification
        parser.add_argument('--wavelet_func', type=str, default='haar', #'bior2.2',
            help='wavelet function ex: haar, bior2.2, or etc.')
        parser.add_argument('--swt_lv', type=int, default=2,
            help='Level of stationary wavelet transform')
        
        
        parser.add_argument('--sigma_x', type=float, default=1.0)
        parser.add_argument('--sigma_y', type=float, default=1.0)
        parser.add_argument('--sigma_z', type=float, default=1.0)
        parser.add_argument('--color_sigma', type=float, default=0.01)
        
        parser.add_argument('--n_layers', type=int, default=3)
        parser.add_argument('--double_loss', type=str, default='plain', choices=['plain', 'double', 'doublep', 'perceptual'])
        
        parser.add_argument('--perceptual_loss', type=str, default=None,
            choices=['srgan', 'wavelet_transfer', 'perceptual_loss'],
            help='specity loss_type')
        
        if is_train:
            
            parser.add_argument('--ll_weight', type=float, default=0.2,
                help='weight of LL loss to high loss')
            parser = parse_wavelet_perceptual_loss(parser)
            parser.add_argument('--img_loss', default=False, action='store_true',
                help='include img loss')
            # parser = parse_perceptual_loss(parser)
        else:
            parser.set_defaults(test_patches=False)
        return parser
        

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        
        if opt.perceptual_loss is not None:
            self.perceptual_loss = True
            self.loss_type = opt.perceptual_loss
            self.double_loss = 'perceptual'
        else:
            self.perceptual_loss = False
            
        if self.perceptual_loss and self.is_train:
            self.loss_name = [
                'll_content_loss', 'll_style_loss', 'll_loss', 
                'high_content_loss', 'high_style_loss', 'high_loss',
                'content_loss', 'style_loss', 'total_loss'
            ]
        else:
            self.loss_name = [
                'll_loss', 'high_loss', 'total_loss'
            ]

        self.model_names = ['net']
        self.n_frames = opt.n_frames
 
        # Create model
        self.net = WaveletBilateral(opt).to(self.device)
        
        # Define SWTForward and SWTInverse
        self.swt_lv = opt.swt_lv
        self.swt = SWTForward(J=opt.swt_lv, wave=opt.wavelet_func).to(self.device)
        self.iswt = SWTInverse(wave=opt.wavelet_func).to(self.device)
        
        # Define losses and optimizers
        if self.is_train:
            
            if self.perceptual_loss:
                self.loss_criterion_p = PerceptualLoss(opt)
                
            self.loss_criterion = nn.MSELoss().to(self.device)

            self.optimizer_spat1 = torch.optim.Adam([parameter for name, parameter in self.net.named_parameters()
                                            if name.endswith(".sigma_x") or name.endswith(".sigma_y") or name.endswith(".sigma_z")], lr=1e-2, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0 )
            self.optimizer_color1 = torch.optim.Adam([parameter for name, parameter in self.net.named_parameters()
                                                 if not name.endswith(".sigma_x") or name.endswith(".sigma_y") or name.endswith(".sigma_z")], lr=5e-3)
    
            self.optimizer_names = ['optimizer_spat1', 'optimizer_color1']
       
            self.optimizers.append(self.optimizer_spat1)
            self.optimizers.append(self.optimizer_color1)

        # for calculating PSNR
        self.mse_loss_criterion = nn.MSELoss()

    def set_input(self, input):
        self.x = input['lr'].to(self.device)
        b, c, n, h, w = self.x.shape
        print("x initial: ", self.x.shape)
        
        self.x = self.x.squeeze(dim=1)    #32, 1, 1, 512, 512 => 32, 1, 512, 512
        # self.x_B = self.x.view(b, c, h, w, n)    #[32, 1, 120, 120, 5]
        
        if 'hr' in input:
            self.target = input['hr'].to(self.device)
            # self.target_B = self.target.view(b, c, h, w, n)    #[32, 1, 120, 120, 5]

    def forward(self):
        x = self.swt(self.x)
        x = serialize_swt(x)
        print("x serialize: ", x.shape) #32, 7, 512, 512

        self.swt_out = self.net(x)
        
        outw = unserialize_swt(self.swt_out, J=self.swt_lv, C=self.nc) # out = (ll, swt_coeffs)
        self.out = self.iswt(outw)

    def backward(self):        
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer_spat1.zero_grad()
        self.optimizer_color1.zero_grad()
        
        self.forward()
        self.calc_loss()
        
        self.backward()
        self.optimizer_spat1.step()
        self.optimizer_color1.step()

    def calc_loss(self):
        
        swt_target = self.swt(self.target)
        swt_target = serialize_swt(swt_target)

        swt_out = self.swt_out
        
        if self.perceptual_loss:
            self.content_loss1, self.style_loss1 = self.loss_criterion(set_target, swt_out)
            self.loss = self.content_loss1 + self.style_loss1
        else:
            self.loss = self.loss_criterion(self.target, self.out)

        mse_loss = self.mse_loss_criterion(self.target.detach(), self.out.detach())
        self.psnr = 10 * torch.log10(1 / mse_loss)

    def get_logs(self):
        if self.perceptual_loss:
            log_dict = {
            'loss': '{:.8f}'.format(self.loss),
            'content_loss': '{:.8f}'.format(self.content_loss),
            'style_loss': '{:.8f}'.format(self.style_loss),
            'mse_loss': '{:.8f}'.format(self.mse_loss),
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
                self.outMasking()  ## !! out3의 경우에만
            
            ########## out1
            # bo, co, ho, wo, no = self.out_B1.shape
            # out = self.out_B1.view(bo, co, no, ho, wo)
            # out = out[:, :, 0, :, :].detach()
            # out = out.unsqueeze(2)
            # predicted_video.append(out)
            # predicted_idxs.append(predicted_idx)
            
            # ########## out2
            # bo, co, ho, wo, no = self.out_B2.shape
            # out = self.out_B2.view(bo, co, no, ho, wo)
            # out = out[:, :, 0, :, :].detach()
            # out = out.unsqueeze(2)
            # predicted_video.append(out)
            # predicted_idxs.append(predicted_idx)
            
            # ########## out3 (위에 self.masking 주석 처리 해제하기)
            # bo, co, ho, wo = self.out.shape
            # out = self.out.unsqueeze(dim=1).detach()
            # predicted_video.append(out)
            # predicted_idxs.append(predicted_idx)
            out = self.out[:, :, 0, :, :].detach()
            out = out.unsqueeze(2)
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
        # assert len(input_img.shape) == 5, "Input shape of 3d bilateral filter layer must equal [B, C, X, Y, Z]."
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
        # assert len(input_img.shape) == 5, "Input shape of 3d bilateral filter layer must equal [B, C, X, Y, Z]."
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

        # assert len(input_tensor.shape) == 5, "Input shape of 3d bilateral filter layer must equal [B, C, X, Y, Z]."
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
        
        self.n_layers = opt.n_layers
        
        # self.BF_img_1 = BilateralFilter3d(0.5, 0.5, 0.5, 0.01, True).to("cuda")
        # self.BF_img_2 = BilateralFilter3d(0.5, 0.5, 0.5, 0.01, True).to("cuda")
        # self.BF_img_3 = BilateralFilter3d(0.5, 0.5, 0.5, 0.01, True).to("cuda")
        
        # if opt.n_layers == 5:
        #     self.BF_img_4 = BilateralFilter3d(0.5, 0.5, 0.5, 0.01, True).to("cuda")
        #     self.BF_img_5 = BilateralFilter3d(0.5, 0.5, 0.5, 0.01, True).to("cuda")
            
        self.BF_img_1 = BilateralFilter3d(0.5, 0.5, 0.5, 0.01, True)
        self.BF_img_2 = BilateralFilter3d(0.5, 0.5, 0.5, 0.01, True)
        self.BF_img_3 = BilateralFilter3d(0.5, 0.5, 0.5, 0.01, True)
        
        if opt.n_layers == 5:
            self.BF_img_4 = BilateralFilter3d(0.5, 0.5, 0.5, 0.01, True)
            self.BF_img_5 = BilateralFilter3d(0.5, 0.5, 0.5, 0.01, True)
    
    def forward(self, x):
        
        out1 = self.BF_img_1(x)
        out2 = self.BF_img_2(out1)
        out3 = self.BF_img_3(out2)
        
        if self.n_layers == 5:
            out4 = self.BF_img_4(out3)
            out5 = self.BF_img_5(out4)
            return out5

        return out3

def create_bnn(opt):
    return BilateralFilterLayer(opt)



class WaveletBilateral(nn.Module):
    def __init__(self, opt):
        super(WaveletBilateral, self).__init__()
        
        bnn = create_bnn(opt)
        multibnn = [bnn for _ in range(7)]
        self.multibnns = nn.ModuleList(multibnn)
        
    def forward(self, x):
        
        x = x.unsqueeze(dim=4)
        #print("x unsqueeze: ", x.shape) #32, 7, 512, 512, 1
        
        outs = []
        for i in range(7):
            #print("ja: ", i)
            xi = x[:, i:i+1, :, :, :]
            #print("xi :", xi.shape)
            self.bnni = self.multibnns[i]
            outi = self.bnni(xi)
            #print("ja: ", i)
            outs.append(outi)
        
        outs = torch.cat(outs, dim=1)
        
        return outs