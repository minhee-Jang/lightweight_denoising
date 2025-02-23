##### BNN and UNet with GRADIENT #####

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from models.convs import common
from models.common.bilateral import create_bnn
from models.common.unet import Up

from data.srdata import SRData
from .gaussian import gauss_img
from models.loss.perceptual_loss import parse_perceptual_loss, PerceptualLoss
from torchvision import models, transforms
import numpy as np


extract_list = ["conv1", "maxpool", "layer1", "avgpool", "fc"]
resnet = models.resnet50(pretrained=True)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor()]
)

class deepBNN(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--sigma_x', type=float, default=0.5)
        parser.add_argument('--sigma_y', type=float, default=0.5)
        parser.add_argument('--sigma_z', type=float, default=0.5)
        parser.add_argument('--color_sigma', type=float, default=0.01)
        parser.add_argument('--downsample_factor', type=int, default=2)
        parser.add_argument('--bilinear', type=str, default='bilinear',
            help='up convolution type (bilineaer or transposed2d)')

        parser.add_argument('--n_layers', type=int, default=3)
        parser.add_argument('--bilateral_loss', type=str, default='plain', choices=['plain', 'double'])  
        parser.add_argument('--bf_loss', type=float, default=0.3)
        parser.add_argument('--perceptual_loss', type=str, default=None,
            choices=['srgan', 'wavelet_transfer', 'perceptual_loss'],
            help='specity loss_type')
        

        if is_train:
            parser = parse_perceptual_loss(parser)
            parser.set_defaults(n_frames=1)
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

        #self.model_names = ['bf3d', 'bf3d2', 'bf3d3']
        self.n_frames = opt.n_frames


 
        # Create model
        extract_list = ["conv1", "maxpool", "layer1", "avgpool", "fc"]
        resnet = models.resnet50(pretrained=True)
        self.inputtofeature = FeatureExtractor(resnet, extract_list).to(self.device)
        self.deep_bnn = BilaterlDeepNet(opt).to(self.device)


        # Define losses and optimizers
        if self.is_train:

            if opt.content_loss == 'l1':
                self.loss_criterion = nn.L1Loss()
            elif opt.content_loss == 'l2':
                self.loss_criterion = nn.MSELoss()

            if self.perceptual_loss:
                self.loss_criterion = PerceptualLoss(opt)
                self.loss_criterion2 = nn.MSELoss()


            #self.optimizer_B = torch.optim.Adam(self.bf3d.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
            #self.optimizer_U = torch.optim.Adam(self.unet.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)


            self.optimizer_F =torch.optim.Adam(self.inputtofeature.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
            self.optimizer_B = torch.optim.Adam([parameter for name, parameter in self.deep_bnn.named_parameters()
                                            if not name.endswith(".sigma_x") or name.endswith(".sigma_y") or name.endswith(".sigma_z") or name.endswith(".color_sigma")], lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
            
            self.optimizer_spat1 = torch.optim.Adam([parameter for name, parameter in self.deep_bnn.named_parameters()
                                            if name.endswith(".sigma_x") or name.endswith(".sigma_y") or name.endswith(".sigma_z")], lr=1e-2, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0 )
            self.optimizer_color1 = torch.optim.Adam([parameter for name, parameter in self.deep_bnn.named_parameters()
                                                 if name.endswith(".color_sigma")], lr=5e-3)
    
            self.optimizer_names = ['optimizer_spat1', 'optimizer_color1', 'optimizer_F', 'optimizer_B']


            self.optimizer = []
            # self.optimizers.append(self.optimizer_B)
            self.optimizers.append(self.optimizer_spat1)
            self.optimizers.append(self.optimizer_color1)


            if opt.bilateral_loss == 'plain':
                self.calc_loss = self.calc_loss_A
            else:
                self.calc_loss = self.calc_loss_B
            
        # for calculating PSNR
        self.mse_loss_criterion = nn.MSELoss()


    def set_input(self, input):

        x = input['lr'].to(self.device)
        #print(x.shape)   #torch.size([1,1,1,512,512])
        #print(type(x))   

  
        x =  F.interpolate(x, size=(1, 256,256)).to(self.device)
        #print(x.shape)  #[1, 1, 1, 256, 256]
 
    


        #b, c, n, h, w = self.x.shape

        self.x_B = torch.squeeze(x, dim=0)    #[1, 1, 256, 256]  
     
        
        if 'hr' in input:
            self.target = input['hr'].to(self.device)
            self.target_B = torch.squeeze(self.target, dim=0) 
            #print(self.target_B.shape)



    def forward(self):
        feature = self.inputtofeature(self.x_B)         #[n..?, channel, width, height]
        #print(feature.shape)

        self.dBNN_out = self.deep_bnn(feature)
         
        #print('deep bnn 최종 out', dBNN_out.shape)
        
 

      
        # for name, param in self.deep_bnn.named_parameters():
        #     print(name, param)
        # # print("bilateral2")
        # for name, param in self.inputtofeature.named_parameters():
        #     print(name, param)
        # print("bilateral3")
        # for name, param in self.bf3d3.named_parameters():
        #     print(name, param)


    def backward_B1(self):     
        self.loss.backward()       #loss 수정 주의

    def optimize_parameters(self):
        with torch.autograd.set_detect_anomaly(True):
            # self.optimizer_B.zero_grad()
            self.optimizer_spat1.zero_grad()
            self.optimizer_color1.zero_grad()
            self.optimizer_F.zero_grad()
            self.optimizer_B.zero_grad()

            self.forward()
            self.calc_loss()

            self.backward_B1() 

            self.optimizer_spat1.step()           #para 업뎃
            self.optimizer_color1.step()
            self.optimizer_F.step()
            self.optimizer_B.step()


    def calc_loss_A(self):
 
        if self.perceptual_loss:
            self.content_loss1, self.style_loss1 = self.loss_criterion(self.target_B, self.dBNN_out)
            self.loss = self.content_loss1 + self.style_loss1
  
        else:
            self.loss = self.loss_criterion(self.target_B, self.dBNN_out)
  

        #self.loss = self.loss_criterion(self.tb, self.out)
        mse_loss = self.mse_loss_criterion(self.target_B.detach(), self.dBNN_out.detach())
    
        self.psnr = 10 * torch.log10(1 / mse_loss)
       

    def calc_loss_B(self):
        self.masking()
        
        bt, ct, ht, wt, nt = self.target_B.shape
        self.tb = self.target_B.view(bt, ct, nt, ht, wt).clone()    #[32, 1, 5, 120, 120]
        
        self.target2b_1 = self.tb.clone()
        self.target2b_1.masked_fill_(self.mask_B1, 0.0) #[32, 1, 5, 120, 120]

        self.target2b_2 = self.tb.clone()
        self.target2b_2.masked_fill_(self.mask_B2, 0.0) #[32, 1, 5, 120, 120]

        self.target2b_3 = self.tb.clone()
        self.target2b_3.masked_fill_(self.mask_B3, 0.0) #[32, 1, 5, 120, 120]

        
        self.tx = self.target_B.clone()
    
        #### New Loss Calculation
        self.loss_b1 = self.loss_criterion(self.target2b_1, self.out2b_1)
        self.loss_b2 = self.loss_criterion(self.target2b_2, self.out2b_2)
        self.loss_b3 = self.loss_criterion(self.target2b_3, self.out2b_3)

        self.loss = self.loss_criterion(self.tb, self.out)
        self.loss_B1 = (self.loss_b1 + self.loss.item()) * 0.5
        self.loss_B2 = (self.loss_b2 + self.loss.item()) * 0.5
        self.loss_B3 = (self.loss_b3 + self.loss.item()) * 0.5

        mse_loss = self.mse_loss_criterion(self.tb.detach(), self.out.detach())

        self.psnr = 10 * torch.log10(1 / mse_loss)




    def get_logs(self):
        if PerceptualLoss is None:
            log_dict = {
                'loss': '{:.8f}'.format(self.loss),
                'content_loss': '{:.8f}'.format(self.content_loss),
                'style_loss': '{:.8f}'.format(self.style_loss),
                'psnr': '{:.8f}'.format(self.psnr)
            }
        else:
            log_dict = {
                'loss': '{:.8f}'.format(self.loss),
            # 'content_loss': '{:.8f}'.format(self.content_loss),
            # 'style_loss': '{:.8f}'.format(self.style_loss),
                'psnr': '{:.8f}'.format(self.psnr)
            }
        return log_dict


    def get_batch_measure(self):
        return self.loss.detach(), self.psnr.detach()
    
    def predict(self, batch):
        n_frames = self.n_frames
        x = batch['lr']
        y = batch['mask_E1']             #mask 따로 넣어줘야함 
        z = batch['mask_E2']
        r = batch['mask_NE']

        b, c, n, h, w = x.shape # here n is the number of whole images in the video

        predicted_video = []
        predicted_idxs = []

        for d in range(n):
            predicted_idx = d
           
            if n-d < n_frames:
                xd = x[:, :, -1*n_frames:]
                yd = y[:, :, -1*n_frames:]
                zd = z[:, :, -1*n_frames:]
                rd = r[:, :, -1*n_frames:]

            else:
                xd = x[:, :, d:d+n_frames]
                yd = y[:, :, d:d+n_frames]
                zd = z[:, :, d:d+n_frames]
                rd = r[:, :, d:d+n_frames]

            tensors_input = {
                "lr": xd,
                "mask_E1": yd,
                "mask_E2": zd,
                "mask_NE": rd
            }
     
            with torch.no_grad():
                self.set_input(tensors_input)
                self.test()
                self.masking()  ## !! out3의 경우에만
            
            # ########## out1   #bilateral 결과
            # bo, co, ho, wo, no = self.out_B.shape
            # out = self.out_B.view(bo, co, no, ho, wo)
            # out = out[:, :, 0, :, :].detach()
            # out = out.unsqueeze(2)
            # predicted_video.append(out)
            # predicted_idxs.append(predicted_idx)
            
            # ########## out2   #unet 결과
            # bo, co, ho, wo = self.out_U.shape
            # out = self.out_U.unsqueeze(dim=1).detach()
            # predicted_video.append(out)
            # predicted_idxs.append(predicted_idx)
            
            ########## out3 (위에 self.masking 주석 처리 해제하기)    #합친거 결과 
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

# def create_model1(opt):
#     return BilateralFilterLayer(opt)


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers
        #print(submodule)
        self.conv1 = nn.Conv2d(1, 3, kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=False)
    
    def forward(self, x):

        x= self.conv1(x)
        for name, module in self.submodule._modules.items():
            # if name is "fc": 
            #     x = x.view(x.size(0), -1)
            x = module(x)             #[b, c, w, h]
 

            if name is "maxpool":                #feature map 수 64일 때 
               # print('최종shape', x.shape)
                outputs =x 
                return outputs
   


class BilaterlDeepNet(nn.Module):
    def __init__(self, opt):
        super(BilaterlDeepNet, self).__init__()

        bnn = create_bnn(opt)
        #self.use_gpu =  opt.use_gpu
        deep_bnn = [bnn for _ in range(64)]
        self.deep_bnn = nn.ModuleList(deep_bnn)

        self.conv2d_1=nn.Conv2d(64, 32, kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=False )
  
        self.conv2d_2=nn.Conv2d(32, 1, kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=False )
        self.conv2d_3=nn.Conv2d(1, 1, kernel_size=(3,3),stride=(1,1),padding=(1,1),bias=False )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, input_tensor):
       # print('input tensor shape', input_tensor.shape)
        bs, c, h, w = input_tensor.shape
        # print('input_tensor.shape:', input_tensor.shape)
        input_tensor = input_tensor.unsqueeze(2)
        #print('unsqeezed input_tensor.shape:', input_tensor.shape)
        input_tensor = input_tensor.permute(0, 2, 3, 4, 1)    #[bs, n ,  h, w, c]
        #print('transposed input_tensor.shape:', input_tensor.shape)
        # input_tensor = input_tensor.unsqueeze(-1)
  

        bnn_out = []
        for i in range(c):
            input = input_tensor[ :, :, :, :, i:i+1]

            #print(input.shape)
            bnn = self.deep_bnn[i]
            out = bnn(input)
            bnn_out.append(out)   

        bnn_out = torch.cat(bnn_out, dim=4)
        #print('bnn_out.shape:', bnn_out.shape)
        bnn_out = bnn_out.squeeze(1)
        # bnn_out = self.conv2d(bnn_out)
        bnn_out = self.up(bnn_out)
        bnn_out = self.conv2d_1(bnn_out)
        bnn_out = self.up(bnn_out)
        bnn_out = self.conv2d_2(bnn_out)
        bnn_out = self.up(bnn_out)
        bnn_out = self.conv2d_3(bnn_out)

       # print('bnn_out.shape:', bnn_out.shape)
        return bnn_out




