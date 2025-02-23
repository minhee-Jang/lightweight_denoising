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



class tripple16BNN_1(BaseModel):
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

        self.model_names = ['deep_bnn']
        self.n_frames = opt.n_frames



        #self.inputtofeature = FeatureExtractor().to(self.device)
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


            
            self.optimizer_spat1 = torch.optim.Adam([parameter for name, parameter in self.deep_bnn.named_parameters()
                                            if name.endswith(".sigma_x") or name.endswith(".sigma_y") or name.endswith(".sigma_z")], lr=1e-2, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0 )
            self.optimizer_color1 = torch.optim.Adam([parameter for name, parameter in self.deep_bnn.named_parameters()
                                                 if not name.endswith(".sigma_x") or name.endswith(".sigma_y") or name.endswith(".sigma_z")], lr=5e-3)
    
            self.optimizer_names = ['optimizer_spat1', 'optimizer_color1']


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

        self.x = input['lr'].to(self.device)
   

        #self.x_B = torch.squeeze(x, dim=0)    #[1, 1, 256, 256]  
     
        
        if 'hr' in input:
            self.target = input['hr'].to(self.device)
            #self.target_B = torch.squeeze(self.target, dim=0) 
            #print(self.target_B.shape)



    def forward(self):
        #feature = self.inputtofeature(self.x_B)         #[n..?, channel, width, height]
        #print(feature.shape)

        self.dBNN_out = self.deep_bnn(self.x)
       

    def backward(self):     
        self.loss.backward()       #loss 수정 주의

    def optimize_parameters(self):
        with torch.autograd.set_detect_anomaly(True):
            # self.optimizer_B.zero_grad()
            
            self.optimizer_spat1.zero_grad()
            self.optimizer_color1.zero_grad()

            self.forward()
            self.calc_loss()
            self.backward() 

            self.optimizer_spat1.step()           #para 업뎃
            self.optimizer_color1.step()




    def calc_loss_A(self):
 
        if self.perceptual_loss:
            self.content_loss1, self.style_loss1 = self.loss_criterion(self.target, self.dBNN_out)
            self.loss = self.content_loss1 + self.style_loss1
  
        else:
            self.loss = self.loss_criterion(self.target, self.dBNN_out)
  

        #self.loss = self.loss_criterion(self.tb, self.out)
        mse_loss = self.mse_loss_criterion(self.target.detach(), self.dBNN_out.detach())
    
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
        # y = batch['mask_E1']             #mask 따로 넣어줘야함 
        # z = batch['mask_E2']
        # r = batch['mask_NE']

        b, c, n, h, w = x.shape # here n is the number of whole images in the video

        predicted_video = []
        predicted_idxs = []

        for d in range(n):
            predicted_idx = d
           
            if n-d < n_frames:
                xd = x[:, :, -1*n_frames:]
                # yd = y[:, :, -1*n_frames:]
                # zd = z[:, :, -1*n_frames:]
                # rd = r[:, :, -1*n_frames:]

            else:
                xd = x[:, :, d:d+n_frames]
                # yd = y[:, :, d:d+n_frames]
                # zd = z[:, :, d:d+n_frames]
                # rd = r[:, :, d:d+n_frames]

            tensors_input = {
                "lr": xd,
                # "mask_E1": yd,
                # "mask_E2": zd,
                # "mask_NE": rd
            }
     
            with torch.no_grad():
                self.set_input(tensors_input)
                self.test()
                #self.masking()  ## !! out3의 경우에만
            
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
            #bo, co, ho, wo = self.dBNN_out.shape
            #print(self.dBNN_out.shape)
            #print(self.dBNN_out.dtype)
            out = self.dBNN_out.unsqueeze(dim=1).detach()
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

class BilaterlDeepNet(nn.Module):
    def __init__(self, opt):
        super(BilaterlDeepNet, self).__init__()

        self.feature_extractor1 = nn.Conv3d(1, 16, kernel_size=(1,3,3), padding=(0,1,1),bias=False)
        deep_bnn1 = [create_bnn(opt) for _ in range(16)]
        self.deep_bnn1 = nn.ModuleList(deep_bnn1)
        #self.reconstructor1 = nn.Conv3d(16, 1, kernel_size=(1,1,1), padding=(0,0,0),bias=False)

        self.feature_extractor2 = nn.Conv3d(16, 8, kernel_size=(1,1,1), padding=(0,0,0),bias=False)
        deep_bnn2 = [create_bnn(opt) for _ in range(8)]
        self.deep_bnn2 = nn.ModuleList(deep_bnn2)
        #self.reconstructor2 = nn.Conv3d(16, 32, kernel_size=(1,3,3), padding=(0,1,1),bias=False)

        self.feature_extractor3 = nn.Conv3d(8, 16, kernel_size=(1,1,1), padding=(0,0,0),bias=False)
        deep_bnn3 = [create_bnn(opt) for _ in range(16)]
        self.deep_bnn3 = nn.ModuleList(deep_bnn3)
        self.reconstructor3 = nn.Conv3d(16, 1, kernel_size=(1,3,3), padding=(0,1,1),bias=False)


    def forward(self, x):
        #print(x.shape)

        #bs, c, n, h, w= x.shape
        #assert n == 1, 'You must set n_frames as 1'   
        #print("x.shape", x.shape)
        # [bs, c, n, h, w] -> [bs, c, h, w, n]
        #x = x.unsqueeze(2)

        x =self.feature_extractor1(x)
        x = x.permute(0, 1, 3, 4, 2)    #[bs, c, h, w, n] 
        bs, c, h, w, n = x.shape
        #print(x.shape)

        bnn_out1 = []
        for i in range(c):

            input = x[ :, i:i+1, :, :, :]
            #print(input.shape)
            bnn = self.deep_bnn1[i]
            out = bnn(input)
            bnn_out1.append(out)   

        x = torch.cat(bnn_out1, dim=1)
        x = x.permute(0, 1, 4, 2, 3)   #다시 [bs, c, n, h, w]
        #x = self.reconstructor1(x)
       # print("first bnn finished")
        #print('bnn_out.shape:', x.shape)
        #x = x.squeeze(1)
        # bnn_out = self.conv2d(bnn_out)

        x =self.feature_extractor2(x)
        x = x.permute(0, 1, 3, 4, 2)    #[bs, c, h, w, n] 
        bs, c, h, w, n = x.shape
        bnn_out2 = []
        for i in range(c):
            input = x[ :, i:i+1, :, :, :]
            #print(input.shape)
            bnn = self.deep_bnn2[i]
            out = bnn(input)
            bnn_out2.append(out)   

        x = torch.cat(bnn_out2, dim=1)
        x = x.permute(0, 1, 4, 2, 3)
        #x = self.reconstructor2(x)
        #print("second bnn finished")



        x =self.feature_extractor3(x)
        x = x.permute(0, 1, 3, 4, 2)    #[bs, c, h, w, n] 
        bs, c, h, w, n = x.shape
        #print("32bnn", x.shape)
        bnn_out3 = []
        for i in range(c):
            input = x[ :, i:i+1, :, :, :]
            #print(input.shape)
            bnn = self.deep_bnn3[i]
            out = bnn(input)
            bnn_out3.append(out)   

        x = torch.cat(bnn_out3, dim=1)
        x = x.permute(0, 1, 4, 2, 3)
        out = self.reconstructor3(x)
        #print("third bnn finished")

        # bnn_out = self.conv2d_2(bnn_out)
        # bnn_out = self.conv2d_3(bnn_out)

       # print('bnn_out.shape:', bnn_out.shape)
        return out

