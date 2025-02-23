import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from models.convs import common
from models.common.bilateralw import create_bnn
import torch.nn.init as init
from models.common.kb_utils import KBAFunction
from models.common.kb_utils import LayerNorm2d, SimpleGate
import math
import numpy as np
import os
import cv2
import imageio


class best_loss(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--alpha', type=float, default=0.5)
        parser.add_argument('--beta', type=float, default=0.3)
        parser.add_argument('--sigma_x', type=float, default=0.5)
        parser.add_argument('--sigma_y', type=float, default=0.5)
        parser.add_argument('--sigma_z', type=float, default=0.5)
        parser.add_argument('--color_sigma', type=float, default=0.01)
        parser.add_argument('--downsample_factor', type=int, default=2)
        parser.add_argument('--bilinear', type=str, default='bilinear',
            help='up convolution type (bilineaer or transposed2d)')

        parser.add_argument('--n_layers', type=int, default=3)
        parser.set_defaults(batch_size=1)
        parser.set_defaults(n_frames=1)
        
        # Wavelet deep learning model specification
        parser.add_argument('--wavelet_func', type=str, default='haar', #'bior2.2',
            help='wavelet function ex: haar, bior2.2, or etc.')
        parser.add_argument('--swt_lv', type=int, default=2,
            help='Level of stationary wavelet transform')
        
        if is_train:
            parser.add_argument('--content_loss', type=str, choices=['l1', 'l2'], default='l2',
                help='loss function (l1, l2)')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.n_frames = opt.n_frames
        self.alpha = opt.alpha
        self.beta = opt.beta
        self.model_names = ['net']
        self.net = create_model(opt).to(self.device)
        self.sobel =  MakeEdgeBins().to(self.device)
        self.canny = CannyEdgeDetector(opt).to(self.device)
        self.gamma = 1 - self.alpha - self.beta

        print("parameter 갯수 : ", sum(p.numel() for p in self.net.parameters() if p.requires_grad))


        # Define losses and optimizers
        if self.is_train:
            if opt.content_loss == 'l1':
                self.loss_criterion = nn.L1Loss()
            elif opt.content_loss == 'l2':
                self.loss_criterion = nn.MSELoss()

            self.optimizer_names = [ 'optimizer_spat1', 'optimizer_color1']
            self.optimizer_spat1 = torch.optim.Adam(
                [parameter for name,parameter in self.net.named_parameters() if name.endswith(".sigma_x") or name.endswith(".sigma_y") or name.endswith(".sigma_z")],
                lr=1e-2, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)
            self.optimizer_color1 = torch.optim.Adam(
                [parameter for name, parameter in self.net.named_parameters() if not name.endswith(".sigma_x") or name.endswith(".sigma_y") or name.endswith(".sigma_z")],
                lr=5e-3)

            self.optimizer = []       
            self.optimizers.append(self.optimizer_spat1)
            self.optimizers.append(self.optimizer_color1)
            
        # for calculating PSNR
        self.mse_loss_criterion = nn.MSELoss()

    def set_input(self, input):
        self.x = input['lr'].to(self.device)
        self.mask1 = input['mask_E1'].to(self.device) 
        self.mask2 = input['mask_E2'].to(self.device) 
        self.mask3 = input['mask_NE'].to(self.device) 

        self.mask_B1 = self.mask1.type(torch.bool)        
        self.mask_B2 = self.mask2.type(torch.bool)  
        self.mask_NE = self.mask3.type(torch.bool)

        self.ones = torch.ones(self.mask_B1.size(0), 1, 1, 512, 512).to(self.device) 
        #print(self.mask_B1.shape)

        self.mask_i1 = self.ones.clone()
        self.mask_i1.masked_fill_(self.mask_B1, 0.0)
        self.mask_i2 = self.ones.clone()
        self.mask_i2.masked_fill_(self.mask_B2, 0.0)
        self.mask_i3 = self.ones.clone()
        self.mask_i3.masked_fill_(self.mask_NE, 0.0)


        masks=[]
        masks.append(self.mask_i1)
        masks.append(self.mask_i2)        
        masks.append(self.mask_i3)  
              
        self.masks=torch.cat(masks, dim=1)        
       
        if 'hr' in input:
            self.target = input['hr'].to(self.device)  #[bs, c, n, h, w]


    def forward(self):
  
        self.out, self.out_sobel, self.dn1, self.dn2 = self.net(self.x, self.masks)
         
        # forTE = self.target.clone()
        # forTE = forTE.squeeze(dim=2)
        # ebt = self.sobel(forTE)
        # self.target_sobel = ebt.unsqueeze(dim=2)
        # cannyE = self.canny(forTE)

        # forOE = self.out.detach()
        # forOE = forOE.squeeze(dim=2)
        # cannyofout = self.canny(forOE)

        # self.target_canny = cannyE.unsqueeze(dim=2)
        # self.out_canny = cannyofout.unsqueeze(dim=2)

        # for name, param in self.net.named_parameters():
        #     print(name, param)
         
    def backward(self):
        self.loss.backward()       #loss 수정 주의

    def optimize_parameters(self):
        self.optimizer_spat1.zero_grad()
        self.optimizer_color1.zero_grad()
        self.forward()
        self.calc_loss()
        self.backward() 
        self.optimizer_spat1.step()           #para 업뎃
        self.optimizer_color1.step()

    def calc_loss(self):
        self.loss_i1 = self.loss_criterion(self.target, self.out)
        self.loss_i2 = self.loss_criterion(self.target, self.dn1)
        self.loss_i3 = self.loss_criterion(self.target, self.dn2)
        self.loss_e = self.loss_criterion(self.target_sobel, self.out_sobel)
        self.loss_e2 = self.loss_criterion(self.target_canny, self.out_canny)

        # self.mapX = self.edge_map*self.out
        # self.mapY = self.edge_map*self.target
        # self.loss_map = self.loss_criterion(self.mapY, self.mapX)

       # self.loss = (1 - self.alpha) * (self.loss_i1 + self.loss_i2 + self.loss_i3) + self.alpha*self.loss_e
        #self.loss =  self.alpha * self.loss_i1 + (1 - self.alpha) * (self.loss_e + self.loss_e2)
        self.loss =  self.alpha * self.loss_i1 + self.gamma * (self.loss_i2 + self.loss_i3) + self.beta * self.loss_e

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
        y = batch['mask_E1']      
        z = batch['mask_E2']
        r = batch['mask_NE']
        # print('predicting x.shape:', x.shape)
        b, c, n, h, w = x.shape # here n is the number of whole images in the video

        predicted_video = []
        predicted_idxs = []
        # predicted_1 = []
        # predicted_2 = []

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
            
            # co, no, ho, wo = self.out.shape
            # print('self.dBNN_out.shape', self.out.shape)
            # # out = self.out.unsqueeze(dim=2).detach()
            # print('out.shape', self.out.shape)
            out = self.out
            # dn1 = self.dn1
            # dn2 = self.dn2
            # predicted_1.append(dn1)
            # predicted_2.append(dn2)
            predicted_video.append(out)
            predicted_idxs.append(predicted_idx)

        # predicted_1 = torch.cat(predicted_1, dim=2)
        # predicted_2 = torch.cat(predicted_2, dim=2)
        predicted_video = torch.cat(predicted_video, dim=2)

        #return predicted_1, predicted_2, predicted_video, predicted_idxs
        return predicted_video, predicted_idxs

def create_model(opt):
    return BilateralDeepNet(opt)


class BilateralDeepNet(nn.Module):
    def __init__(self, opt):
        super(BilateralDeepNet, self).__init__()

        self.device = opt.device
    
        self.makeEB = MakeEdgeBins().to(self.device)
        self.canny = CannyEdgeDetector(opt).to(self.device)
        self.fe = nn.Conv3d(5, 3, kernel_size=(1,3,3), padding=(0,1,1), bias=False) 
        self.fe1 = nn.Conv3d(5, 1, kernel_size=(1,3,3), padding=(0,1,1), bias=False) 

        # fusion = [SemanticFusion(opt) for _ in range(5)]  #fusion out 1
        # self.fusion = nn.ModuleList(fusion)
        # edge = [Edge_net(opt) for _ in range(5)]    #edge out 4
        # self.Edge = nn.ModuleList(edge)

        self.fusion1=SemanticFusion(opt, 6)
        self.fusion2=SemanticFusion(opt, 7)
        self.fusion3=SemanticFusion(opt, 7)

        self.Edge1 = Edge_net(opt)
        self.Edge2 = Edge_net(opt)
        self.Edge3 = Edge_net(opt)

        self.fe2 = nn.Conv3d(3, 3, kernel_size=(1,3,3), padding=(0,1,1), bias=False)
        self.edge_extractor = nn.Conv3d(5, 4, kernel_size=(1,3,3), padding=(0,1,1), bias=False)
        self.recon_hi = nn.Conv3d(7, 1, kernel_size=(1,3,3), padding=(0,1,1), bias=False)
        self.recon_hi2 = nn.Conv3d(2, 1, kernel_size=(1,3,3), padding=(0,1,1), bias=False)

        self.act = nn.Sigmoid()
        self.conv1_1=nn.Conv3d(1, 1, kernel_size=(1,1,1), padding=(0,0,0), bias=False)
        self.conv1_2=nn.Conv3d(1, 1, kernel_size=(1,1,1), padding=(0,0,0), bias=False)
        self.conv1_3=nn.Conv3d(1, 1, kernel_size=(1,1,1), padding=(0,0,0), bias=False)
        self.conv2_1=nn.Conv3d(1, 1, kernel_size=(1,1,1), padding=(0,0,0), bias=False)
        self.conv2_2=nn.Conv3d(1, 1, kernel_size=(1,1,1), padding=(0,0,0), bias=False)
        self.conv2_3=nn.Conv3d(1, 1, kernel_size=(1,1,1), padding=(0,0,0), bias=False)
        # self.conv3_1=nn.Conv3d(1, 1, kernel_size=(1,1,1), padding=(0,0,0), bias=False)
        # self.conv3_2=nn.Conv3d(1, 1, kernel_size=(1,1,1), padding=(0,0,0), bias=False)
        # self.conv3_3=nn.Conv3d(1, 1, kernel_size=(1,1,1), padding=(0,0,0), bias=False)
        
    def forward(self, x, masks):
        #maintaining x input
        forE = x.clone()
        forE = forE.squeeze(dim=2)
        ebs = self.makeEB(forE)
        detail_e = self.canny(forE)  
        detail_e = detail_e.unsqueeze(dim=2) #c =1 
        ebs = ebs.unsqueeze(dim=2)
        #[1, 4, 1, 512, 512]
        ############################################
        x_in = torch.cat((x, ebs), dim=1)
        input = self.fe(x_in)   #c = 3
        #x_b = self.fe1(x_b)  #bnn input   c =1 
        
        edge1 = self.Edge1(x_in)   #out c = 5
        x_b1 = x_in + edge1 #5
        detail_e = detail_e.to(x_b1.device)
        x_b1 = torch.cat([x_b1, detail_e], dim=1)  #c=6
        out1 = self.fusion1(x_b1, masks)    # out 1  F in

        out1f = self.conv1_1(out1)  #1x1
        dn1 = out1f + x          #c=1
        fe1 = self.conv1_2(out1f)
        atm = self.conv1_3(dn1)
        atm = self.act(atm)  #sig  > attention map
        fe1 = fe1 * atm + out1    #c=1 


        edge2 = edge1 + x_in #5 유지 
        edge2 = self.Edge2(edge2) 
        x_b2 = torch.cat([dn1, ebs], dim=1) 
        x_b2 = x_b2 + edge2
        x_b2 = torch.cat([x_b2, fe1, detail_e], dim=1) #c=7
        out2 = self.fusion2(x_b2, masks)


        out2f = self.conv2_1(out2)  #1x1
        dn2 = out2f + x          #c=1
        fe2 = self.conv2_2(out2f)
        atm2 = self.conv2_3(dn2)
        atm2 = self.act(atm2)  #sig  > attention map
        fe2 = fe2 * atm2 + out2    #c=1 


        edge3 = edge2 + x_in #5 유지 
        edge3 = self.Edge3(edge3) 
        x_b3 = torch.cat([dn2, ebs], dim=1) 
        x_b3 = x_b3 + edge3
        x_b3 = torch.cat([x_b3, fe2, detail_e], dim=1)
        out3 = self.fusion3(x_b3, masks)

        # out3f = self.conv3_1(out3)  #1x1
        # dn3 = out3f + x          #c=1
        # fe3 = self.conv3_2(out3f)
        # atm3 = self.conv3_3(dn3)
        # atm3 = self.act(atm3)  #sig  > attention map
        # fe3 = fe3 * atm3 + out3    #c=1 


        out_f = torch.cat([out1, out2, out3], dim=1)

        b_out = self.fe2(out_f)   #3로 
        b_out = b_out + input     

        edge_sobel = self.edge_extractor(edge3)   

        out_f = torch.cat((b_out, edge_sobel), dim =1)
        out = self.recon_hi(out_f)

        out = torch.cat([detail_e, out], dim=1)
        out = self.recon_hi2(out)


        out = out + x
        
   

        return out, edge_sobel, dn1, dn2
    
class CannyEdgeDetector(nn.Module):
    def __init__(self, opt):
        super(CannyEdgeDetector, self).__init__()

        self.device = opt.device
        # self.low_threshold = low_threshold
        # self.high_threshold = high_threshold

    def forward(self, image):
        
        b, c, h, w = image.shape
        
        edge = []
        for bs in range(b):
            img = image[bs,:]
            img = img.squeeze()
            img = img.cpu().numpy()*255
            img = img.astype(np.uint8)

            # Apply Gaussian blur
            edges = cv2.GaussianBlur(img, (5,5), 3)
            #v = np.median(image)
            # apply automatic Canny edge detection using the computed median
            #lower = int(max(0, (1.0 - 0.3) * v))
            #upper = int(min(255, (1.0 + 0.3) * v))
            edges = cv2.Canny(edges, 20, 20)
            #edges = cv2.GaussianBlur(edges, (5,5), 3)   # edge output gaussian
            #cv2.imwrite('sample_edge.jpg', edges)
            
            edges = edges/255
            edges = edges.astype(np.float32)
            edges = torch.from_numpy(edges).to(self.device)
            edges = edges.unsqueeze(dim=0)
            edges = edges.unsqueeze(dim=0)

            edge.append(edges)
        edge = torch.cat(edge, dim=0)

        return edge

class SemanticFusion(nn.Module):
    def __init__(self, opt, in_c):
        super(SemanticFusion, self).__init__()

        #  self.sca = nn.Sequential(      #enhanced feauter map
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
        #               groups=1, bias=True),
        # )
        self.conv = nn.Conv3d(in_c, 3, kernel_size=(1,3,3), padding=(0,1,1), bias=False)
        self.mask_conv = nn.Sequential(
            nn.Conv3d(3, 3, kernel_size=(1,3,3), padding=(0,1,1), bias=False),
            nn.AdaptiveAvgPool3d(1),
        )   #[1 3 1 1 1]

        self.act = nn.Sigmoid()


        self.fconv1 = nn.Conv3d(in_c, 6, kernel_size=(1,3,3), padding=(0,1,1), bias=False)
        self.kba = KBBlock_s(opt, 6, FFN_Expand=2, nset=6, k=3, gc=4, lightweight=True)
        self.fconv2 = nn.Conv3d(6, 6, kernel_size=(1,3,3), padding=(0,1,1), bias=False)
        
        bnn = [create_bnn(opt) for _ in range(6)]
        self.bnn = nn.ModuleList(bnn)
        self.bconv = nn.Conv3d(6, 3, kernel_size=(1,3,3), padding=(0,1,1), bias=False)

        self.fe_conv =  nn.Sequential(
            nn.Conv3d(3, 3, kernel_size=(1,3,3), padding=(0,1,1), bias=False),
            nn.Conv3d(3, 1, kernel_size=(1,3,3), padding=(0,1,1), bias=False)
        )

    def forward(self, x, masks):
        x_res = self.conv(x)
        mask_ch = self.mask_conv(masks)         #[1, 3, 1, 1, 1]
        #print(mask_ch.shape)
        mask_fe = masks * mask_ch     # 1 3 1 512 512
        #print(mask_fe.shape)
        mask_fe = self.act(mask_fe)
 
        kba = self.fconv1(x)
        kba = kba.squeeze(dim=2)
        kba = self.kba(kba)
        kba_out = kba.unsqueeze(dim=2)  #c=6
        bnn_in = self.fconv2(kba_out)

        bnn_out = []
        b = bnn_in.permute(0, 1, 3, 4, 2)
        for i in range(b.size(1)):    #6개
            inputs = b[: , i:i+1]
            bnn = self.bnn[i]
            b_out = bnn(inputs)
            bnn_out.append(b_out)

        bnn_out = torch.cat(bnn_out, dim=1)
        bnn_out = bnn_out.permute(0, 1, 4, 2, 3)   #1 6 1 512 512

        bnn_out = self.bconv(bnn_out)
        out = bnn_out * mask_fe 

        out = out + x_res  #c=3

        out = self.fe_conv(out)
        #out = out + bnn_in

        return out

# class SemanticFusion(nn.Module):
#     def __init__(self, opt, in_c):
#         super(SemanticFusion, self).__init__()

#         self.mask_conv = nn.Sequential(
#             nn.Conv3d(3, 3, kernel_size=(1,3,3), padding=(0,1,1), bias=False),
#             nn.Conv3d(3, 1, kernel_size=(1,3,3), padding=(0,1,1), bias=False),   
#             nn.Sigmoid()
#         )
#         self.fconv1 = nn.Conv3d(in_c, 6, kernel_size=(1,3,3), padding=(0,1,1), bias=False)
#         self.kba = KBBlock_s(opt, 6, FFN_Expand=2, nset=6, k=3, gc=4, lightweight=True)
#         self.fconv2 = nn.Conv3d(6, 1, kernel_size=(1,3,3), padding=(0,1,1), bias=False)
        
#         self.bnn = create_bnn(opt)

#         self.fe_conv =  nn.Sequential(
#             nn.Conv3d(1, 3, kernel_size=(1,3,3), padding=(0,1,1), bias=False),
#             nn.Conv3d(3, 1, kernel_size=(1,3,3), padding=(0,1,1), bias=False)
#         )

#     def forward(self, x, masks):
#         mask_fe = self.mask_conv(masks)         #[1, 3, 1, 512, 512]

#         kba = self.fconv1(x)
#         kba = kba.squeeze(dim=2)
#         kba = self.kba(kba)
#         kba_out = kba.unsqueeze(dim=2)
#         bnn_in = self.fconv2(kba_out)

#         b = bnn_in.permute(0, 1, 3, 4, 2)
#         bnn_out = self.bnn(b)
#         bnn_out= bnn_out.permute(0, 1, 4, 2, 3)
#         out = bnn_out * mask_fe

#         out = self.fe_conv(out)
#         out = out + bnn_in

#         return out
    
class Edge_net(nn.Module):
    def __init__(self, opt):
        super(Edge_net, self).__init__()
        
        self.conv1 = nn.Conv3d(5, 8, kernel_size=(1,3,3), padding=(0,1,1), bias=False)
        self.act = nn.ReLU()

        self.edge = nn.Sequential(
            ResidualB(opt),
            DenseB(opt),
            DenseB(opt),
            ResidualB(opt),
        )

        self.conv2 = nn.Conv3d(8, 5, kernel_size=(1,3,3), padding=(0,1,1), bias=False)
        
    def forward(self, input):
        x = self.conv1(input)
        x = self.act(x)
        x = self.edge(x)
        x = self.conv2(x)
        x = self.act(x)           #channel = 5
 
        return x

class ResidualB(nn.Module):
    def __init__(self, opt):
        super(ResidualB, self).__init__()

        self.res = nn.Sequential(
            nn.Conv3d(8, 8, kernel_size=(1,3,3), padding=(0,1,1), bias=False),
            nn.ReLU(),
            nn.Conv3d(8, 8, kernel_size=(1,3,3), padding=(0,1,1), bias=False)
        )
 
    def forward(self, input):
        x = input.clone()
        x = self.res(x)
        x = x + input

        return x
    
class DenseB(nn.Module):
    def __init__(self, opt):
        super(DenseB, self).__init__()
        
        self.res1 = ResidualB(opt)
        self.res2 = ResidualB(opt)
        self.res3 = ResidualB(opt)
        self.res4 = ResidualB(opt)

        self.conv = nn.Conv3d(32, 8, kernel_size=(1,3,3), padding=(0,1,1), bias=False)
        self.act = nn.ReLU()
  
    def forward(self, input):

        x = input.clone()
        x1 = self.res1(x)
        x2 = self.res2(x1+x)
        x3 = self.res3(x2+x1+x)
        x4 = self.res4(x3+x2+x1+x)

        out = torch.cat([x1, x2, x3, x4], dim=1)

        out = self.conv(out)
        out = self.act(out)
        out = out + x

        return out


class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        
        self.filter1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.filter2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.filter3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.filter4 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)


        # Gx1 = torch.tensor([[[[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]]]])
        # Gx2 = torch.tensor([[[[-2.0, 0.0, 2.0], [-4.0, 0.0, 4.0], [-2.0, 0.0, 2.0]]]])
        # Gy1 = torch.tensor([[[[-2.0, -4.0, -2.0], [0.0, 0.0, 0.0], [2.0, 4.0, 2.0]]]])
        # Gy2 = torch.tensor([[[[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]]]])

        Gx1 = torch.tensor([[[[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]]])
        Gx2 = torch.tensor([[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]])
        Gy1 = torch.tensor([[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]])
        Gy2 = torch.tensor([[[[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]]])
        
        self.filter1.weight = nn.Parameter(Gx1, requires_grad=False)
        self.filter2.weight = nn.Parameter(Gx2, requires_grad=False)
        self.filter3.weight = nn.Parameter(Gy1, requires_grad=False)
        self.filter4.weight = nn.Parameter(Gy2, requires_grad=False)

    def forward(self, x):
        edges = []
        x1 = self.filter1(x)
        edges.append(x1)
        x2 = self.filter2(x)
        edges.append(x2)
        y1 = self.filter3(x)
        edges.append(y1)
        y2 = self.filter4(x)
        edges.append(y2)

        edges = torch.cat(edges, dim=1)

        return edges

class MakeEdgeBins(nn.Module):
    def __init__(self):
        super(MakeEdgeBins, self).__init__()
        
        self.sobel = Sobel().to('cuda')

    def forward(self, x):

        edges = self.sobel(x)
        min_value = torch.min(edges)
        max_value = torch.max(edges)
        edges = (edges - min_value) * (1 / (max_value - min_value))   #all noramalization

        #ebi = edges.clone()
        edgeBins = []
    
        # 1st group
        for i in range(edges.size(1)):
            ebi = edges[:, i:i+1]
            ebj = ebi.lt(0.4).type(torch.float32)
       
            ebN = x * ebj
            edgeBins.append(ebN)
            
        ebs = torch.cat(edgeBins, dim=1)

        return ebs

class KBBlock_s(nn.Module):
    def __init__(self, opt, c, DW_Expand=2, FFN_Expand=2, nset=6, k=3, gc=4, lightweight=True):
        super(KBBlock_s, self).__init__()
        self.k, self.c = k, c
        self.nset = nset    #default 32
        dw_ch = int(c * DW_Expand)
        ffn_ch = int(FFN_Expand * c)

        self.g = c // gc    #c//gc
        self.w = nn.Parameter(torch.zeros(1, nset, c * c // self.g * self.k ** 2))
        self.b = nn.Parameter(torch.zeros(1, nset, c))
        self.init_p(self.w, self.b)   #파라미터 초기화 하는듯 

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.sca = nn.Sequential(      #1x1xdepht로 channel attention 역할
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
 
        if not lightweight:          
            self.conv11 = nn.Sequential(
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                          bias=True),
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=5, padding=2, stride=1, groups=c // 4,   #N=c//4
                          bias=True),
            )
        else:
            self.conv11 = nn.Sequential(
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                          bias=True),
                nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c,
                          bias=True),
            )

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv21 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c,
                                bias=True)

        interc = min(c, 32)
        self.conv2 = nn.Sequential(  #fusion coefficient map 생성
            nn.Conv2d(in_channels=c, out_channels=interc, kernel_size=3, padding=1, stride=1, groups=interc,
                      bias=True),
            SimpleGate(),
            nn.Conv2d(interc // 2, self.nset, 1, padding=0, stride=1),
        )

        self.conv211 = nn.Conv2d(in_channels=c, out_channels=self.nset, kernel_size=1)

        self.conv3 = nn.Conv2d(in_channels=dw_ch // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_ch, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_ch // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.dropout1 = nn.Identity()
        self.dropout2 = nn.Identity()

        self.ga1 = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)
        self.attgamma = nn.Parameter(torch.zeros((1, self.nset, 1, 1)) + 1e-2, requires_grad=True)
        self.sg = SimpleGate()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)) + 1e-2, requires_grad=True)

    def init_p(self, weight, bias=None):     ##가중치 초기화 
        init.kaiming_uniform_(weight, a=math.sqrt(5))
        if bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(bias, -bound, bound)

    def KBA(self, x, att, selfk, selfg, selfb, selfw):
        return KBAFunction.apply(x, att, selfk, selfg, selfb, selfw)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)   #layer norma
        #print('x norm1.shape:', x.shape) torch.Size([1, 6, 512, 512])
        sca = self.sca(x)   #####enhanced feature map 생성 
        #print('sca.shape:', x.shape) torch.Size([1, 6, 512, 512])
        x1 = self.conv11(x)  
        #print('x1.shape:', x1.shape) torch.Size([1, 6, 512, 512])
 
        # KBA module
        # a = self.conv2(x)
        # b = self.attgamma
        # c = self.conv211(x)
        # print("a", a.shape) #torch.Size([8, 32, 256, 256])
        # print(b.shape) #torch.Size([1, 32, 1, 1])
        # print(c.shape) #torch.Size([8, 32, 256, 256])
        att = self.conv2(x) * self.attgamma + self.conv211(x)  #channel : c >> self.nset  torch.Size([1, 32, 512, 512])
        #att = a * b + c
        #print('att', att.shape)
        uf = self.conv21(self.conv1(x))
       # print('uf', uf.shape)
        x = self.KBA(uf, att, self.k, self.g, self.b, self.w) * self.ga1 + uf
        #print("x와 x1, sca가 shape 같아야, x.shape:", x.shape)
        x = x * x1 * sca  #x=kba 결과, sca가 channel attention 같음, x1이 depthwise convolution인가 

        x = self.conv3(x)   #여기서 1x1해주고    
        x = self.dropout1(x)
        y = inp + x * self.beta    #여기 residual shortcut 
        #print('y', y.shape)

        # FFN  Feed foward network   
        #  KBnets adopts the normal FFN block with SimpleGate activation funciont to perform the position-wise non-linear transformation
        x = self.norm2(y)   
        x = self.conv4(x)    #ffn channel 
        x = self.sg(x)   #simple gate adopt 함 
        x = self.conv5(x)

        x = self.dropout2(x)
        return y + x * self.gamma

