##### 36.5를 향해서 #####
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from models.convs import common
from models.common.bilateralw import create_bnn

from models.convs.wavelet import SWTForward, SWTInverse
import numpy as np
import os
import sys
import cv2
import imageio

class EMGB(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.add_argument('--threshold1', type=int, default = 30)
        parser.add_argument('--threshold2', type=int, default = 55)
        parser.add_argument('--alpha', type=float, default=0.5)
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
        self.model_names = ['net']
        self.net = create_model(opt).to(self.device)
        self.canny =  CannyEdgeDetector(opt).to(self.device)

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

        self.ones = torch.ones(1, 1, 1, 512, 512).to(self.device) 

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
  
        self.out, self.out_e = self.net(self.x, self.masks)
      
        ebt = self.target.clone()
        ebt = ebt.squeeze(dim=2)
        ebt = self.canny(ebt)
        self.edge_gt = ebt.unsqueeze(dim=2)

        for name, param in self.net.named_parameters():
            print(name, param)
         
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
        self.loss_i = self.loss_criterion(self.target, self.out)
        #self.loss_e = self.loss_criterion(self.edge_gt, self.out_e)

        self.mapX = self.edge_gt*self.out
        self.mapY = self.edge_gt*self.target
        self.loss_map = self.loss_criterion(self.mapY, self.mapX)

        self.loss = self.loss_i + self.loss_map
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
            predicted_video.append(out)
            predicted_idxs.append(predicted_idx)

        predicted_video = torch.cat(predicted_video, dim=2)

        return predicted_video, predicted_idxs

def create_model(opt):
    return BilateralDeepNet(opt)


class SemanticFusion(nn.Module):
    def __init__(self, opt):
        super(SemanticFusion, self).__init__()

        self.mask_conv = nn.Sequential(
            nn.Conv3d(3, 3, kernel_size=(1,3,3), padding=(0,1,1), bias=False),
            nn.Conv3d(3, 1, kernel_size=(1,3,3), padding=(0,1,1), bias=False),   
            nn.Sigmoid()
        )
        self.fconv1 = nn.Conv3d(2, 1, kernel_size=(1,3,3), padding=(0,1,1), bias=False)
        
        self.bnn = create_bnn(opt)

        self.fe_conv =  nn.Sequential(
            nn.Conv3d(1, 3, kernel_size=(1,3,3), padding=(0,1,1), bias=False),
            nn.Conv3d(3, 1, kernel_size=(1,3,3), padding=(0,1,1), bias=False)
        )

    def forward(self, x, masks):
        mask_fe = self.mask_conv(masks)         #[1, 3, 1, 512, 512]

        bnn_in = self.fconv1(x)

        b = bnn_in.permute(0, 1, 3, 4, 2)
        bnn_out = self.bnn(b)
        bnn_out= bnn_out.permute(0, 1, 4, 2, 3)
        out = bnn_out * mask_fe

        out = self.fe_conv(out)
        out = out + bnn_in

        return out
    
class Edge_net(nn.Module):
    def __init__(self, opt):
        super(Edge_net, self).__init__()
        
        self.conv1 = nn.Conv3d(2, 8, kernel_size=(1,3,3), padding=(0,1,1), bias=False)
        self.act = nn.ReLU()

        self.edge = nn.Sequential(
            ResidualB(opt),
            DenseB(opt),
            DenseB(opt),
            ResidualB(opt),
        )

        self.conv2 = nn.Conv3d(8, 2, kernel_size=(1,3,3), padding=(0,1,1), bias=False)
        
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

class BilateralDeepNet(nn.Module):
    def __init__(self, opt):
        super(BilateralDeepNet, self).__init__()

        self.device = opt.device
    
        self.makeEB = CannyEdgeDetector(opt).to(self.device)
        self.fe = nn.Conv3d(2, 2, kernel_size=(1,3,3), padding=(0,1,1), bias=False) 
        self.fe1 = nn.Conv3d(5, 1, kernel_size=(1,3,3), padding=(0,1,1), bias=False) 

        # fusion = [SemanticFusion(opt) for _ in range(5)]  #fusion out 1
        # self.fusion = nn.ModuleList(fusion)
        # edge = [Edge_net(opt) for _ in range(5)]    #edge out 4
        # self.Edge = nn.ModuleList(edge)

        self.fusion1 = SemanticFusion(opt)
        self.fusion2 = SemanticFusion(opt)
        self.fusion3 = SemanticFusion(opt)

        self.Edge1 = Edge_net(opt)
        self.Edge2 = Edge_net(opt)
        self.Edge3 = Edge_net(opt)

        self.fe2 = nn.Conv3d(3, 2, kernel_size=(1,3,3), padding=(0,1,1), bias=False)
        self.edge_extractor = nn.Conv3d(2, 1, kernel_size=(1,3,3), padding=(0,1,1), bias=False)
        self.recon_hi = nn.Conv3d(3, 1, kernel_size=(1,3,3), padding=(0,1,1), bias=False)
        
    def forward(self, x, masks):

        ebs = x.clone()
        ebs = ebs.squeeze(dim=2)
        ebs = self.makeEB(ebs)
        ebs = ebs.unsqueeze(dim=2)
        #[1, 1, 1, 512, 512]

        x_in = torch.cat((x, ebs), dim=1)   #c = 2
        input = self.fe(x_in)   #c = 2
        #x_b = self.fe1(x_b)  #bnn input   c =1 
        
        edge1 = self.Edge1(x_in)   #out c = 2
        x_b1 = x_in + edge1 #2
        out1 = self.fusion1(x_b1, masks)    # out 1

        edge2 = edge1 + x_in #2 유지 
        edge2 = self.Edge2(edge2) 
        x_b2 = torch.cat([out1, ebs], dim=1) 
        x_b2 = x_b2 + edge2
        out2 = self.fusion2(x_b2, masks)

        edge3 = edge2 + x_in #2유지 
        edge3 = self.Edge3(edge3) 
        x_b3 = torch.cat([out2, ebs], dim=1) 
        x_b3 = x_b3 + edge3
        out3 = self.fusion3(x_b3, masks)



        out_f = torch.cat([out1, out2, out3], dim=1)

        b_out = self.fe2(out_f)   #2로 
        b_out = b_out + input     

        edge = self.edge_extractor(edge3)    #1

        out_f = torch.cat((b_out, edge), dim =1)
        out = self.recon_hi(out_f)

        out = out + x

        return out, edge

class CannyEdgeDetector(nn.Module):
    def __init__(self, opt):
        super(CannyEdgeDetector, self).__init__()

        self.device = opt.device
        self.lower = opt.threshold1
        self.upper = opt.threshold2
        # self.low_threshold = low_threshold
        # self.high_threshold = high_threshold
  

    def forward(self, image):

        image = image.squeeze()

        image = image.cpu().numpy()*255
        image = image.astype(np.uint8)

        # Apply Gaussian blur
        edges = cv2.GaussianBlur(image, (5,5), 3)
        v = np.median(image)
	    # apply automatic Canny edge detection using the computed median
        # lower = int(max(0, (1.0 - 0.3) * v))
        # upper = int(min(255, (1.0 + 0.3) * v))
        lower = torch.randint(0, 255, (1), requires_grad=True)
        upper = torch.randint(0, 255, (1), reguires_grad=True)
        print(lower, upper)
        edges = cv2.Canny(edges, lower, upper)
        #edges = cv2.GaussianBlur(edges, (5,5), 3)   # edge output gaussian
        #cv2.imwrite('sample_edge.jpg', edges)
        
        edges = edges/255
        edges = edges.astype(np.float32)
        edges = torch.from_numpy(edges).to(self.device)
        edges = edges.unsqueeze(dim=0)
        edges = edges.unsqueeze(dim=0)
     
        return edges


            
class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        
        self.filter1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.filter2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

        Gx1 = torch.tensor([[[[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]]])
        Gy1 = torch.tensor([[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]])

        self.filter1.weight = nn.Parameter(Gx1, requires_grad=False)
        self.filter2.weight = nn.Parameter(Gy1, requires_grad=False)

    def forward(self, x):
    
        grad_x = self.filter1(x)
        grad_y = self.filter2(x)

        return grad_x, grad_y