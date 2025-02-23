import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch import einsum
from einops import rearrange
from einops.layers.torch import Rearrange
from .base_model import BaseModel
from models.convs.wavelet import serialize_swt, SWTForward, SWTInverse, unserialize_swt

class LDN06_2(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(batch_size=1)
        parser.set_defaults(n_frames=1)
        
        # model specification
        parser.add_argument('--wavelet_func', type=str, default='haar', #'bior2.2',
            help='wavelet function ex: haar, bior2.2, or etc.')
        parser.add_argument('--swt_lv', type=int, default=1,
            help='Level of stationary wavelet transform')
        parser.add_argument('--lv', type=int, default=4,
            help='the number of image after wavelet transform')
        parser.add_argument('--vit_patch_size', type=int, default=8,
            help='the patch size of vit input')
        parser.add_argument('--n_heads', type=int, default=1,
            help='the number of attention heads')
        
        if is_train:
            parser.add_argument('--content_loss', type=str, choices=['l1', 'l2'], default='l2',
                help='loss function (l1, l2)')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.n_frames = opt.n_frames
        self.model_names = ['net']
        self.net = create_model(opt).to(self.device)

        # Define losses and optimizers
        if self.is_train:
            if opt.content_loss == 'l1':
                self.loss_criterion = nn.L1Loss()
            elif opt.content_loss == 'l2':
                self.loss_criterion = nn.MSELoss()

            self.optimizer_names = ['optimizer']
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), eps=1e-8, weight_decay=0)

            self.optimizers = []
            self.optimizers.append(self.optimizer)

        self.mse_loss_criterion = nn.MSELoss()

    def set_input(self, input):
        self.x = input['lr'].to(self.device)
        
        bs, c, n, h, w = self.x.shape
        self.x = self.x.view(bs, c*n, h, w)
        
        if 'hr' in input:
            self.target = input['hr'].to(self.device).view(bs, c*n, h, w)

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
    return DeepDenoisingNet(opt)


class PatchEmbedding(nn.Module):
    def __init__(self, opt, in_channels=1, img_size=512):
        super(PatchEmbedding, self).__init__()
        
        patch_size = opt.vit_patch_size
        emb_size = patch_size**2*in_channels
        
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e')
        )
        
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2, emb_size))
        
    def forward(self, x):
        x1 = self.projection(x)  #(1, 4096, 64)
        x2 = x1 + self.positions  #(1, 4096, 64)
        return x2


class CrossAttention(nn.Module):
    def __init__(self, opt, in_channels=1, img_size=512, dropout=0):
        super(CrossAttention, self).__init__()
        self.patch_size = opt.vit_patch_size
        self.emb_size = self.patch_size**2*in_channels
        self.img_size = img_size
        self.w = self.img_size // self.patch_size
        self.n_heads = opt.n_heads
        
        self.queries = nn.Linear(self.emb_size, self.emb_size)
        self.keys = nn.Linear(self.emb_size**2, 1)
        self.values = nn.Linear(self.emb_size**2, 1)
        
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(self.emb_size, self.emb_size)
        
        self.pe_q = PatchEmbedding(opt)
        self.pe_k = PatchEmbedding(opt)
        
        self.ln = nn.LayerNorm(self.emb_size)
        
    def forward(self, x1, x2):
        x11 = self.pe_q(x1)
        x22 = self.pe_k(x2)
        
        x11 = self.ln(x11)
        x22 = self.ln(x22)
        
        queries = rearrange(self.queries(x11), "b n (h d) -> b h n d", h=self.n_heads)  # [1, 1, 4096, 64]
        xx= rearrange(x22, "b e2 e -> b e e2")
        keys = rearrange(self.keys(xx), "b n (h d)-> b h n d", h=self.n_heads)   # [1, 1, 64, 1]
        values = rearrange(self.values(xx), "b n (h d) -> b h n d", h=self.n_heads)   # [1, 1, 64, 1]
        
        energy = einsum('bhqd, bhdk -> bhqk', queries, keys)
        scaling = self.emb_size**(1/2)
        att = F.softmax(energy, dim=-1)/scaling
        att = self.att_drop(att)    # [1, 1, 4096, 1]
        
        out = einsum('bhal, bhvl -> bhav', att, values)    # [1, 1, 4096, 64]
        out = rearrange(out, "b h n d -> b n (h d)")    # [1, 4096, 64]
        
        out = self.projection(out)
        out = rearrange(out, "b (h w) (s1 s2 c) -> b c (h s1) (w s2)", s1=self.patch_size, s2=self.patch_size, w=self.w, h=self.w)

        return out


def diff_x(input, r):
    left   = input[:, :,         r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:         ] - input[:, :,           :-2 * r - 1]
    right  = input[:, :,        -1:         ] - input[:, :, -2 * r - 1:    -r - 1]
    output = torch.cat([left, middle, right], dim=2)
    return output

def diff_y(input, r):
    left   = input[:, :, :,         r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:         ] - input[:, :, :,           :-2 * r - 1]
    right  = input[:, :, :,        -1:         ] - input[:, :, :, -2 * r - 1:    -r - 1]
    output = torch.cat([left, middle, right], dim=3)
    return output

class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()
        self.r = r
    def forward(self, x):
        return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)


class GuidedFilter(nn.Module):
    def __init__(self, radius, eps):
        super(GuidedFilter, self).__init__()
        self.eps = eps
        self.boxfilter = BoxFilter(radius)
        
    def forward(self, guide, src):
        ones = torch.ones_like(guide)
        N = self.boxfilter(ones)

        mean_I = self.boxfilter(guide) / N
        mean_p = self.boxfilter(src) / N
        mean_Ip = self.boxfilter(guide*src) / N
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = self.boxfilter(guide*guide) / N
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + self.eps)
        b = mean_p - a * mean_I

        mean_a = self.boxfilter(a) / N
        mean_b = self.boxfilter(b) / N

        q = mean_a * guide + mean_b
        return q


class DeepGuidedFilter(nn.Module):
    def __init__(self, radius=2, eps=1e-8):
        super(DeepGuidedFilter, self).__init__()
        
        self.lr = MDCN()
        self.gf = GuidedFilter(radius, eps)

    def forward(self, x):
        return self.gf(x, self.lr(x))


class SWTModule(nn.Module):
    def __init__(self, opt):
        super(SWTModule, self).__init__()
        self.swt = SWTForward(J=opt.swt_lv, wave=opt.wavelet_func)
    
    def forward(self, x):
        x = self.swt(x)
        x = serialize_swt(x)
        
        return x


class iSWTModule(nn.Module):
    def __init__(self, opt):
        super(iSWTModule, self).__init__()
        self.iswt = SWTInverse(wave=opt.wavelet_func)
    
    def forward(self, x):
        x = unserialize_swt(x, J=1, C=1)
        x = self.iswt(x)
        
        return x


class MDCN(nn.Module):
    def __init__(self, in_channels=1, mid_channels=32):
        super(MDCN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=2**1,  dilation=2**1,  bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=2**2,  dilation=2**2,  bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=2**3,  dilation=2**3,  bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=2**4,  dilation=2**4,  bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=2**5,  dilation=2**5,  bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0, dilation=1)
        )
    
    def forward(self, x):
        x = self.net(x)
        return x


class FDN(nn.Module):
    def __init__(self, opt):
        super(FDN, self).__init__()
        self.lv = opt.lv
        
        self.swtm = SWTModule(opt)
        self.iswtm = iSWTModule(opt)
        
        dgf = [DeepGuidedFilter() for _ in range(self.lv)]
        self.dgf = nn.ModuleList(dgf)

    def forward(self, x):
        xe = self.swtm(x)
        xe = [self.dgf[j](xe[:, j:j+1]) for j in range(self.lv)]
        xe = torch.cat(xe, dim=1)
        xe = self.iswtm(xe)
        
        return xe


class DeepDenoisingNet(nn.Module):
    def __init__(self, opt):
        super(DeepDenoisingNet, self).__init__()
        self.fdn = FDN(opt)
        self.ca = CrossAttention(opt)

    def forward(self, x):
        xe = self.fdn(x)
        o_ca = self.ca(xe, x)
        out = xe * o_ca + x
        
        return out
