import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
import numpy as np
import os
import imageio

class GradLayer(nn.Module):     #[1, 120, 120]
    def __init__(self):
        super(GradLayer, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):
        x_v = F.conv2d(x, self.weight_v, padding=1)
        x_h = F.conv2d(x, self.weight_h, padding=1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)

        return x


class GradientMask(nn.Module):
    def __init__(self):
        super(GradientMask, self).__init__()

    def forward(self, x):
        gradnet = GradLayer().to('cuda')
        b, c, n, w, _ = x.shape

        maskEdge = torch.zeros(x.shape, dtype=torch.bool).to('cuda')
        mask = torch.zeros(x.shape).to('cuda')
        maskNonEdge = torch.zeros(x.shape, dtype=torch.bool).to('cuda')
        maskNon = torch.zeros(x.shape).to('cuda')

        for bs in range(b):
            for k in range(n):
                img = x[bs, :, k]
                img = gradnet(img)
                
                # grad = np.array(img.to('cpu').detach().numpy())
                # kernel1d = cv2.getGaussianKernel(7,5)
                # kernel2d = np.outer(kernel1d, kernel1d.transpose())
                # gauss = cv2.filter2D(grad, -1, kernel2d)
                # gauss = torch.tensor(gauss).to('cuda')
                
                blurrer = T.GaussianBlur(kernel_size=(7, 13), sigma=(9, 11))
                gauss = blurrer(img)
                # gauss = [blurrer(img) for _ in range(4)]
                
                min_value = torch.min(gauss)
                max_value = torch.max(gauss)
                gauss = (gauss - min_value) * (1 / (max_value - min_value))
                
                gauss = gauss.reshape([-1, w])
                maskE = gauss.lt(0.1)
                # maskE = gauss.ge(0.1)
                maskEdge[bs][0][k] = maskE
                mask[bs][0][k] = maskE
                maskNE = gauss.ge(0.1)
                # maskNE = gauss.lt(0.1)
                maskNonEdge[bs][0][k] = maskNE
                maskNon[bs][0][k] = maskNE
                
                # pppath = "./hahahaha1109/02"
                # os.makedirs(pppath, exist_ok=True)
                # fmt = '.tiff'
                # filenamee = str(bs) + str(k) + "__ne__" 
                # filenamene = str(bs) + str(k)  + "__e__"
                # out_fn_pathe = os.path.join(pppath, filenamee + fmt)
                # out_fn_pathne = os.path.join(pppath, filenamene + fmt)
                # mk = mask[bs, :, k].detach().to('cpu').numpy().transpose((1, 2, 0)).squeeze()
                # mkK = maskNon[bs, :, k].detach().to('cpu').numpy().transpose((1, 2, 0)).squeeze()
                # imageio.imwrite(out_fn_pathe, mk)
                # imageio.imwrite(out_fn_pathne, mkK)
                # print("Writing {}".format(os.path.abspath(out_fn_pathe)))

        return maskEdge, maskNonEdge
    