import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from copy import deepcopy as dc
# import os
# import imageio

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
        # print("x: ", x.shape)

        return x


class Gradient(nn.Module):
    def __init__(self):
        super(Gradient, self).__init__()

    def forward(self, x, out, target):
        gradnet = GradLayer()   #
        b, _, n, w, _ = x.shape

        gradout = torch.zeros(x.shape)
        maskout_edge = dc(gradout)
        masktarget_edge = dc(gradout)

        for bs in range(b):
            for k in range(n):
                img = x[bs, :, k]
                outimg = out[bs, :, k]
                targetimg = target[bs, :, k]
                
                img = gradnet(img)  #
                
                # imgarray = np.array(img.detach().numpy())
                outimgarray = np.array(outimg.detach().numpy())
                targetimgarray = np.array(targetimg.detach().numpy())
                # print("out: ", outimgarray.shape)
                
                # gradient 구하기
                # grad1 = cv2.Sobel(imgarray, cv2.CV_8U, 0, 1, ksize=5)
                # grad2 = cv2.Sobel(imgarray, cv2.CV_8U, 1, 0, ksize=5)
                # grad = grad1 + grad2
                
                grad = np.array(img.detach().numpy())
                
                # gaussian 강하게 쓰기
                kernel1d = cv2.getGaussianKernel(5,3)
                kernel2d = np.outer(kernel1d, kernel1d.transpose())
                gauss = cv2.filter2D(grad, -1, kernel2d)

                # Normalize (after gauss)
                min_value = np.min(gauss)
                max_value = np.max(gauss)
                gauss = (gauss - min_value) * (1/(max_value - min_value))
                
                mo = np.zeros(gauss.shape)
                mt = np.zeros(gauss.shape)
                
                # print("gauss: ", gauss.shape)
                
                for i in range(w):
                    for j in range(w):
                        mo[0][i][j] = outimgarray[0][i][j]
                        mt[0][i][j] = targetimgarray[0][i][j]
                        # if gauss[0][i][j] > 0.1:   # 이 부분이 강한 edge
                        #     mo[0][i][j] = outimgarray[0][i][j]
                        #     mt[0][i][j] = targetimgarray[0][i][j]

                maskout_edge[bs, :, k] = torch.tensor(mo).requires_grad_(True)
                masktarget_edge[bs, :, k] = torch.tensor(mt).requires_grad_(True)

        return maskout_edge, masktarget_edge


