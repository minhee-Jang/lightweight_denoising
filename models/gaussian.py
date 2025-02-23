import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
import numpy as np
import os
import imageio

class gauss_img(nn.Module):
    def __init__(self):
        super(gauss_img, self).__init__()

    def forward(self, x):

        b, c, n, w, _ = x.shape

        maskEdge = torch.zeros(x.shape, dtype=torch.bool).to('cuda')
        #mask = torch.zeros(x.shape).to('cuda')
        maskNonEdge = torch.zeros(x.shape, dtype=torch.bool).to('cuda')
        #maskNon = torch.zeros(x.shape).to('cuda')
        blurrer = T.GaussianBlur(kernel_size=(7, 13), sigma=(9, 11))

        for bs in range(b):
            for k in range(n):
                img = x[bs, :, k]
                print("img.shape", img.shape)    # torch.Size([1, 512, 512]) full_img
                # img_s = img.to('cpu').permute(1,2,0)
                # trans = T.ToPILImage()
                # img_s=trans(img)
                # img_s.save("./img_s.png")
                # plt.imshow(img_s)
                # plt.show


                gauss = blurrer(img)
                # gauss = [blurrer(img) for _ in range(4)]
                
                min_value = torch.min(gauss)
                max_value = torch.max(gauss)
                gauss = (gauss - min_value) * (1 / (max_value - min_value))
                #rint(img.shape)
                gauss = gauss.reshape([-1, w])
                #print(gauss.shape)

                # img_s2=trans(gauss)
                # img_s2.save("./mask_edge.png")
                maskE = gauss.ge(0.1) #0.1보다 큰
                # print(maskNE)
                # exit()
                maskNE = gauss.lt(0.1)  #0.1보다 작은

                # img_s3=trans(gauss)
                # img_s3.save("./img_s3.png")
          
                
                maskEdge[bs][0][k] = maskE
                #mask[bs][0][k] = maskE
        
                # maskNE = gauss.lt(0.1)
                maskNonEdge[bs][0][k] = maskNE
                # maskNon[bs][0][k] = maskNE
                

        return maskEdge, maskNonEdge
