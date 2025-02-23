import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from skimage import segmentation
import os

from data.srdata import SRData


# class make_mask(SRData):
#     def __init__(self, input):



# class Args(object):
#     train_epoch = 2 ** 5
#     mod_dim1 = 64  #
#     mod_dim2 = 32
#     gpu_id = 0

#     min_label_num = 4  # if the label number small than it, break loop
#     max_label_num = 256  # if the label number small than it, start to show result image.

# class MyNet(nn.Module):
#     def __init__(self, inp_dim, mod_dim1, mod_dim2):
#         super(MyNet, self).__init__()
#           #cnn 4 block 
#         self.seq = nn.Sequential(
#             nn.Conv2d(inp_dim, mod_dim1, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(mod_dim1),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(mod_dim2),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(mod_dim2, mod_dim1, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(mod_dim1),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(mod_dim2),
#         )

#     def forward(self, x):
#         return self.seq(x)


# class MaskE(nn.Module):  #edge mask 뽑기 (mask 넘겨 loss back-propa) #마지막에 cuda로 올려야함
#     def __init__(self):
#         super(MaskE, self).__init__()

#     def forward(self, x):
#         b, c, n, w, _ = x.shape
#         #print("x.shape:", x.shape)
#         img = np.array(x.to('cpu').detach().numpy())
#         model2 = torch.load('D:/workspaces/bi-denoising/model.pth')
      
#         #print(img.shape)

#         args = Args()
#         torch.cuda.manual_seed_all(1943)
#         np.random.seed(1943)
#         os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)  # choose GPU:0

#         maskEdge = torch.zeros(x.shape, dtype=torch.bool).to('cuda')

#         for bs in range(b):
#             for k in range(n):

#                 image= img[bs, 0, k]
#                 #print("image.shape:",image.shape)
        
#                 device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        
#                 #image = np.expand_dims(image, axis=2)
#                 #tensor = image.transpose((2, 0, 1))
#                 image = image*255
#                 #tensor = image.astype(np.float32)
#                 tensor = tensor[np.newaxis, np.newaxis, ]
#                 tensor = torch.from_numpy(tensor).to(device)

#                 #model load
#                 target= model2(tensor)
#                 im_target = target.data.cpu().numpy()
                  

#                 image_flatten = image.reshape(-1,1)
#                 #print('flatten', image_flatten)
#                 color_avg = np.random.randint(255, size=(args.max_label_num, 3))
          

#                 '''show image'''
#                 un_label, lab_inverse = np.unique(im_target, return_inverse=True, )
#                 if un_label.shape[0] < args.max_label_num:  # update show
#                     img_flatten = image_flatten.copy()
#                     if len(color_avg) != un_label.shape[0]:
#                         color_avg = [np.mean(img_flatten[im_target == label], axis=0, dtype=np.int) for label in un_label]   

#                     for lab_id, color in enumerate(color_avg):
#                         img_flatten[lab_inverse == lab_id] = color
#                         #print("최종 flatten", image_flatten.shape) #262144, 1

#                     show = img_flatten.reshape(image.shape)
#                     print(show.shape)
#                     cv2.waitKey(1)

#                 #print(batch_idx,'Loss:',  loss.item())
#                 if len(un_label) < args.min_label_num:
#                     break

#             '''save'''
#             #seg_value = np.unique(show, return_inverse=True, )
#             show_fin = np.where((show >=160) & (show <=190), 255, 0)   #특정값만 mask로 뽑음 
#             print("show:", show_fin.shape)
#             #show_fin = show_fin.transpose((2,0,1))  #1,512,512 -> 512, 512, 1
#             show_fin = torch.tensor(show_fin)
#             #show_fin =torch.unsqueeze(show_fin, axis=2)
#             #print("show_fin:", show_fin.shape)
    
#             maskEdge[bs][0][k] = show_fin
                  
#         print("mask_edge:", maskEdge.shape)
#         return maskEdge
    

# class MaskNE(nn.Module):
#     def __init__(self):
#         super(MaskNE, self).__init__()

#     def forward(self, x):
#         gradnet = GradLayer().to('cuda')
#         b, c, n, w, _ = x.shape

#         maskNonEdge = torch.zeros(x.shape, dtype=torch.bool).to('cuda')
#         # maskNon = torch.zeros(x.shape).to('cuda')
#         print(maskNonEdge.shape)

#         for bs in range(b):
#             for k in range(n):
#                 img = x[bs, :, k]
#                 img = gradnet(img)
                
#                 grad = np.array(img.to('cpu').detach().numpy())
#                 kernel1d = cv2.getGaussianKernel(5,3)
#                 kernel2d = np.outer(kernel1d, kernel1d.transpose())
#                 gauss = cv2.filter2D(grad, -1, kernel2d)
                
#                 gauss = torch.tensor(gauss).to('cuda')
#                 min_value = torch.min(gauss)
#                 max_value = torch.max(gauss)
#                 gauss = (gauss - min_value) * (1 / (max_value - min_value))
                
#                 gauss = gauss.reshape([-1, w])
#                 maskNE = gauss.le(0.1)
#                 maskNonEdge[bs][0][k] = maskNE

#         return maskNonEdge