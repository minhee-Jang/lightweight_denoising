import torch.nn as nn
import numpy as np
import torch
from copy import deepcopy as dc

class EstimationDelta(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, rf, mf, move_thr, n_frames):
        _, _, _, h, w = x.shape
    
        # 프레임 한 장씩 가져와 array로 변환
        for i in range(0, n_frames):
            arr = x[:, :, i]
            arr = arr.reshape([-1, w])
            arr = np.array(arr)
            globals()['o{}'.format(i)] = arr

        rf = rf.reshape([-1, w])
        mf = mf.reshape([-1, w])
        rf = np.array(rf)
        mf = np.array(mf)
        
        out = torch.zeros(x[:, :, 0].shape)
        dout = torch.zeros(x[:, :, 0].shape)
        dout = dout.reshape([-1, w])
        dout = np.array(dout)
        # cout = dout.copy()
        # tout = dout.copy()
        
        # Dd = dout.copy()
        # mu = dout.copy()
        # f_reverse = dout.copy()
        
        cout = dc(dout)
        tout = dc(dout)
        
        Dd = dc(dout)
        mu = dc(dout)
        f_reverse = dc(dout)
        
        # Calculate delta
        for i in range(0, h):
            for j in range(0, w):
                intensityMean = 0
                total = 0
                for k in range(n_frames):
                    intensityMean += globals()['o{}'.format(k)][i][j]
                    if k == 0:  continue
                    subResult = globals()['o{}'.format(k)][i][j] - globals()['o{}'.format(k-1)][i][j]
                    total += abs(subResult)
                    
                intensityMean = (intensityMean) / n_frames
                # print(intensityMean)
                
                # air 부분 intensity high한 경우 (intensityMean이 0인 경우 존재)
                #delta = total*1000 / ((intensityMean**2)+0.001)
                delta = total*1000 / ((intensityMean)+0.001)
            
                # for figures
                Dd[i][j] = dc(total)
                
                # air 부분 dark한 경우
                # delta = total*((intensityMean**2)+1)
                
                dout[i][j] = dc(delta)
                mu[i][j] = dc(delta)
        
        min_valued = np.min(Dd)
        max_valued = np.max(Dd)
        Dd = (Dd - min_valued) * ((1-(0))/(max_valued - min_valued)) + (0)
        
        min_valuem = np.min(mu)
        max_valuem = np.max(mu)
        mu = (mu - min_valuem) * ((1-(0))/(max_valuem - min_valuem)) + (0)
        
        # Apply Gaussian filter to dout(delta result)
        import cv2
        darray = np.asarray(dout)
        darray = np.float32(darray)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!여기", darray.shape)
        kernel1d = cv2.getGaussianKernel(5,3)
        kernel2d = np.outer(kernel1d, kernel1d.transpose())
        dout = cv2.filter2D(darray, -1, kernel2d)
    
        # Normalize (delta)
        min_value = np.min(dout)
        max_value = np.max(dout)
        #dout = (dout - min_value) * ((255-(0))/(max_value - min_value)) + (0)
        dout = (dout - min_value) * ((1-(0))/(max_value - min_value)) + (0)
    
        # Combine DL and RF (기존 방법_binary moving map)
        # for i in range(0, h):
        #     for j in range(0, w):
        #         if dout[i][j] >= move_thr:
        #             cout[i][j] = mf[i][j]
        #             tout[i][j] = 255
        #         else:
        #             cout[i][j] = rf[i][j]
        #             tout[i][j] = 0
                    
        # Combine DL and RF (다른 방법_비율따라 적용)
        for i in range(0, h):
            for j in range(0, w):
                #factor = dout[i][j]/255
                factor = dout[i][j]
                cout[i][j] = factor * mf[i][j] + (1-factor) * rf[i][j]
                f_reverse[i][j] = (1-factor)
                tout[i][j] = cout[i][j] - rf[i][j]
                
                if tout[i][j] < 0:
                     tout[i][j] = 0
    
        # Normalize
        min_valuet = np.min(tout)
        max_valuet = np.max(tout)
        tout = (tout - min_valuet) * ((1-(0))/(max_valuet - min_valuet)) + (0)

        # Return output imgs
        mf = torch.tensor(mf)
        mf = mf.reshape(out.shape)
        rf = torch.tensor(rf)
        rf = rf.reshape(out.shape)
        cout = torch.tensor(cout)
        cout = cout.reshape(out.shape)
        dout = torch.tensor(dout)
        dout = dout.reshape(out.shape)
        tout = torch.tensor(tout)
        tout = tout.reshape(out.shape)
        
        Dd = torch.tensor(Dd)
        Dd = Dd.reshape(out.shape)
        mu = torch.tensor(mu)
        mu = mu.reshape(out.shape)
        f_reverse = torch.tensor(f_reverse)
        f_reverse = f_reverse.reshape(out.shape)

        return mf, rf, cout, dout, tout
        #return Dd, mu, dout, f_reverse, cout