import os, sys
utils_path = os.path.join(sys.path[0], '..')
utlis_path = os.path.abspath(utils_path)
sys.path.append(utils_path)
from data import create_dataset
from options.test_options import TestOptions
from skimage.io import imsave
import torch
import torch.nn as nn

##### BM#D!!

import os
import cv2
import time
import sys
from scipy.fftpack import dct, idct
import numpy as np


# ==================================================================================================
#                                           Preprocessing
# ==================================================================================================

def AddNoise(Img, sigma):
    GuassNoise = np.random.normal(0, sigma, Img.shape)
    noisyImg = Img + GuassNoise # float type noisy image
    
    return noisyImg  


def Initialization(Img, BlockSize, Kaiser_Window_beta):
    InitImg = np.zeros(Img.shape, dtype=float)
    InitWeight = np.zeros(Img.shape, dtype=float)
    Window = np.matrix(np.kaiser(BlockSize, Kaiser_Window_beta))
    InitKaiser = np.array(Window.T * Window)            

    return InitImg, InitWeight, InitKaiser


def SearchWindow(Img, RefPoint, BlockSize, WindowSize):

    if BlockSize >= WindowSize:
        print('Error: BlockSize is smaller than WindowSize.\n')
        exit()

    Margin = np.zeros((2,2), dtype = int)
    Margin[0, 0] = max(0, RefPoint[0]+int((BlockSize-WindowSize)/2)) # left-top x
    Margin[0, 1] = max(0, RefPoint[1]+int((BlockSize-WindowSize)/2)) # left-top y               
    Margin[1, 0] = Margin[0, 0] + WindowSize # right-bottom x
    Margin[1, 1] = Margin[0, 1] + WindowSize # right-bottom y             

    if Margin[1, 0] >= Img.shape[0]:
        Margin[1, 0] = Img.shape[0] - 1
        Margin[0, 0] = Margin[1, 0] - WindowSize

    if Margin[1, 1] >= Img.shape[1]:
        Margin[1, 1] = Img.shape[1] - 1
        Margin[0, 1] = Margin[1, 1] - WindowSize
    
    return Margin


def dct2D(A):
    
    return dct(dct(A, axis = 0, norm = 'ortho'), axis = 1, norm = 'ortho')


def idct2D(A):
    
    return idct(idct(A, axis = 0, norm = 'ortho'), axis = 1, norm = 'ortho') 

    
def PreDCT(Img, BlockSize):
    
    BlockDCT_all = np.zeros((Img.shape[0]-BlockSize, Img.shape[1]-BlockSize, BlockSize, BlockSize),\
                            dtype = float)
    
    for i in range(BlockDCT_all.shape[0]):
        for j in range(BlockDCT_all.shape[1]):
            Block = Img[i:i+BlockSize, j:j+BlockSize]
            BlockDCT_all[i, j, :, :] = dct2D(Block.astype(np.float64))
            
    return BlockDCT_all


def ComputePSNR(Img1, Img2):
    if Img1.size != Img2.size:
        print('ERROR: two images should be in same size in computing PSNR.\n')
        sys.exit()
    
    Img1 = Img1.astype(np.float64)
    Img2 = Img2.astype(np.float64)
    RMSE = np.sqrt(np.sum((Img1-Img2)**2)/Img1.size)
    
    return 20*np.log10(255./RMSE)




# ==================================================================================================
#                                         Basic estimate    
# ==================================================================================================

def Step1_Grouping(noisyImg, RefPoint, BlockDCT_all, BlockSize, ThreDist, MaxMatch, WindowSize):

    # initialization
    WindowLoc = SearchWindow(noisyImg, RefPoint, BlockSize, WindowSize)
    Block_Num_Searched = (WindowSize-BlockSize+1)**2                    # number of searched blocks  
    BlockPos = np.zeros((Block_Num_Searched, 2), dtype = int)
    BlockGroup = np.zeros((Block_Num_Searched, BlockSize, BlockSize), dtype = float)
    Dist = np.zeros(Block_Num_Searched, dtype = float)
    RefDCT = BlockDCT_all[RefPoint[0],RefPoint[1], :, :]
    match_cnt = 0


    # Block searching and similarity (distance) computing
    for i in range(WindowSize-BlockSize+1):
        for j in range(WindowSize-BlockSize+1):
            SearchedDCT = BlockDCT_all[WindowLoc[0, 0]+i, WindowLoc[0, 1]+j, :, :]
            dist = Step1_ComputeDist(RefDCT, SearchedDCT)
            if dist < ThreDist:
                BlockPos[match_cnt, :] = [WindowLoc[0, 0]+i, WindowLoc[0, 1]+j]
                BlockGroup[match_cnt, :, :] = SearchedDCT
                Dist[match_cnt] = dist
                match_cnt += 1

    
    if match_cnt <= MaxMatch:
        # less than MaxMatch similar blocks founded, return similar blocks
        BlockPos = BlockPos[:match_cnt, :]
        BlockGroup = BlockGroup[:match_cnt, :, :]
    
    else:
        # more than MaxMatch similar blocks founded, return MaxMatch similarest blocks
        idx = np.argpartition(Dist[:match_cnt], MaxMatch)  # indices of MaxMatch smallest distances
        BlockPos = BlockPos[idx[:MaxMatch], :]
        BlockGroup = BlockGroup[idx[:MaxMatch], :]
    
    return BlockPos, BlockGroup


def Step1_ComputeDist(BlockDCT1, BlockDCT2):
    
    if BlockDCT1.shape != BlockDCT1.shape:
        print('ERROR: two DCT Blocks are not at the same shape in step1 computing distance.\n')
        sys.exit()
        
    elif BlockDCT1.shape[0] != BlockDCT1.shape[1]:
        print('ERROR: DCT Block is not square in step1 computing distance.\n')
        sys.exit()
    
    BlockSize = BlockDCT1.shape[0]
    
    #!!
    if sigma > 40:
       ThreValue = lamb2d * sigma
       BlockDCT1 = np.where(abs(BlockDCT1) < ThreValue, 0, BlockDCT1)
       BlockDCT2 = np.where(abs(BlockDCT2) < ThreValue, 0, BlockDCT2)

    return np.linalg.norm(BlockDCT1 - BlockDCT2)**2 / (BlockSize**2)


def Step1_3DFiltering(BlockGroup):
    ThreValue = lamb3d * sigma
    nonzero_cnt = 0
    
    # since 2D transform has been done, we do 1D transform, hard-thresholding and inverse 1D 
    # transform, the inverse 2D transform is left in aggregation processing
    
    for i in range(BlockGroup.shape[1]):
        for j in range(BlockGroup.shape[2]):
            ThirdVector = dct(BlockGroup[:, i, j], norm = 'ortho') # 1D DCT
            ThirdVector[abs(ThirdVector[:]) < ThreValue] = 0.
            nonzero_cnt += np.nonzero(ThirdVector)[0].size
            BlockGroup[:, i, j] = list(idct(ThirdVector, norm = 'ortho'))

    return BlockGroup, nonzero_cnt


def Step1_Aggregation(BlockGroup, BlockPos, basicImg, basicWeight, basicKaiser, nonzero_cnt):
    if nonzero_cnt < 1:
        BlockWeight = 1.0 * basicKaiser
    
    else:
        BlockWeight = (1./(sigma**2 * nonzero_cnt)) * basicKaiser

    for i in range(BlockPos.shape[0]):
        basicImg[BlockPos[i, 0]:BlockPos[i, 0]+BlockGroup.shape[1],\
                 BlockPos[i, 1]:BlockPos[i, 1]+BlockGroup.shape[2]]\
                                 += BlockWeight * idct2D(BlockGroup[i, :, :])
        basicWeight[BlockPos[i, 0]:BlockPos[i, 0]+BlockGroup.shape[1],\
                    BlockPos[i, 1]:BlockPos[i, 1]+BlockGroup.shape[2]] += BlockWeight


def BM3D_Step1(noisyImg):

    # preprocessing
    BlockSize = Step1_BlockSize
    ThreDist = Step1_ThreDist
    MaxMatch = Step1_MaxMatch
    WindowSize = Step1_WindowSize
    spdup_factor = Step1_spdup_factor
    basicImg, basicWeight, basicKaiser = Initialization(noisyImg, BlockSize, Kaiser_Window_beta)
    BlockDCT_all = PreDCT(noisyImg, BlockSize)

    # block-wise estimate with speed-up factor 
    for i in range(int((noisyImg.shape[0]-BlockSize)/spdup_factor)+2):
        for j in range(int((noisyImg.shape[1]-BlockSize)/spdup_factor)+2):
            RefPoint = [min(spdup_factor*i, noisyImg.shape[0]-BlockSize-1), \
                        min(spdup_factor*j, noisyImg.shape[1]-BlockSize-1)]
            BlockPos, BlockGroup = Step1_Grouping(noisyImg, RefPoint, BlockDCT_all, BlockSize, \
                                                  ThreDist, MaxMatch, WindowSize)
            BlockGroup, nonzero_cnt = Step1_3DFiltering(BlockGroup)
            Step1_Aggregation(BlockGroup, BlockPos, basicImg, basicWeight, basicKaiser, nonzero_cnt)

    basicWeight = np.where(basicWeight == 0, 1, basicWeight)
    basicImg[:, :] /= basicWeight[:, :]

    return basicImg




# ==================================================================================================
#                                         Final estimate    
# ==================================================================================================
    
def Step2_Grouping(basicImg, noisyImg, RefPoint, BlockSize, ThreDist, MaxMatch, WindowSize, 
                   BlockDCT_basic, BlockDCT_noisy):

    # initialization (same as Step1)
    WindowLoc = SearchWindow(basicImg, RefPoint, BlockSize, WindowSize)
    Block_Num_Searched = (WindowSize-BlockSize+1)**2     
    BlockPos = np.zeros((Block_Num_Searched, 2), dtype = int)
    BlockGroup_basic = np.zeros((Block_Num_Searched, BlockSize, BlockSize), dtype = float)
    BlockGroup_noisy = np.zeros((Block_Num_Searched, BlockSize, BlockSize), dtype = float)
    Dist = np.zeros(Block_Num_Searched, dtype = float)
    match_cnt = 0
    
    # Block searching and similarity (distance) computing
    # Note the distance computing method is different from that of Step1
    for i in range(WindowSize-BlockSize+1):
        for j in range(WindowSize-BlockSize+1):
            SearchedPoint = [WindowLoc[0, 0]+i, WindowLoc[0, 1]+j]
            dist = Step2_ComputeDist(basicImg, RefPoint, SearchedPoint, BlockSize)
            if dist < ThreDist:
                BlockPos[match_cnt, :] = SearchedPoint
                Dist[match_cnt] = dist
                match_cnt += 1
         
    if match_cnt <= MaxMatch:
        # less than MaxMatch similar blocks founded, return similar blocks
        BlockPos = BlockPos[:match_cnt, :]
    
    else:
        # more than MaxMatch similar blocks founded, return MaxMatch similarest blocks
        idx = np.argpartition(Dist[:match_cnt], MaxMatch)  # indices of MaxMatch smallest distances
        BlockPos = BlockPos[idx[:MaxMatch], :]
        
    for i in range(BlockPos.shape[0]):
        SimilarPoint = BlockPos[i, :]
        BlockGroup_basic[i, :, :] = BlockDCT_basic[SimilarPoint[0], SimilarPoint[1], :, :]
        BlockGroup_noisy[i, :, :] = BlockDCT_noisy[SimilarPoint[0], SimilarPoint[1], :, :]
        
    BlockGroup_basic = BlockGroup_basic[:BlockPos.shape[0], :, :]
    BlockGroup_noisy = BlockGroup_noisy[:BlockPos.shape[0], :, :]

    return BlockPos, BlockGroup_basic, BlockGroup_noisy
    

def Step2_ComputeDist(img, Point1, Point2, BlockSize):
    Block1 = (img[Point1[0]:Point1[0]+BlockSize, Point1[1]:Point1[1]+BlockSize]).astype(np.float64)
    Block2 = (img[Point2[0]:Point2[0]+BlockSize, Point2[1]:Point2[1]+BlockSize]).astype(np.float64)
    
    return np.linalg.norm(Block1-Block2)**2 / (BlockSize**2)


def Step2_3DFiltering(BlockGroup_basic, BlockGroup_noisy):
    Weight = 0
    coef = 1.0 / BlockGroup_noisy.shape[0]
    
    for i in range(BlockGroup_noisy.shape[1]):
        for j in range(BlockGroup_noisy.shape[2]):
            Vec_basic = dct(BlockGroup_basic[:, i, j], norm = 'ortho')
            Vec_noisy = dct(BlockGroup_noisy[:, i, j], norm = 'ortho')
            Vec_value = Vec_basic**2 * coef
            Vec_value /= (Vec_value + sigma**2) # pixel weight
            Vec_noisy *= Vec_value
            Weight += np.sum(Vec_value)
            BlockGroup_noisy[:, i, j] = list(idct(Vec_noisy, norm = 'ortho'))
    
    if Weight > 0:
        WienerWeight = 1./(sigma**2 * Weight)
    
    else:
        WienerWeight = 1.0
                
    return BlockGroup_noisy, WienerWeight


def Step2_Aggregation(BlockGroup_noisy, WienerWeight, BlockPos, finalImg, finalWeight, finalKaiser):
    BlockWeight = WienerWeight * finalKaiser

    for i in range(BlockPos.shape[0]):
        finalImg[BlockPos[i, 0]:BlockPos[i, 0]+BlockGroup_noisy.shape[1],\
                 BlockPos[i, 1]:BlockPos[i, 1]+BlockGroup_noisy.shape[2]]\
                                 += BlockWeight * idct2D(BlockGroup_noisy[i, :, :])

        finalWeight[BlockPos[i, 0]:BlockPos[i, 0]+BlockGroup_noisy.shape[1],\
                    BlockPos[i, 1]:BlockPos[i, 1]+BlockGroup_noisy.shape[2]] += BlockWeight
    

def BM3D_Step2(basicImg, noisyImg):

    # parameters setting
    BlockSize = Step2_BlockSize
    ThreDist = Step2_ThreDist
    MaxMatch = Step2_MaxMatch
    WindowSize = Step2_WindowSize
    spdup_factor = Step2_spdup_factor
    finalImg, finalWeight, finalKaiser = Initialization(basicImg, BlockSize, Kaiser_Window_beta)
    BlockDCT_noisy = PreDCT(noisyImg, BlockSize)
    BlockDCT_basic = PreDCT(basicImg, BlockSize)

    # block-wise estimate with speed-up factor
    for i in range(int((basicImg.shape[0]-BlockSize)/spdup_factor)+2):
        for j in range(int((basicImg.shape[1]-BlockSize)/spdup_factor)+2):
            RefPoint = [min(spdup_factor*i, basicImg.shape[0]-BlockSize-1), \
                        min(spdup_factor*j, basicImg.shape[1]-BlockSize-1)]
            BlockPos, BlockGroup_basic, BlockGroup_noisy = Step2_Grouping(basicImg, noisyImg, \
                                                                          RefPoint, BlockSize, \
                                                                          ThreDist, MaxMatch, \
                                                                          WindowSize, \
                                                                          BlockDCT_basic, \
                                                                          BlockDCT_noisy)
            BlockGroup_noisy, WienerWeight = Step2_3DFiltering(BlockGroup_basic, BlockGroup_noisy)
            Step2_Aggregation(BlockGroup_noisy, WienerWeight, BlockPos, finalImg, finalWeight, \
                              finalKaiser)
    
    finalWeight = np.where(finalWeight == 0, 1, finalWeight)
    finalImg[:, :] /= finalWeight[:, :]

    return finalImg

##### BM#D!! end


if __name__ == '__main__':
    
    opt = TestOptions(r'../../../data/multi-frame-image-enhancement').parse()
    opt.n_frames = 1
    dataloader = create_dataset(opt)
    
    test_dir = opt.test_results_dir + '-BM3D/0714_2/'
    os.makedirs(test_dir, exist_ok=True)
    
    #================================== Parameters initialization ==================================
    sigma = 25 # variance of the noise
    lamb2d = 2.0
    lamb3d = 2.7
    Step1_ThreDist = 2500 # threshold distance
    Step1_MaxMatch = 16 # max matched blocks
    Step1_BlockSize = 8
    Step1_spdup_factor = 3 # pixel jump for new reference block
    Step1_WindowSize = 39 # search window size  
    Step2_ThreDist = 400
    Step2_MaxMatch = 32
    Step2_BlockSize = 8
    Step2_spdup_factor = 3
    Step2_WindowSize = 39
    Kaiser_Window_beta = 2.0
    #===============================================================================================
    from PIL import Image
    
    for i, batch in enumerate(dataloader['test']):
        tAll = batch['hr']    #original
        bs, c, n, h, w = tAll.shape
        #tAll = tAll.to('cuda')
       
        for j in range(n):
            print('j: ', j)
            idx = '{:03d}'.format(j)
            
            out = torch.zeros(tAll[:, :, 0].shape)
            img = tAll[:, :, j:j+1]
            img = img.view(h, w)
            img = np.array(img, dtype=np.uint8)
            iimg = Image.fromarray(img)
            bm3d_pathi = os.path.join(test_dir, idx + '-si.tiff')
            iimg.save(bm3d_pathi)
            print("si ok")

            noisy_img = AddNoise(img, sigma)
            nimg = np.array(noisy_img, dtype=np.uint8)
            nimg = Image.fromarray(nimg)
            bm3d_path0 = os.path.join(test_dir, idx + '-s0.tiff')
            nimg.save(bm3d_path0)
            print("s0 ok")
            
            # STEP1
            basic_img = BM3D_Step1(noisy_img)
            #basic_img_uint = np.zeros(img.shape)
            #cv2.normalize(basic_img, basic_img_uint, 0, 255, cv2.NORM_MINMAX, dtype=-1)
            #basic_img_uint = basic_img_uint.astype(np.uint8)
            # basic_img = basic_img.astype(np.uint8)
            # basic_img = torch.tensor(basic_img)
            # basic_img = basic_img.reshape(out.shape)
            #bout = basic_img.detach().to('cpu').numpy().transpose(0, 2, 3, 1).squeeze()
            bm3d_path1 = os.path.join(test_dir, idx + '-s1.tiff')
            imsave(bm3d_path1, basic_img)
            
            # STEP2
            final_img = BM3D_Step2(basic_img, noisy_img)
            #cv2.normalize(final_img, final_img, 0, 255, cv2.NORM_MINMAX, dtype=-1)
            # final_img = final_img.astype(np.uint8)
            # final_img = torch.tensor(final_img)
            # final_img = final_img.reshape(out.shape)
            #fout =  final_img.detach().to('cpu').numpy().transpose(0, 2, 3, 1).squeeze()
            bm3d_path2 = os.path.join(test_dir, idx + '-s2.tiff')
            imsave(bm3d_path2, final_img)
