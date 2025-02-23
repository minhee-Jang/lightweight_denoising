
import random
import numpy as np
import skimage.color as sc

import torch

def augment(*args, hflip=True, rot=True):
    # print('len(args):', len(args))
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        # print('type(img)', type(img))
        if img.ndim == 2:
            if hflip: img = img[:, ::-1].copy()
            if vflip: img = img[::-1, :].copy()
            if rot90: img = img.transpose(1, 0).copy()
        elif img.ndim == 3:
            if hflip: img = img[:, ::-1, :].copy()
            if vflip: img = img[::-1, :, :].copy()
            if rot90: img = img.transpose(1, 0, 2).copy()
            
        return img

    # for arg in args:
        # arg = [_augment(a) for a in arg]
    args = [[_augment(a) for a in arg] for arg in args]
    # lr_n = [_augment(lr) for lr in lr_n]
    # hr_n = [_augment(hr) for hr in hr_n]

    return args

def get_patch(*args, patch_size=160, scale=2, center_crop=False):
    #print('args[0][0].shape:', args[0][0].shape)
    ih, iw = args[0][0].shape[:2]

    tp = patch_size
    ip = tp // scale

    if center_crop:
        ix = iw // 2 - ip // 2
        iy = ih // 2 - ip // 2
    
    else:
        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)

    tx, ty = scale * ix, scale * iy

    args = [
        [a[iy:iy + ip, ix:ix + ip] for a in args[0]],
        *[[a[ty:ty + tp, tx:tx + tp] for a in arg] for arg in args[1:]]
    ]

    return args

def set_channel(*args, n_channels=3):

    def _set_channel(img):
        #print(img.ndim)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    # lr_n = [_set_channel(lr) for lr in lr_n]
    # hr_n = [_set_channel(hr) for hr in hr_n]

    # for arg in args:
    args = [[_set_channel(a) for a in arg] for arg in args]

    return args

def np2Tensor(*args, pixel_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(1.0 / pixel_range)
  
        return tensor

    # lr_n = [_np2Tensor(lr) for lr in lr_n]
    # hr_n = [_np2Tensor(hr) for hr in hr_n]
    # for arg in args:
    args = [[_np2Tensor(a) for a in arg] for arg in args]

    return args

def concat_tensor(*args):
    def _concate(img_l):
        img_l = [img.unsqueeze(1) for img in img_l]
        # for img in img_l:
        #     print(img.shape)
        # print('img_l[0].shape:', img_l[0].shape)
        cat_img = torch.cat(img_l, dim=1)
        # print('cat_img.shape:', cat_img.shape)
        return cat_img

    # lr_n = _concate(lr_n)
    # hr_n = _concate(hr_n)
    # for arg in args:
        # print('arg:', arg)
    args = [_concate(arg) for arg in args]
    return args

