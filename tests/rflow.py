import os, sys
import glob

import numpy as np
from skimage.transform import rescale
import torch
import matplotlib.pyplot as plt
from skimage.io import imsave

utils_path = os.path.join(sys.path[0], '..')
utlis_path = os.path.abspath(utils_path)
sys.path.append(utils_path)

from data import create_dataset
from options.train_options import TrainOptions
from options.test_options import TestOptions

from utils.misc import print_numpy
from models.common.spynet import SpyNet
from models import create_model
from models.common.recursive_filter import RecursiveFilter

from models.common.flow_vis import flow_to_color
from models.mmedit.models.common.flow_warp import flow_warp

def normalize_flow(flow):
    x = flow.clone()
    bs, _, h, w = flow.shape
    x = x.view(flow.size(0), -1)
    x -= x.min(1, keepdim=True)[0]
    x /= x.max(1, keepdim=True)[0]
    x = x.view(bs, 1, h, w)
    return x

if __name__ == '__main__':
    move_thr = 0.5
    opt = TestOptions(r'../../../data/multi-frame-image-enhancement').parse()
    dataloader = create_dataset(opt)

    n_frames = opt.n_frames
    center = n_frames // 2

    spynet = SpyNet().to('cuda')
    mfcnn = create_model(opt)
    recursive_filter = RecursiveFilter(w=0.8)

    test_dir = opt.test_results_dir + '-recursive-flow/16/'
    os.makedirs(test_dir, exist_ok=True)
    mfcnn_dir = os.path.join(test_dir, 'mfcnn')
    os.makedirs(mfcnn_dir, exist_ok=True)
    recursive_dir = os.path.join(test_dir, 'recursive')
    os.makedirs(recursive_dir, exist_ok=True)
    warped_dir = os.path.join(test_dir, 'wapred')
    os.makedirs(warped_dir, exist_ok=True)
    flow_dir = os.path.join(test_dir, 'flow')
    os.makedirs(flow_dir, exist_ok=True)

    for i, batch in enumerate(dataloader['test']):
        xn = batch['lr']
        tn = batch['hr']
        filenames = batch['filenames']
        print("xn.shape:", xn.shape)
        # print(filenames)

        bs, c, n, h, w = xn.shape

        rf_out = recursive_filter(xn)
        rf_out = rf_out.to('cuda')
        print('rf_out.shape:', rf_out.shape)
        
        for i in range(n - n_frames):
            print('i:', i)
            xi = xn[:, :, i:i+n_frames]
            ti = tn[:, :, i:i+n_frames]
            rf = rf_out[:, :, i:i+n_frames]
            fns = filenames[i:i+n_frames]
            bs, c, n, h, w = xi.shape

            frame1 = xi[:, :, :-1].to('cuda').permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
            frame2 = xi[:, :, 1:].to('cuda').permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)

            print('rf.shape:', rf.shape)
            print('frame1:', frame1.shape)
            print('frame2:', frame2.shape)
            # print('fns:', fns)

            flow = spynet(frame2, frame1)
            print('flow.shape:', flow.shape)

            flow_center = flow[center:center+1]
            print('flow_center.shape:', flow_center.shape)
            flow_intensity = torch.sqrt(flow_center[:, 0:1].square() + flow_center[:, 1:2].square())
            flow_intensity[flow_intensity<move_thr] = 0.000001
            flow_r = normalize_flow(flow_intensity)

            # print('flow_center[:, 0:1]:', flow_center[:, 0:1])
            # print('flow_center[:, 1:2]:', flow_center[:, 1:2])
            print('flow_intensity.shape:', flow_intensity.shape)
            # print('flow_intensity:', flow_intensity)

            tensors_dict = {
                'lr': xi,
                'target': ti,
                # 'filename': batch['filename']
            }
            mfcnn.set_input(tensors_dict)
            mfcnn.test()
            mfcnn_out = mfcnn.out
            print('mfcnn_out.shape:', mfcnn_out.shape)
            fn_center = fns[center][0]
            print('fn_center:', fn_center)

            rf_center = rf[:, :, center]
            out = torch.mul(rf_center, 1- flow_r) + torch.mul(mfcnn_out, flow_r)
            print('out,shape:', out.shape)

            lr_x_np = xi[:, :, center].detach().numpy().transpose(0, 2, 3, 1).squeeze()
            recursive_center = rf[:, :, center].detach().to('cpu').numpy().transpose(0, 2, 3, 1).squeeze()
            flow_np = flow_center[0].detach().to('cpu').numpy().transpose(1, 2, 0)
            flow_color = flow_to_color(flow_np)

            lr_x = xi[:, :, center].to('cuda')
            flow_center_warp = flow_center.permute(0, 2, 3, 1).to('cuda')
            warped_t = flow_warp(lr_x, flow_center_warp)
            warped_np = warped_t.detach().to('cpu').numpy().transpose(0, 2, 3, 1).squeeze()


            out_path = os.path.join(test_dir, fn_center + '.tiff')
            out_np = out.detach().to('cpu').numpy().transpose(0, 2, 3, 1).squeeze()
            imsave(out_path, out_np)

            mfcnn_path = os.path.join(mfcnn_dir, fn_center + '.tiff')
            mfcnn_out_np = mfcnn_out.detach().to('cpu').numpy().transpose(0, 2, 3, 1).squeeze()
            imsave(mfcnn_path, mfcnn_out_np)

            recursive_path = os.path.join(recursive_dir, fn_center + '.tiff')
            imsave(recursive_path, recursive_center)

            warped_path = os.path.join(warped_dir, fn_center + '.tiff')
            imsave(warped_path, warped_np)

            flow_path = os.path.join(flow_dir, fn_center + '.tiff')
            imsave(flow_path, flow_color)





