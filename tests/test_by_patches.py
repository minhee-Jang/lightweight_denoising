import os, sys

import numpy as np
from skimage.transform import rescale
import torch
import matplotlib.pyplot as plt
import time
from torch.utils import data

utils_path = os.path.join(sys.path[0], '..')
utlis_path = os.path.abspath(utils_path)
sys.path.append(utils_path)

from data import create_dataset
from options.test_options import TestOptions

from utils.misc import print_numpy
from utils.tester import Tensor2PatchDataset


if __name__ == '__main__':
    opt = TestOptions(r'../../../data/video-sr').parse()
    # opt.data_dir = r'../../../data/video-sr'
    opt.report_ldv = True
    opt.n_threads = 0
    dataloader = create_dataset(opt)

    for i, batch in enumerate(dataloader['test']):
        x = batch['lr']
        n_frames = opt.n_frames
        _, c, n, h, w = x.shape
        print("x.shape:", x.shape)

        if not opt.report_ldv:
            tn = batch['hr']
            print("targetn.shape:", tn.shape)

        vid = batch['videoname']
        print('Videoname::', vid[0])
        

        for d in range(0, n, n_frames):
            print('[*] Frames[{}:{}]'.format(d, d+n_frames))
            mp_start_time = time.time()

            xd = x[:, :, d:d+n_frames]
            tensor_x = xd

            img_patch_dataset = Tensor2PatchDataset(opt, tensor_x)
            img_patch_dataloader = data.DataLoader(
                dataset=img_patch_dataset,
                batch_size=opt.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=opt.n_threads
            )
            
            img_shape = img_patch_dataset.get_img_shape()
            pad_img_shape = img_patch_dataset.get_padded_img_shape()

            mp_end_time = time.time()
            print("[*] input shape", tensor_x.shape)
            print("[*] Making patches: {:4f}s".format((mp_end_time - mp_start_time)))

            net_start_time = time.time()
            print("[*] Network started")
            out_list = []

            for i, img_patch in enumerate(img_patch_dataloader):
                # test with input patch
                # print('img_patch.shape:', img_patch.shape)
                out = img_patch
                out_list.append(out)
                
            net_end_time = time.time()
            print("[*] Network process: {:4f}s".format((net_end_time - net_start_time)))

            out = torch.cat(out_list, dim=0)
            
            recon_start_time = time.time()
            
            out_img = img_patch_dataset.recon_tensor_arr_patches(out)
            print('[*] Reconstructed out_img.shape:', out_img.shape)
            
            recon_end_time = time.time()
            print("[*] Reconstruction time: {:4f}s".format((recon_end_time - recon_start_time)))
            print("[*] Reconstructed volume:", out.shape)
            print("[*] Total time {:.4f}s".format(recon_end_time - mp_start_time))
            out_img = out_img.detach()
            out_np = out_img.to('cpu').numpy()
            # print('[*] out.shape:', out.shape)

            out_np[out_np > 1.0] = 1.0
            out_np[out_np < 0.0] = 0.
            out_np = out_np * 255
            out_np = np.rint(out_np)
            out_np  = out_np.astype(np.uint8)

            in_np = tensor_x.detach().to('cpu').numpy() * 255
            in_np = np.rint(in_np)
            in_np  = in_np.astype(np.uint8)


            for i in range(out_np.shape[2]):

                in_x = in_np[:, :, i].squeeze().transpose(1, 2, 0)
                out_x = out_np[:, :, i].squeeze().transpose(1, 2, 0)
                print('in_x.shape:', in_x.shape, ', out_x.shape:', out_x.shape)
                concat_img = np.concatenate((in_x, out_x), axis=1)
                plt.imshow(concat_img)
                plt.show()
