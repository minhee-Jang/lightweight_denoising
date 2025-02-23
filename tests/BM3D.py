"""
https://webpages.tuni.fi/foi/GCF-BM3D/
https://pypi.org/project/bm3d/
This implementation is based on Y. Mäkinen, L. Azzari, A. Foi,
"Exact Transform-Domain Noise Variance for Collaborative Filtering of Stationary Correlated Noise",
in Proc. 2019 IEEE Int. Conf. Image Process. (ICIP), pp. 185-189.

The whole process uses the functions of image-denoising which is implemented by our pytorch project framework.
Thus, you should put image-denoising project like as follows:

+-/image-denoising
|-/ml-image-denoising
"""

import os, sys
import time
import numpy as np
import torch
import argparse

from bm3d import bm3d, bm3d_rgb

sys_path = os.path.join(sys.path[0], '..', 'image-denoising')
sys_path = os.path.abspath(sys_path)
sys.path.append(sys_path)

from options.test_options import TestOptions
from data import create_dataset
from utils.tester import save_tensors, save_metrics, save_summary, calc_metrics, calc_ssim

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options

    opt.is_train = False
    opt.test_random_patch = False
    opt.test_ratio = 1.0
    opt.multi_gpu = 0
    # hard-code some parameters for test
    opt.n_threads = 0   # test code only supports num_threads = 1

    dataloader = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    # hard-code some parameters for test

    opt.test_results_dir = os.path.join(opt.data_dir, 'ml-test-results', 'bm3d')
    os.makedirs(opt.test_results_dir, exist_ok=True)

    print('opt:', opt)
    start_time = time.time()

    avg_loss = 0.0
    avg_psnr = 0.0
    noise_avg_loss = 0.0
    noise_avg_psnr = 0.0
    avg_ssim = 0.0
    base_avg_ssim = 0.0
    total_n = 0
    itr = 0
    for di, batch in enumerate(dataloader['test'],1):

            #print(batch)
        x, target, filenames = batch['lr'], batch['hr'], batch['filenames']

        for i in range(x.shape[2]):
        
            # print("x.shape:", x.shape)
            x_t = x[:, :, i].to(opt.device).detach()
            x_np = x_t.to('cpu').numpy()
            x_np = x_np.transpose(0, 2, 3, 1).squeeze()
            # print("x_np.shape:", x_np.shape)

            x_std = np.std(x_np)
            if opt.n_channels == 1:
                out_np = bm3d(x_np, sigma_psd=x_std/4)
                #print('bm3d 수행')
            else:
                out_np = bm3d_rgb(x_np, sigma_psd=x_std)

            # print("out_np.shape:", out_np.shape)

            if opt.n_channels == 1:
                out_np = np.expand_dims(out_np, axis=2)
            out_np = np.expand_dims(out_np, axis=0)

            out_np = out_np.transpose((0, 3, 1, 2))
            out = torch.from_numpy(out_np).float().to(opt.device)

            target_t = target[:, :, i].to(opt.device).detach()
            fn = filenames[i][0]

            #print(type(x), type(out), type(target_t))
            #print(x.shape, out.shape, target_t.shape)
            tensors_dict = {
                "x": x_t,
                "out": out,
                "target": target_t,
                "filename": fn
            }

            noise_loss, noise_psnr, batch_loss, batch_psnr = calc_metrics(tensors_dict)
            base_ssim, batch_ssim = calc_ssim(tensors_dict)
            end_time = time.time()

            avg_loss += batch_loss
            avg_psnr += batch_psnr
            noise_avg_loss += noise_loss
            noise_avg_psnr += noise_psnr

            avg_ssim += batch_ssim
            base_avg_ssim += base_ssim

            itr += 1

            print("Test {:.3f}s => Image({}/{}): Noise Loss: {:.8f}, Noise PSNR: {:.8f}, Loss: {:.8f}, PSNR: {:.8f}".format(
                end_time - start_time, i+1, x.shape[2], noise_loss.item(), noise_psnr.item(), batch_loss.item(), batch_psnr.item()
            ))
            save_tensors(opt, tensors_dict)
            save_metrics(opt, itr, fn, noise_loss, noise_psnr, batch_loss, batch_psnr)
            print("Average Noise Loss: {:.8f}, Average Noise PSNR: {:.8f}, Average Loss: {:.8f}, Average PSNR: {:.8f}".format(
                noise_avg_loss / itr, noise_avg_psnr / itr, avg_loss / itr, avg_psnr / itr
            ))
            print("** SSIM => Base SSIM: {:.8f}, SSIM: {:.8f}, Average Base SSIM: {:.8f}, Average SSIM: {:.8f}".format(
                base_ssim, batch_ssim, base_avg_ssim / itr, avg_ssim / itr
            ))


            # save_tensors(opt, tensors_dict)
            # save_metrics(opt, itr, fn, noise_loss, noise_psnr, batch_loss, batch_psnr)

    noise_avg_loss, noise_avg_psnr = noise_avg_loss / itr, noise_avg_psnr / itr
    avg_loss, avg_psnr = avg_loss / itr, avg_psnr / itr
        
    print("===> Test on {} - Noise Average Loss: {:.8f}, Noise Average PSNR: {:.8f}, Average Loss: {:.8f}, Average PSNR: {:.8f}".format(
        opt.test_datasets, noise_avg_loss, noise_avg_psnr, avg_loss, avg_psnr
    ))
    save_summary(opt, di, noise_avg_loss, noise_avg_psnr, avg_loss, avg_psnr)