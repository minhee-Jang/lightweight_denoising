import os
import time
import torch

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from utils.tester import test_net_by_tensor_patches, calc_metrics, calc_ssim, save_tensors, save_metrics, save_summary


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    opt.is_train = False
    opt.load_epoch = 'best'
    opt.in_mem = False
    opt.multi_gpu = False
    # hard-code some parameters for test
    opt.n_threads = 0   # test code only supports num_threads = 1

    dataloader = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    model.eval()
    start_time = time.time()

    pd = opt.n_frames


    print("*** Test on {}***".format(opt.test_dataset))
    avg_loss = 0.0
    avg_psnr = 0.0
    base_avg_loss = 0.0
    base_avg_psnr = 0.0
    avg_ssim = 0.0
    base_avg_ssim = 0.0
    total_n = 0

    itr = 0
    for bi, batch in enumerate(dataloader['test'], 1):
        vn = batch['videoname'][0]
        #print(f'predicting video {vn}')
        predicted_imgs, predicted_idxs = model.predict(batch)
        
        x = batch['lr']
        #print(x.shape)
        target = batch['hr']
        out = predicted_imgs
        filenames = batch['filenames']
        
        for i in range(x.shape[2]):
            # print(i)
            # print(x.shape[2])
            x_t = x[:, :, i].to(opt.device).detach()
            target_t = target[:, :, i].to(opt.device).detach()
    
            out_t = out[:, :, i].to(opt.device).detach()
        
            fn = filenames[i][0]
            tensors_dict = {
                'x': x_t,
                'out': out_t,
                'target': target_t,
                'filename': fn
            }
            base_loss, base_psnr, batch_loss, batch_psnr = calc_metrics(tensors_dict)
            base_ssim, batch_ssim = calc_ssim(tensors_dict)

            base_avg_loss += base_loss
            base_avg_psnr += base_psnr
            avg_loss += batch_loss
            avg_psnr += batch_psnr

            avg_ssim += batch_ssim
            base_avg_ssim += base_ssim

            end_time = time.time()
            itr += 1

            print("** Test {:.3f}s => Image({}/{})/Video({}/{}): Base Loss: {:.8f}, Base PSNR: {:.8f}, Loss: {:.8f}, PSNR: {:.8f}".format(
                end_time - start_time, i+1, predicted_imgs.shape[2], bi, len(dataloader['test']), base_loss.item(), base_psnr.item(), batch_loss.item(), batch_psnr.item()
            ))
            print("** Test Average Base Loss: {:.8f}, Average Base PSNR: {:.8f}, Average Loss: {:.8f}, Average PSNR: {:.8f}".format(
                base_avg_loss / itr, base_avg_psnr / itr, avg_loss / itr, avg_psnr / itr
            ))
            print("** SSIM => Base SSIM: {:.8f}, SSIM: {:.8f}, Average Base SSIM: {:.8f}, Average SSIM: {:.8f}".format(
                base_ssim, batch_ssim, base_avg_ssim / itr, avg_ssim / itr
            ))
            save_tensors(opt, tensors_dict)
            save_metrics(opt, itr, fn, base_loss, base_psnr, batch_loss, batch_psnr)

        total_n += len(predicted_imgs)

    avg_loss, avg_psnr = avg_loss / itr, avg_psnr / itr
    base_avg_loss, base_avg_psnr = base_avg_loss / itr, base_avg_psnr / itr

    print("===> Test on {} - Base Average Loss: {:.8f}, Base Average PSNR: {:.8f},Average Loss: {:.8f}, Average PSNR: {:.8f}".format(
        opt.test_dataset, base_avg_loss, base_avg_psnr,avg_loss, avg_psnr
    ))
    save_summary(opt, base_avg_loss, base_avg_psnr, base_avg_ssim, avg_loss, avg_psnr, avg_ssim)