import os
import time
import imageio

import torch
from torch.utils import data
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity

"""
This function make output from patches of image.
To do so, opt.test_patches shoule be set True.
If opt.test_patches is False, we test the network with shole image. --test_image -> set opt.test_patches False

We test divided patches because waveletdl cannot make the proper output when the input is whole image,
thus we divide the image into the patch with trained patch size
To do this, we need to pad the image, divide into patches, run the network, reconstruct the output image from patches,
and unpad the output image.

If SWT is not used, the whole image can be used.
"""

def test_net_by_tensor_patches(opt, model, tensor_x):
    mp_start_time = time.time()

    img_patch_dataset = Tensor2PatchDataset(opt, tensor_x)
    img_patch_dataloader = data.DataLoader(dataset=img_patch_dataset,
                            batch_size=opt.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=opt.n_threads)

    img_shape = img_patch_dataset.get_img_shape()
    pad_img_shape = img_patch_dataset.get_padded_img_shape()

    mp_end_time = time.time()
    if not opt.is_train: print("[*]", tensor_x.shape)
    if not opt.is_train: print("[*] Making patches: {:4f}s".format((mp_end_time - mp_start_time)))

    net_start_time = time.time()
    if not opt.is_train: print("[*] Network started")
    out_list = []

    for batch in img_patch_dataloader:
        # print("batch:", batch.shape)
        input = {
            'x': batch,
            'target': None
        }
        with torch.no_grad():
            model.set_input(input)
            model.test()

        out = model.out
        # out = out.to('cpu').detach().numpy()
        out_list.append(out)
        
    net_end_time = time.time()
    if not opt.is_train: print("[*] Network process: {:4f}s".format((net_end_time - net_start_time)))

    out = torch.cat(out_list, dim=0)
    # print("out.shape:", out.shape)
    
    recon_start_time = time.time()
    
    out_img = recon_tensor_arr_patches(out, pad_img_shape[3], pad_img_shape[2], opt.patch_size, opt.patch_offset)
    out_img = unpad_tensor(out_img, opt.patch_offset, img_shape)
    
    recon_end_time = time.time()
    if not opt.is_train: print("[*] Reconstruction time: {:4f}s".format((recon_end_time - recon_start_time)))
    if not opt.is_train: print("[*] Total time {:.4f}s".format(recon_end_time - mp_start_time))

    return out_img

"""
This Dataset class converts one tensor image into divided patches
"""
class Tensor2PatchDataset(data.Dataset):
    def __init__(self, opt, tensor_img):
        super(Tensor2PatchDataset, self).__init__()
        self.tensor_img = tensor_img
        # self.tensor_img_shape = tensor_img.shape
        self.opt = opt
        self.patch_size = opt.patch_size
        self.patch_offset = opt.patch_offset

        self.padded_tensor = self.pad_tensor(tensor_img)

        # self.padded_tensor_shape = self.padded_tensor.shape
        self.tensor_patches = self.make_tensor_arr_patches(self.padded_tensor)

    def __getitem__(self, idx):
        patch = self.tensor_patches[idx]
        return patch

    def __len__(self):
        return len(self.tensor_patches)

    def get_img_shape(self):
        return self.tensor_img.shape

    def get_padded_img_shape(self):
        return self.padded_tensor.shape

    def pad_tensor(self, tensor_img):
        patch_size = self.patch_size
        patch_offset = self.patch_offset
        # print("tensor_img.shape:", tensor_img.shape)
        stride = patch_size - 2 * patch_offset
        h, w = tensor_img.shape[-2:]

        # print('h:{}, w:: {}'.format(h, w))

        rw = (patch_offset + w) % stride
        rh = (patch_offset + h) % stride
        w_stride_pad_size = stride - rw
        h_stride_pad_size = stride - rh

        stride_pad_w = patch_offset + w + w_stride_pad_size
        stride_pad_h = patch_offset + h + h_stride_pad_size

        w_pad_size = w_stride_pad_size + patch_size
        h_pad_size = h_stride_pad_size + patch_size

        # print('dim:', tensor_img.ndim)
        if tensor_img.ndim == 4:
            npad = (patch_offset, h_pad_size, patch_offset, w_pad_size)
            padded_tensor_img = F.pad(tensor_img, npad, mode='reflect')
        elif tensor_img.ndim == 5:
            npad = (patch_offset, h_pad_size, patch_offset, w_pad_size, 0, 0)
            padded_tensor_img = F.pad(tensor_img, npad, mode='replicate')

        return padded_tensor_img

    def make_tensor_arr_patches(self, tensor_img):
        patch_size = self.patch_size
        patch_offset = self.patch_offset

        # print('make_tensor:', tensor6
        # _img.shape)
        
        if tensor_img.ndim == 4:
            bs, c, h, w = tensor_img.shape
        elif tensor_img.ndim == 5:
            bs, c, d, h, w = tensor_img.shape

        assert bs == 1

        stride = patch_size - 2 * patch_offset
        mod_h = h - np.mod(h - patch_size, stride)
        mod_w = w - np.mod(w - patch_size, stride)
        
        num_patches = (mod_h // stride) * (mod_w // stride)

        if tensor_img.ndim == 4:
            patch_arr = torch.zeros((num_patches, c, patch_size, patch_size), dtype=tensor_img.dtype)
        elif tensor_img.ndim == 5:
            patch_arr = torch.zeros((num_patches, c, d, patch_size, patch_size), dtype=tensor_img.dtype)
        
        ps = patch_size
        patch_idx = 0
        for y in range(0, mod_h - stride + 1, stride):
            for x in range(0, mod_w - stride + 1, stride):
                if tensor_img.ndim == 4:
                    patch = tensor_img[:, :, y:y+ps, x:x+ps]
                elif tensor_img.ndim == 5:
                    patch = tensor_img[:, :, :, y:y+ps, x:x+ps]

                # print('{} patch.shape: {}'.format(patch_idx, patch.shape))
                patch_arr[patch_idx] = patch
                patch_idx += 1

        return patch_arr

    def recon_tensor_arr_patches(self, patch_arr):
        height, width = self.padded_tensor.shape[-2:]
        patch_size = self.patch_size
        patch_offset = self.patch_offset
        stride = patch_size - 2 * patch_offset

        if patch_arr.ndim == 4:
            _, c, _, _ = patch_arr.shape
            tensor_img = torch.zeros((1, c, height, width), dtype=patch_arr.dtype)
        elif patch_arr.ndim == 5:
            _, c, d, _, _ = patch_arr.shape
            tensor_img = torch.zeros((1, c, d, height, width), dtype=patch_arr.dtype)

        mod_h = height - np.mod(height - 2 * patch_offset, stride)
        mod_w = width - np.mod(width - 2 * patch_offset, stride)

        ps = patch_size
        po = patch_offset

        patch_idx = 0
        for y in range(0, mod_h - (patch_size - patch_offset) + 1, stride):
            for x in range(0, mod_w - (patch_size - patch_offset) + 1, stride):
                patch = patch_arr[patch_idx]
                    
                if patch_arr.ndim == 4:
                    tensor_img[:, : ,y+po:y+ps-po, x+po:x+ps-po] = patch[:, po:-po, po:-po]
                elif patch_arr.ndim == 5:
                    tensor_img[:, : , :, y+po:y+ps-po, x+po:x+ps-po] = patch[:, :, po:-po, po:-po]
                patch_idx += 1

        # Unpadding
        h, w = self.tensor_img.shape[-2:]
        if patch_arr.ndim == 4:
            tensor_ret = tensor_img[:, :, patch_offset:patch_offset+h, patch_offset:patch_offset+w]
        elif patch_arr.ndim == 5:
            tensor_ret = tensor_img[:, :, :, patch_offset:patch_offset+h, patch_offset:patch_offset+w]
        return tensor_ret

"""
Some functions for save results
Is it better to create class object rather than functions?
"""

def calc_metrics(tensors_dict):
    x = tensors_dict['x']
    out = tensors_dict['out']
    target = tensors_dict['target']


   
    out[out>1.0] = 1.0
    out[out<0.0] = 0.0

    mse_criterion = torch.nn.MSELoss()
    
    noise_loss = mse_criterion(x, target)
    noise_psnr = 10 * torch.log10(1 / noise_loss)

    out_loss = mse_criterion(out, target)
    out_psnr = 10 * torch.log10(1 / out_loss)

    return noise_loss, noise_psnr, out_loss, out_psnr

def calc_ssim(tensors_dict):
    x = tensors_dict['x']
    out = tensors_dict['out']
    target = tensors_dict['target']

    out[out>1.0] = 1.0
    out[out<0.0] = 0.0

    total_noise_ssim = 0
    total_batch_ssim = 0

    bs, c, h, w = x.shape
    ##### !!
    # print('calc x.shape:', x.shape)
    # print('calc out.shape:', out.shape)
    # print('calc target.shape:', target.shape)
    # x = x.permute(0, 2, 3, 1)
    # out = out.permute(0, 2, 3, 1)
    # target = target.permute(0, 2, 3, 1)

    for i in range(bs):
        xi = x[i]
        outi = out[i]
        targeti = target[i]

        xi = xi.squeeze()
        outi = outi.squeeze()
        targeti = targeti.squeeze()

        xi = xi.detach().to('cpu').numpy()
        outi = outi.detach().to('cpu').numpy()
        targeti = targeti.detach().to('cpu').numpy()

        if c == 1:
            noise_ssim = structural_similarity(xi, targeti)
            batch_ssim = structural_similarity(outi, targeti)
        else:
            noise_ssim = structural_similarity(xi, targeti, multichannel=True)
            batch_ssim = structural_similarity(outi, targeti, multichannel=True)

        total_noise_ssim += noise_ssim
        total_batch_ssim += batch_ssim

    return total_noise_ssim / bs, total_batch_ssim / bs

def save_tensors_m(opt, tensors_dict, outname):
    bs, c, h, w = tensors_dict['x'].shape

    tensor_x = tensors_dict['x']
    tensor_out = tensors_dict['out']
    tensor_target = tensors_dict['target']
    filename = tensors_dict['filename']

    test_results_dir= os.path.join(opt.test_results_dir, opt.test_dataset, outname)
    os.makedirs(test_results_dir, exist_ok=True)

    x = tensor_x[0].detach().to('cpu').numpy()
    out = tensor_out[0].detach().to('cpu').numpy()
    target = tensor_target[0].detach().to('cpu').numpy()

    # print("x.shape:", x.shape)
    x = x.transpose((1, 2, 0)).squeeze()
    out = out.transpose((1, 2, 0)).squeeze()
    target = target.transpose((1, 2, 0)).squeeze()
    
    ##### !!
    fmt = '.tiff' if c == 1 else '.png'
    out_fn_path = os.path.join(test_results_dir, opt.model + '-' + filename + fmt)
    print("Writing {}".format(os.path.abspath(out_fn_path)))
    imageio.imwrite(out_fn_path, out)

def save_tensors(opt, tensors_dict):
    bs, c, h, w = tensors_dict['x'].shape

    tensor_x = tensors_dict['x']
    tensor_out = tensors_dict['out']
    tensor_target = tensors_dict['target']
    filename = tensors_dict['filename']

    test_results_dir= os.path.join(opt.test_results_dir, opt.test_dataset)
    os.makedirs(test_results_dir, exist_ok=True)

    x = tensor_x[0].detach().to('cpu').numpy()
    out = tensor_out[0].detach().to('cpu').numpy()
    target = tensor_target[0].detach().to('cpu').numpy()

    # print("x.shape:", x.shape)
    x = x.transpose((1, 2, 0)).squeeze()
    out = out.transpose((1, 2, 0)).squeeze()
    target = target.transpose((1, 2, 0)).squeeze()
    
    ##### !!
    fmt = '.tiff' if c == 1 else '.png'
    out_fn_path = os.path.join(test_results_dir, opt.model  + '-' + filename + fmt)
    print("Writing {}".format(os.path.abspath(out_fn_path)))
    imageio.imwrite(out_fn_path, out)

    # out[out > 1.0] = 1.0
    # out[out < 0.0] = 0.

    # if opt.compare:
    #     compare_test_results_dir = os.path.join(test_results_dir, opt.model + '-compare')
    #     os.makedirs(compare_test_results_dir, exist_ok=True)


    # if c == 3:
    #     x = x * 255
    #     out = out * 255
    #     target = target * 255
    #     x = np.rint(x)
    #     out = np.rint(out)
    #     target = np.rint(target)
    #     x  = x.astype(np.uint8)
    #     out  = out.astype(np.uint8)
    #     target  = target.astype(np.uint8)

        

    #     fmt = '.tiff' if c == 1 else '.png'
    #     out_fn_path = os.path.join(test_results_dir, opt.model + '-' + filename + fmt)
    #     # if os.path.isfile(out_fn_path):
    #     #     out_fn_path = os.path.join(test_results_dir, '1-out-' + filenames[b] + fmt)
    #     # if os.path.isfile(compare_fn_path):
    #     #     compare_fn_path = os.path.join(compare_test_results_dir, '1-' + filenames[b] + fmt)

    #     print("Writing {}".format(os.path.abspath(out_fn_path)))
    #     imageio.imwrite(out_fn_path, out)

    #     if opt.compare:
    #         compare_img = np.concatenate((x, out, target), axis=1)
    #         compare_fn_path = os.path.join(compare_test_results_dir, filename + fmt)
    #         imageio.imwrite(compare_fn_path, compare_img)

def save_metrics(opt, fi, filename, noise_loss, noise_psnr, out_loss, out_psnr):
    report_path = os.path.join(opt.test_results_dir, '{}-report.csv'.format(opt.test_dataset))

    if not os.path.isfile(report_path):
        with open(report_path, 'w') as f:
            f.write("dataset,idx,filename,noise_loss,noise_psnr,loss,psnr\n")

    with open(report_path, 'a') as f:
        f.write("{},{},{},{:.8f},{:.8f},{:.8f},{:.8f}\n".format(
            opt.test_dataset, fi, filename, noise_loss, noise_psnr, out_loss, out_psnr
        ))


def save_metrics_m(opt, fi, filename, noise_psnr, out_loss, out_psnr, outname):
    report_path = os.path.join(opt.test_results_dir, '{}-report_{}.csv'.format(opt.test_dataset, outname))

    if not os.path.isfile(report_path):
        with open(report_path, 'w') as f:
            f.write("dataset,idx,filename,noise_psnr,loss,psnr\n")

    with open(report_path, 'a') as f:
        f.write("{},{},{},{:.8f},{:.8f},{:.8f}\n".format(
            opt.test_dataset, fi, filename, noise_psnr, out_loss, out_psnr
        ))

def save_summary(opt, avg_noise_loss, avg_noise_psnr, avg_noise_ssim, avg_loss, avg_psnr, avg_ssim):
    report_path = os.path.join(opt.test_results_dir, '{}-summary.txt'.format(opt.test_dataset))

    if not os.path.isfile(report_path):
        with open(report_path, 'w') as f:
            f.write("*** SUMMARY ***\n")

    with open(report_path, 'a') as f:
        f.write("{} - Average Noise Loss: {:.8f}, Average Noise PSNR: {:.8f}, Average Noise SSIM: {:.8f}, Average Loss: {:.8f} Average PSNR: {:.8f}, Average SSIM: {:.8f}\n".format(
            opt.test_dataset, avg_noise_loss, avg_noise_psnr, avg_noise_ssim, avg_loss, avg_psnr, avg_ssim
        ))