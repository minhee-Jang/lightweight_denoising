import os, sys
import torch
from skimage.io import imsave
from skimage.metrics import structural_similarity

utils_path = os.path.join(sys.path[0], '..')
utlis_path = os.path.abspath(utils_path)
sys.path.append(utils_path)

from data import create_dataset
from models import create_model
from options.test_options import TestOptions


def save_metrics(path, noise_loss, noise_psnr, noise_ssim, out_loss, out_psnr, out_ssim):
    report_path = os.path.join(path, 'report_0725_dncnn_moving700_1.csv')

    if not os.path.isfile(report_path):
        with open(report_path, 'w') as f:
            f.write("noise_loss,noise_psnr,noise_ssim,out_loss,out_psnr,out_ssim\n")

    with open(report_path, 'a') as f:
        f.write("{:.8f},{:.8f},{:.8f},{:.8f},{:.8f},{:.8f}\n".format(
            noise_loss, noise_psnr, noise_ssim, out_loss, out_psnr, out_ssim
        ))


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
    
    x = tensors_dict['x']
    out = tensors_dict['out']
    target = tensors_dict['target']

    out[out>1.0] = 1.0
    out[out<0.0] = 0.0
    
    total_noise_ssim = 0   
    total_out_ssim = 0

    bs, c, h, w = x.shape
    x = x.permute(0, 2, 3, 1)
    out = out.permute(0, 2, 3, 1)
    target = target.permute(0, 2, 3, 1)

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
    
        noise_ssim = structural_similarity(xi, targeti)
        out_ssim = structural_similarity(outi, targeti)
        
        total_noise_ssim += noise_ssim
        total_out_ssim += out_ssim
        
    total_noise_ssim /= bs
    total_out_ssim /= bs
    
    return noise_loss, noise_psnr, total_noise_ssim, out_loss, out_psnr, total_out_ssim

if __name__ == '__main__':
    opt = TestOptions(r'../../../data/multi-frame-image-enhancement').parse()
    opt.n_frames = 1
    dataloader = create_dataset(opt)
    dncnn = create_model(opt)
    #dncnn.eval()

    test_dir = opt.test_results_dir + '-dncnn/0725-2/'
    os.makedirs(test_dir, exist_ok=True)
    dncnn_dir = os.path.join(test_dir, 'dncnn')
    os.makedirs(dncnn_dir, exist_ok=True)

    for i, batch in enumerate(dataloader['test'], 1):
        #predicted_imgs, predicted_idxs = dncnn.predict(batch)
        # out = predicted_imgs
        x = batch['lr']
        target = batch['hr']
        
        tensors_dict = {
                    "x": x,
                    "target": target,
                }
        
        with torch.no_grad():
                    dncnn.set_input(batch)
                    dncnn.test()
                    out = dncnn.out
        
        bs, c, n, h, w = x.shape
        
        for j in range(n):
            print('j:', j)
  
            x_t = x[:, :, j].to(opt.device).detach()
            target_t = target[:, :, j].to(opt.device).detach()
            out_t = out[:, :, j].to(opt.device).detach()
            fns = '{:03d}'.format(j)
            
            tensors_dict = {
                'x': x_t,
                'out': out_t,
                'target': target_t,
            }
            noise_loss, noise_psnr, noise_ssim, out_loss, out_psnr, out_ssim = calc_metrics(tensors_dict)
            save_metrics(dncnn_dir, noise_loss, noise_psnr, noise_ssim, out_loss, out_psnr, out_ssim)

            dncnn_path = os.path.join(dncnn_dir, fns + '.tiff')
            dncnn_out_np = out_t.detach().to('cpu').numpy().transpose(0, 2, 3, 1).squeeze()
            imsave(dncnn_path, dncnn_out_np)