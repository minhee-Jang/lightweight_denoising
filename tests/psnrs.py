import os, sys
import torch
from skimage.metrics import structural_similarity
utils_path = os.path.join(sys.path[0], '..')
utlis_path = os.path.abspath(utils_path)
sys.path.append(utils_path)
from data import create_dataset
from options.test_options import TestOptions

day = '0803'
trytime = '-moving_unetrft3-01'

def save_metrics(path, t1_loss, t1_psnr, t1_ssim, t2_loss, t2_psnr, t2_ssim, t3_loss, t3_psnr, t3_ssim):
    report_name = 'report_' + day + trytime + '.csv'
    report_path = os.path.join(path, report_name)

    if not os.path.isfile(report_path):
        with open(report_path, 'w') as f:
            f.write("t1_psnr,t1_ssim,t2_psnr,t2_ssim,t3_psnr,t3_ssim\n")

    with open(report_path, 'a') as f:
        f.write("{:.8f},{:.8f},{:.8f},{:.8f},{:.8f},{:.8f}\n".format(
            t1_psnr, t1_ssim, t2_psnr, t2_ssim, t3_psnr, t3_ssim
        ))


def calc_metrics(tensors_dict):
    target = tensors_dict['target']
    t1 = tensors_dict['t1']
    t2 = tensors_dict['t2']
    t3 = tensors_dict['t3']

    mse_criterion = torch.nn.MSELoss()
    
    t1_loss = mse_criterion(t1, target)
    t1_psnr = 10 * torch.log10(1 / t1_loss)

    t2_loss = mse_criterion (t2, target)
    t2_psnr = 10 * torch.log10(1 / t2_loss)

    t3_loss = mse_criterion(t3, target)
    t3_psnr = 10 * torch.log10(1 / t3_loss)

    total_t1_ssim = 0   
    total_t2_ssim = 0   
    total_t3_ssim = 0

    bs, _, _, _ = target.shape
    target = target.permute(0, 2, 3, 1)
    t1 = t1.permute(0, 2, 3, 1)
    t2 =  t2.permute(0, 2, 3, 1)
    t3 = t3.permute(0, 2, 3, 1)

    for i in range(bs):
        targeti = target[i].squeeze()
        t1i = t1[i].squeeze()
        t2i = t2[i].squeeze()
        t3i = t3[i].squeeze()
        
        targeti = targeti.detach().to('cpu').numpy()
        t1i = t1i.detach().to('cpu').numpy()
        t2i = t2i.detach().to('cpu').numpy()
        t3i = t3i.detach().to('cpu').numpy()
    
        t1_ssim = structural_similarity(t1i, targeti)
        t2_ssim = structural_similarity (t2i, targeti)
        t3_ssim = structural_similarity(t3i, targeti)
        
        total_t1_ssim += t1_ssim
        total_t2_ssim += t2_ssim
        total_t3_ssim += t3_ssim
        
    total_t1_ssim /= bs
    total_t2_ssim /= bs   
    total_t3_ssim /= bs
    
    return t1_loss, t1_psnr, total_t1_ssim, t2_loss, t2_psnr, total_t2_ssim, t3_loss, t3_psnr, total_t3_ssim


if __name__ == '__main__':
    opt = TestOptions(r'../../../data/multi-frame-image-enhancement').parse()
    dataloader = create_dataset(opt)

    # Make directories for output imgs
    test_name = '-psnrs/' + day + trytime
    test_dir = opt.test_results_dir + test_name
    os.makedirs(test_dir, exist_ok=True)

    for i, batch in enumerate(dataloader['test']):
        tAll = batch['hr']   # target 넣기
        t1All = batch['lr']  # t1 넣기
        t2All = batch['mf']  # t2 넣기
        t3All = batch['rf']  # t3 넣기
        bs, c, n, h, w = tAll.shape

        for j in range(n):
            print('j:', j)
            target_t = tAll[:, :, j].to(opt.device).detach()
            t1t = t1All[:, :, j].to(opt.device).detach()
            t2t = t2All[:, :, j].to(opt.device).detach()
            t3t = t3All[:, :, j].to(opt.device).detach()
            
            tensors_dict = {
                'target': target_t,
                't1': t1t,
                't2': t2t,
                't3': t3t
            }
            t1_loss, t1_psnr, t1_ssim, t2_loss, t2_psnr, t2_ssim, t3_loss, t3_psnr, t3_ssim = calc_metrics(tensors_dict)
            save_metrics(test_dir, t1_loss, t1_psnr, t1_ssim, t2_loss, t2_psnr, t2_ssim, t3_loss, t3_psnr, t3_ssim)