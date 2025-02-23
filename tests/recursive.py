import os, sys
utils_path = os.path.join(sys.path[0], '..')
utlis_path = os.path.abspath(utils_path)
sys.path.append(utils_path)
from data import create_dataset
from options.test_options import TestOptions
from skimage.io import imsave
from skimage.metrics import structural_similarity
import torch
import torch.nn as nn

day = '0803_'
trytime = '_moving_5framesrf_01'
thre = 0.3
nframe = 5

class RecursiveFilter(nn.Module):
    def __init__(self, w=thre):
        super().__init__()
        self.w  = w

    def forward(self, x):
        reout = x.clone().detach()
        _, _, n, _, _ = x.shape

        reout[:, :, 0] = x[:, :, 0]

        for i in range(1, n):
            reout[:, :, i] = self.w * x[:, :, i] + (1 - self.w) * reout[:, :, i-1]

        return reout
    
def save_metrics(path, rf_loss, rf_psnr, rf_ssim):
    report_name = 'report_' + day + str(thre) + trytime + '.csv'
    report_path = os.path.join(path, report_name)

    if not os.path.isfile(report_path):
        with open(report_path, 'w') as f:
            f.write("rf_loss,rf_psnr,rf_ssim\n")

    with open(report_path, 'a') as f:
        f.write("{:.8f},{:.8f},{:.8f}\n".format(
            rf_loss, rf_psnr, rf_ssim
        ))

def calc_metrics(tensors_dict):
    target = tensors_dict['target']
    rfout = tensors_dict['rf']

    mse_criterion = torch.nn.MSELoss()
    
    rf_loss = mse_criterion(rfout, target)
    rf_psnr = 10 * torch.log10(1 / rf_loss)
    
    total_rf_ssim = 0

    bs, c, h, w = rfout.shape
    
    target = target.permute(0, 2, 3, 1)
    rfout = rfout.permute(0, 2, 3, 1)

    for i in range(bs):
        targeti = target[i]
        rfouti = rfout[i]

        targeti = targeti.squeeze()
        rfouti = rfouti.squeeze()

        targeti = targeti.detach().to('cpu').numpy()
        rfouti = rfouti.detach().to('cpu').numpy()
    
        rf_ssim = structural_similarity(rfouti, targeti)
        
        total_rf_ssim += rf_ssim
        
    total_rf_ssim /= bs   

    return rf_loss, rf_psnr, total_rf_ssim

if __name__ == '__main__':
    opt = TestOptions(r'../../../data/multi-frame-image-enhancement').parse()
    opt.n_frames = nframe    # 프레임 수 변경 가능
    dataloader = create_dataset(opt)
    n_frames = opt.n_frames

    recursive_filter = RecursiveFilter()

    test_name = '-rf/' + day + str(thre) + trytime
    test_dir = opt.test_results_dir + test_name
    os.makedirs(test_dir, exist_ok=True)

    for i, batch in enumerate(dataloader['test']):
        xAll = batch['lr']    #low
        tAll = batch['hr']    #high

        bs, c, n, h, w = xAll.shape

        rfAll = recursive_filter(xAll)
        rfAll = rfAll.to('cuda')

        _, _, r_frames, _, _ = rfAll.shape

        # 000~009까지는 10 frames 다 쌓이지 않았으므로 기존 방식으로 쌓음
        for j in range(n_frames):
            print('j: ', j)
            rfIdx = '{:03d}'.format(j)
            
            rf = rfAll[:, :, j]
            rout = rf.detach().to('cpu').numpy().transpose(0, 2, 3, 1).squeeze()
            recursive_path = os.path.join(test_dir, rfIdx + '.tiff')
            imsave(recursive_path, rout)
            
            # PSNR 계산
            target_t = tAll[:, :, j].to(opt.device).detach()
            out_t = rf.to(opt.device).detach()
            tensors_dict = {
                'target': target_t,
                'rf' : out_t,
            }
            rf_loss, rf_psnr, rf_ssim = calc_metrics(tensors_dict)
            save_metrics(test_dir, rf_loss, rf_psnr, rf_ssim)

        # 010~191까지는 10 frames씩 가져와서 쌓음
        for k in range(n_frames, r_frames):
            print('k:', k)
            rfIdx = '{:03d}'.format(k)
            
            x = xAll[:, :, k-n_frames:k+1]   # 000~191 (rf쌓을때000부터필요)
            rfAll2 = recursive_filter(x)
            
            rf = rfAll2[:, :, n_frames]
            rout = rf.detach().to('cpu').numpy().transpose(0, 2, 3, 1).squeeze()
            recursive_path = os.path.join(test_dir, rfIdx + '.tiff')
            imsave(recursive_path, rout)
            
            # PSNR 계산
            target_t = tAll[:, :, k].to(opt.device).detach()
            out_t = rf.to(opt.device).detach()
            tensors_dict = {
                'target': target_t,
                'rf' : out_t,
            }
            rf_loss, rf_psnr, rf_ssim = calc_metrics(tensors_dict)
            save_metrics(test_dir, rf_loss, rf_psnr, rf_ssim)