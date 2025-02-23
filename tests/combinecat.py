import os, sys
import torch
from skimage.io import imsave
from skimage.metrics import structural_similarity
utils_path = os.path.join(sys.path[0], '..')
utlis_path = os.path.abspath(utils_path)
sys.path.append(utils_path)
from data import create_dataset
from options.test_options import TestOptions
from deltaMovingmap import EstimationDelta

def save_metrics(path, noise_loss, noise_psnr, noise_ssim, rf_loss, rf_psnr, rf_ssim, mf_loss, mf_psnr, mf_ssim, out_loss, out_psnr, out_ssim):
    report_path = os.path.join(path, 'report_0729_withfsaunet_moving700_1.csv')

    if not os.path.isfile(report_path):
        with open(report_path, 'w') as f:
            f.write("noise_loss,noise_psnr,noise_ssim,rf_loss,rf_psnr,rf_ssim,mf_loss,mf_psnr,mf_ssim,out_loss,out_psnr,out_ssim\n")

    with open(report_path, 'a') as f:
        f.write("{:.8f},{:.8f},{:.8f},{:.8f},{:.8f},{:.8f},{:.8f},{:.8f},{:.8f},{:.8f},{:.8f},{:.8f}\n".format(
            noise_loss, noise_psnr, noise_ssim, rf_loss, rf_psnr, rf_ssim, mf_loss, mf_psnr, mf_ssim, out_loss, out_psnr, out_ssim
        ))


def calc_metrics(tensors_dict):
    x = tensors_dict['x']
    out = tensors_dict['out']
    target = tensors_dict['target']
    rfout = tensors_dict['rf']
    mfout = tensors_dict['mf']

    out[out>1.0] = 1.0
    out[out<0.0] = 0.0

    mse_criterion = torch.nn.MSELoss()
    
    noise_loss = mse_criterion(x, target)
    noise_psnr = 10 * torch.log10(1 / noise_loss)

    rf_loss = mse_criterion(rfout, target)
    rf_psnr = 10 * torch.log10(1 / rf_loss)

    mf_loss = mse_criterion(mfout, target)
    mf_psnr = 10 * torch.log10(1 / mf_loss)
   
    out_loss = mse_criterion(out, target)
    out_psnr = 10 * torch.log10(1 / out_loss)
    
    x = tensors_dict['x']
    out = tensors_dict['out']
    target = tensors_dict['target']
    rfout = tensors_dict['rf']
    mfout = tensors_dict['mf']

    out[out>1.0] = 1.0
    out[out<0.0] = 0.0
    
    total_noise_ssim = 0   
    total_rf_ssim = 0   
    total_mf_ssim = 0
    total_out_ssim = 0

    bs, c, h, w = x.shape
    x = x.permute(0, 2, 3, 1)
    out = out.permute(0, 2, 3, 1)
    target = target.permute(0, 2, 3, 1)
    mfout = mfout.permute(0, 2, 3, 1)
    rfout = rfout.permute(0, 2, 3, 1)

    for i in range(bs):
        xi = x[i]
        outi = out[i]
        targeti = target[i]
        mfouti = mfout[i]
        rfouti = rfout[i]

        xi = xi.squeeze()
        outi = outi.squeeze()
        targeti = targeti.squeeze()
        mfouti = mfouti.squeeze()
        rfouti = rfouti.squeeze()

        xi = xi.detach().to('cpu').numpy()
        outi = outi.detach().to('cpu').numpy()
        targeti = targeti.detach().to('cpu').numpy()
        rfouti = rfouti.detach().to('cpu').numpy()
        mfouti = mfouti.detach().to('cpu').numpy()
    
        noise_ssim = structural_similarity(xi, targeti)
        rf_ssim = structural_similarity(rfouti, targeti)
        mf_ssim = structural_similarity(mfouti, targeti)
        out_ssim = structural_similarity(outi, targeti)
        
        total_noise_ssim += noise_ssim
        total_rf_ssim += rf_ssim   
        total_mf_ssim += mf_ssim
        total_out_ssim += out_ssim
        
    total_noise_ssim /= bs
    total_rf_ssim /= bs   
    total_mf_ssim /= bs
    total_out_ssim /= bs
    
    return noise_loss, noise_psnr, total_noise_ssim, rf_loss, rf_psnr, total_rf_ssim, mf_loss, mf_psnr, total_mf_ssim, out_loss, out_psnr, total_out_ssim


if __name__ == '__main__':
    opt = TestOptions(r'../../../data/multi-frame-image-enhancement').parse()
    opt.n_frames = 5
    dataloader = create_dataset(opt)
    move_thr = opt.move_thr     # 127 (사용 x)
    n_frames = opt.n_frames     # 5
    delta = EstimationDelta()

    # Make directories for output imgs
    # test_dir = opt.test_results_dir + '-psnr/0726_10000mean2_thredynamic_gau5.3_rf0.4new5frames_moving700_4'
    test_dir = opt.test_results_dir + '-psnr/0729_withfsaunet_moving700'
    os.makedirs(test_dir, exist_ok=True)
    mfcnn_dir = os.path.join(test_dir, 'mfcnn')
    os.makedirs(mfcnn_dir, exist_ok=True)
    recursive_dir = os.path.join(test_dir, 'recursive')
    os.makedirs(recursive_dir, exist_ok=True)
    combine_dir = os.path.join(test_dir, 'combine')
    os.makedirs(combine_dir, exist_ok=True)
    deltaout_dir = os.path.join(test_dir, 'deltaout')
    os.makedirs(deltaout_dir, exist_ok=True)
    threout_dir = os.path.join(test_dir, 'threout')
    os.makedirs(threout_dir, exist_ok=True)

    for i, batch in enumerate(dataloader['test']):
        xAll = batch['lr']    #low
        tAll = batch['hr']    #high
        rfAll = batch['rf']    #reculsive output
        mfAll = batch['mf']    #mfcnn output
        bs, c, n, h, w = xAll.shape
        
        # delta로 무빙맵을 구해 DL과 RF combine
        for j in range(n):
            print('j:', j)
            fn_center = '{:03d}'.format(j)
            
            x = xAll[:, :, j:j+1]       # 000~191 (delta구할때000부터필요)
            mf = mfAll[:, :, j:j+1]     # 000~191
            rf = rfAll[:, :, j:j+1]     # 000~191
            
            # Calculate delta moving map and combine DL & RF
            mout, rout, cout, dout, tout = delta(x, rf, mf, move_thr, n_frames)
            # for figures: D, D/mu^2, f, 1-f, out
            
            # PSNR 계산
            x_t = xAll[:, :, j].to(opt.device).detach()
            target_t = tAll[:, :, j].to(opt.device).detach()
            out_t = cout.to(opt.device).detach()
            rout_t = rout.to(opt.device).detach()
            mout_t = mout.to(opt.device).detach()
            tensors_dict = {
                'x': x_t,
                'out': out_t,
                'target': target_t,
                'rf': rout_t,
                'mf': mout_t
            }
            noise_loss, noise_psnr, noise_ssim, rf_loss, rf_psnr, rf_ssim, mf_loss, mf_psnr, mf_ssim, out_loss, out_psnr, out_ssim = calc_metrics(tensors_dict)
            save_metrics(combine_dir, noise_loss, noise_psnr, noise_ssim, rf_loss, rf_psnr, rf_ssim, mf_loss, mf_psnr, mf_ssim, out_loss, out_psnr, out_ssim)
            
            # Save output imgs
            mout = mout.to('cuda')      # MFCNN output
            mout = mout.detach().to('cpu').numpy().transpose(0, 2, 3, 1).squeeze()
            mfcnn_path = os.path.join(mfcnn_dir, fn_center + '.tiff')
            imsave(mfcnn_path, mout)
            rout = rout.to('cuda')      # Recursive Filter output
            rout = rout.detach().to('cpu').numpy().transpose(0, 2, 3, 1).squeeze()
            recursive_path = os.path.join(recursive_dir, fn_center + '.tiff')
            imsave(recursive_path, rout)
            cout = cout.to('cuda')      # Combine output
            cout = cout.detach().to('cpu').numpy().transpose(0, 2, 3, 1).squeeze()
            combine_path = os.path.join(combine_dir, fn_center + '.tiff')
            imsave(combine_path, cout)
            dout = dout.to('cuda')      # Moving map(Delta)
            dout = dout.detach().to('cpu').numpy().transpose(0, 2, 3, 1).squeeze()
            delta_path = os.path.join(deltaout_dir, fn_center + '.tiff')
            imsave(delta_path, dout)
            tout = tout.to('cuda')      # Moving map(Threshold)
            tout = tout.detach().to('cpu').numpy().transpose(0, 2, 3, 1).squeeze()
            thre_path = os.path.join(threout_dir, fn_center + '.tiff')
            imsave(thre_path, tout)