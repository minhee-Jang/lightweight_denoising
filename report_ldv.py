import os
import time
import torch
from scipy import stats

from options.test_options import TestOptions
from data import create_dataset
from models import create_model

from skimage.io import imsave
import numpy as np

import zipfile

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    opt.is_train = False
    opt.load_epoch = 'best'
    opt.in_mem = False
    opt.multi_gpu = False
    # hard-code some parameters for test
    opt.n_threads = 0   # test code only supports num_threads = 1
    opt.report_ldv = True

    dataloader = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    model.eval()
    report_dir = os.path.join(opt.test_results_dir, 'report')

    print("*** Making report file with {}***".format(opt.test_dataset))

    total_n = 0
    start_time = time.time()

    videonames = []
    for batch in dataloader['test']:
        # in test mode, one batch loads whole sequences of the video
        vn = batch['videoname'][0]
        videonames.append(vn)
        print(f'predicting video {vn}')

        vid_dir = os.path.join(report_dir, vn)
        os.makedirs(vid_dir, exist_ok=True)
        predicted_imgs, predicted_idxs = model.predict(batch)
        
        x = batch['lr']
        out = predicted_imgs
        filenames = batch['filenames']
        for i in range(x.shape[2]):
            x_t = x[:, :, i].to(opt.device).detach()
            out_t = out[:, :, i].to(opt.device).detach()
            filename = filenames[i][0]
            vn, fidx = filename.split('-')

            # print('fn:', fn)
            if int(fidx[1:]) % 10 != 0:
                continue
            fn = f'{fidx}.png'
            fp = os.path.join(vid_dir, fn)

            img = out_t.to('cpu').detach().numpy()
            # print('img.shape:', img.shape)
            img = img.squeeze().transpose(1, 2, 0)
            img[img>1.0] = 1.0
            img[img<0.0] = 0
            img = img * 255
            img = np.rint(img)
            img = img.astype(np.uint8)
            print('writing {}'.format(os.path.abspath(fp)))
            imsave(fp, img)

        total_n += len(predicted_imgs)
    end_time = time.time()
    avg_time = (end_time - start_time) / total_n

    msg = 'runtime per image [s] : {:.2f}\nCPU[1] / GPU[0] : 0\n'.format(avg_time)
    msg += 'Extra Data [1] / No Extra Data [0] : 0\n'
    msg += 'Other description : Multi-frame residual dense net.\n'

    print(msg)

    readme_path = os.path.join(report_dir, 'readme.txt')
    with open(readme_path, 'w') as f:
        f.write(msg)

    os.chdir(report_dir)
    sn = 'submission.zip'
    if os.path.isfile(sn):
        os.remove(sn)

    zipfiles = os.listdir('./')
    zip_path = os.path.join(sn)
    output_zip = zipfile.ZipFile(zip_path, 'w')

    for vn in videonames:
        images = os.listdir(vn)
        for img in images:
            ip = os.path.join(vn, img)
            print(f'zipping {ip}')
            output_zip.write(ip)

    output_zip.write('readme.txt')
    output_zip.close()