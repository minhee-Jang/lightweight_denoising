import os, sys

import numpy as np
from skimage.transform import rescale
import torch
import matplotlib.pyplot as plt

utils_path = os.path.join(sys.path[0], '..')
utlis_path = os.path.abspath(utils_path)
sys.path.append(utils_path)

from data import create_dataset
from options.train_options import TrainOptions
from options.test_options import TestOptions

from utils.misc import print_numpy


if __name__ == '__main__':
    opt = TrainOptions(r'../../../data/multi-frame-image-enhancement').parse()
    dataloader = create_dataset(opt)

    make_target = (not 'report_ldv' in opt) or ('report_ldv' in opt and opt.report_ldv)

    for i, batch in enumerate(dataloader['train']):
        xn = batch['lr']
        print("xn.shape:", xn.shape)

        if make_target:
            tn = batch['hr']
            print("targetn.shape:", tn.shape)

        vids = batch['videoname']
        print('len(vids):', len(vids))
        filenames = batch['filenames']
        print('type(filenames):', type(filenames))
        print('len(filenames[:])', len(filenames[:]))
        
        bs, c, n, h, w = xn.shape
        for b in range(bs):
            print("batch:", b)
            print('Video {}'.format(vids[b]))

            x_img = xn[b].to('cpu').data.numpy()
            print('x_img')
            print_numpy(x_img, shp=True)
            x_img = x_img.transpose((1, 2, 3, 0))
            x_img_list = []
            for i in range(n):
                print('filenames[{}][{}]: {}'.format(i, b, filenames[i][b]))
                x_img_list.append(x_img[i].squeeze())

            
            if make_target:
                t_img = tn[b].to('cpu').data.numpy()
                print('t_img')
                print_numpy(t_img, shp=True)
                t_img = t_img.transpose((1, 2, 3, 0))

                if opt.scale != 1:
                    tmp_x_img = np.zeros(t_img.shape)
                    for i in range(opt.n_frames):
                        x_img_list[i] = rescale(x_img[i], (opt.scale, opt.scale, 1), anti_aliasing=False)

                concat_x_img = np.concatenate(x_img_list, axis=1)
                print('resized concat_x_img.shape:', concat_x_img.shape)
            
                t_img_list = []
                
                for i in range(n):
                    t_img_list.append(t_img[i].squeeze())

                concat_t_img = np.concatenate(t_img_list, axis=1)
                concat_img = np.concatenate((concat_x_img, concat_t_img), axis=0)
            else:
                concat_x_img = np.concatenate(x_img_list, axis=1)
                concat_img = concat_x_img

            if c == 3:
                print('channel 3')
                concat_img = concat_img * 255
                concat_img = concat_img.astype(np.uint8)
                print('concat_img')
                
                print_numpy(concat_img)
                plt.imshow(concat_img)
                # plt.imshow(concat_x_img)
            else:
                plt.imshow(concat_img, cmap=plt.cm.gray)
            plt.show()
