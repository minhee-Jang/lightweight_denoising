import os
import glob
import pickle
import numpy as np
import torch

from abc import ABC, abstractmethod
# from skimage.io import imread
from imageio import imread
import random

from . import common
from .base_dataset import BaseDataset

class SRData(BaseDataset):
    def __init__(self, args, name='', is_train=True, is_valid=False):
        self.args = args
        self.dataset = name
        self.in_mem = args.in_mem
        self.n_frames = args.n_frames
        self.n_channels = args.n_channels
        self.scale = args.scale
        
        #!!!!!!!!!!
        self.olala = args.olala

        self.is_train = is_train
        self.is_valid = is_valid
        # print('is_valid:', is_valid)
        if is_train and not is_valid:
            # print('training')
            self.mode = 'train'
        elif is_train and is_valid:
            self.mode = 'validation'
            # print('validation')
        else:
            
            self.mode = 'test'
        # self.mode = 'train' if is_train else 'test'
        
        self._set_filesystem(args.data_dir)
        print("----------------- {} {} dataset -------------------".format(name, self.mode))
        print("Set file system for {} dataset {}".format(self.mode, self.dataset))
        print("apath:", os.path.abspath(self.apath))
        print("dir_lr:", os.path.abspath(self.dir_lr))
        print("dir_hr:", os.path.abspath(self.dir_hr))
        # print("dir_rf:", os.path.abspath(self.dir_rf))
        # print("dir_mf:", os.path.abspath(self.dir_mf))
        # print("dir_maskE:", os.path.abspath(self.dir_maskE))
        # print("dir_maskNE:", os.path.abspath(self.dir_maskNE))



        self.videos, self.videonames, self.filenames = self._scan()
        # self.videonames = self.videos_lr.keys()
        print('len(self.videos):', len(self.videos))
        print('self.videonames:', self.videonames)

        if self.in_mem: self._load2mem()
        
        if self.is_train and not self.is_valid:
            n_patches = args.batch_size * args.test_every
            n_videos = len(args.dataA) * len(self.videonames)
            if n_videos == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_videos, 1)

    @abstractmethod
    def _scan(self):
        pass

    def _set_filesystem(self, data_dir):
        self.apath = os.path.join(data_dir, self.dataset, self.mode)
        self.dir_hr = os.path.join(self.apath, 'hr')
        self.dir_lr = os.path.join(self.apath, 'lr')
        # self.dir_maskE1 = os.path.join(self.apath, 'Edge1')
        # self.dir_maskE2 = os.path.join(self.apath, 'Edge2')
        # self.dir_maskNE = os.path.join(self.apath, 'NEdge')
        # self.dir_maskE = os.path.join(self.apath, 'Edge')
        # self.dir_maskNE = os.path.join(self.apath, 'NEdge')
        # self.dir_mf = os.path.join(self.apath, 'mf') #!!
        # self.dir_rf = os.path.join(self.apath, 'rf') #!!
       
    def __getitem__(self, idx):

        if not self.in_mem:
            frames, videoname, filenames, idx = self._load_file(idx)
      
        else:
            frames, videoname, filenames, idx = self._load_from_mem(idx)
   
        if self.is_train and not self.args.use_img: frames = self.get_patch(*frames)
        #print('1', frames[3])
        frames = common.set_channel(*frames, n_channels=self.n_channels)
        #print('2',frames[3])
        frames = common.np2Tensor(*frames, pixel_range=self.args.pixel_range)
        #print('3',frames[3])
        frames = common.concat_tensor(*frames)
       

        if self.olala == 'A':
            data_dict = {
                'lr': frames[0],
                'hr' : frames[1],
                'videoname': videoname,
                'filenames': filenames, 
                #'idx': idx
            }

        else:   
            data_dict = {
                'lr': frames[0],
                'hr' : frames[1],
                'mask_E1' : frames[2],
                'mask_E2' : frames[3],
                'mask_NE' : frames[4],
                'videoname': videoname,
                'filenames': filenames
            }
        # #if len(frames) == 2:
            #data_dict['hr'] = frames[1]
            
            
        return data_dict

            
    def __len__(self):
        if self.is_train and not self.is_valid:
            return len(self.videonames) * self.repeat
        else:
            return len(self.videonames)

    def _get_index(self, idx):

        if self.is_train and not self.is_valid:
            return idx % len(self.videonames)
        else:
            return idx

    def get_sequences(self, vn):
        #print(vn)
        total_n = len(self.videos[0][vn])
        # print("======")
        # print(len(self.videos))
        # print('total_n:', total_n)
        if self.is_train and not self.is_valid:
            nf = self.n_frames
            si = random.randint(0, total_n - nf)

            frames = [video[vn][si:si+nf] for video in self.videos]
            fns = self.filenames[vn][si:si+nf]
        elif self.is_train and self.is_valid:
            # nf = self.n_frames
            # si = (total_n - nf - 1) // 2

            # frames = [video[vn][si:si+nf] for video in self.videos]
            # fns = self.filenames[vn][si:si+nf]
            fns = self.filenames[vn]
        else:
            frames = [video[vn] for video in self.videos]
            fns = self.filenames[vn]

        return frames, fns
        
    def _load_file(self, idx):
        idx = self._get_index(idx)
        vn = self.videonames[idx]

        frames, filenames = self.get_sequences(vn)
   
      
        frames = [[imread(f) for f in frame] for frame in frames]

        return frames, vn, filenames, idx

    def _load_from_mem(self, idx):
        idx = self._get_index(idx)
        vn = self.videonames[idx]
        frames, filenames = self.get_sequences(vn)

        return frames, vn, filenames, idx

    def _load2mem(self):
        def _intomem(vids):
            for vn, ip_list in vids.items():
                print('loading video {} into memory'.format(vn))
                vids[vn] = [imread(ip) for ip in ip_list]

            return vids

        self.videos = [_intomem(vids) for vids in self.videos]


    def get_patch(self, *frames):
        frames = common.get_patch(
            *frames,
            patch_size=self.args.patch_size,
            scale=self.scale,
            center_crop=self.is_valid
        )
        if self.args.augment: frames = common.augment(*frames)
        
        return frames