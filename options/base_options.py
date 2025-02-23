"""
This code is based from
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""

import argparse
import os
# from util import util
import torch
import datetime
import json

import models
import data


class BaseOptions():
    """This class defines options used during both training and test time.
    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self, dataroot='../../../data'): 
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        self.dataroot = dataroot
        self.checkpoints_dir = os.path.join(self.dataroot, 'checkpoints')
        self.testresult_dir = os.path.join(self.dataroot, 'test-results')

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--prefix', type=str, default='',
            help='customized suffix: opt.savedir = prefix + opt.savedir')
        parser.add_argument('--suffix', type=str, default='',
            help='customized suffix: opt.savedir = opt.savedir + suffix')

        parser.add_argument('--multi_gpu', dest='multi_gpu', action='store_true',
            help='use all GPUs in machines')
        parser.add_argument('--single_gpu', dest='multi_gpu', action='store_false',
            help='do not enable multiple GPUs')
        parser.set_defaults(multi_gpu=True)
        parser.add_argument('--gpu_ids', type=int, nargs='+', default=[],
            help='gpu ids: e.g. 0  0,1,2, 0,2. use [] for CPU')
        parser.add_argument('--device', type=str, default='cpu',
            help='CPU or GPU')
        parser.add_argument('--n_threads', type=int, default=4,
            help='number of threads for data loader to use, Default: 4')

        # !!
        parser.add_argument('--move_thr', type=int, default=127,
            help='threshold for moving map, Default: 127')
        parser.add_argument('--olala', type=str, default='B',
            help='if you want other frames like mf or rf, choose B, Defalut: A')
         
        # directories to save the results
        parser.add_argument('--data_dir', default=self.dataroot,
            help='path to images')
        parser.add_argument('--checkpoints_dir', type=str, default=self.checkpoints_dir,
            help='checkpoint directory')
        parser.add_argument('--savedir', type=str, default=None,
            help='models are saved here')

        parser.add_argument('--is_train', type=bool, default=True,
            help='phase')
        parser.add_argument('--load_epoch', type=str, default='best',
            help='determine which epoch to load? set to latest to use latest cached model')

        # model
        parser.add_argument('--model', type=str, default='bun_m',
            choices=[
                'unet', # test with 1 frame
                'dncnn2',
                'mfcnn', 'estrnn', 'vsrbasic', 'vsrpp',
                'tdan', # serie
                'vrt', # Geonhui !!
                'vid2vid', # transformer
                'dncnn', # Geonhui !!
                'bnn', # Geonhui !!
                'bnn2',  # Geonhui !!
                'bnn_m', #minhee
                'bun_m', #minhee
                'b2u1',  #minhee            
                'bnn3',
                'EGB7',
                'EGB11',
                'best',
                'best_gelu',
                'best_GPUS',
                'multiSP',
                'multiGBK_recur3',
                'multiGBK_recur4',
                'multiGBK_out_dilated',
                'reuse_weight',
                'best_bnn', 'CBDNet','WganVGG','fin_kba'
            ],
            help='specify a model')

        # dataset parameters
        parser.add_argument('--dataA',  nargs='+', default=['mayo2d'],   #dataset option 변경!!
            help='datasets for training')
        parser.add_argument('--dataB',  nargs='+', default=[],
            help='datasets for training')
        parser.add_argument('--dataVal',  nargs='+', default=[],
            help='datasets for validation')

        parser.add_argument('--dataset_mode', type=str, default='unaligned',
            choices=['unaligned', 'aligned', 'single', 'colorization', 'paired_patch'],
            help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')

        parser.add_argument('--test_dataset',  type=str, default='',
            help='dataset for test')
        parser.add_argument('--batch_size', type=int, default=32,
            help='input batch size')
        parser.add_argument('--n_channels', type=int, default=3,
            help='# of image channels')
        parser.add_argument('--pixel_range', type=float, default=255,
            help='maximum value of RGB or pixel value')
        parser.add_argument('--use_img', default=False, action='store_true',
            help='Use image (not patch) when training')    #patch 쓸지 여부

        parser.add_argument('--patch_size', type=int, default=120,
            help='size of patch')

        parser.add_argument('--scale', type=int, default='1',
            help='super resolution scale')
        parser.add_argument('--n_frames', type=int, default=11,      #프레임수변경!!
            help='number of image inputs to the network')

        parser.add_argument('--in_mem', action='store_true',
            help='load whole data into memory, default: False')
        # parser.add_argument('--from_disk', dest='in_mem', action='store_false',
        #     help='load image from disk')
        # parser.set_defaults(in_mem=True)

            
        parser.add_argument('--resume', action='store_true',
            help='continue training: load the latest model')
            
        # additional parameters
        parser.add_argument('--verbose', action='store_true',
             help='if specified, print more debugging information')


        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # Set the basic options
        base_opt, _ = parser.parse_known_args()
        if self.is_train and base_opt.resume:
            savedir = self.select_checkpoint_dir(base_opt)
            base_opt = self.load_options(base_opt, savedir)
        elif not self.is_train and not base_opt.url:
            savedir = self.select_checkpoint_dir(base_opt)
            base_opt = self.load_options(base_opt, savedir)

        # print('base_opt:', base_opt)

        # modify model-related parser options
        model_name = base_opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.is_train)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # print('model_opt:', opt)

        # modify dataset-related parser options
        for dataset_name in base_opt.dataA:
            # dataset_name = dataset
            dataset_option_setter = data.get_option_setter(dataset_name)
            parser = dataset_option_setter(parser, self.is_train)
        
        for dataset_name in base_opt.dataB:
            # dataset_name = dataset
            dataset_option_setter = data.get_option_setter(dataset_name)
            parser = dataset_option_setter(parser, self.is_train)
        
        if opt.test_dataset != '':
            dataset_option_setter = data.get_option_setter(opt.test_dataset)
            parser = dataset_option_setter(parser, self.is_train)
            
        # save and return the parser
        self.parser = parser
        opt = parser.parse_args()

        # print('parsed opt:', opt)
        
        if not self.is_train or opt.resume:
            # print('load_options')
            opt = self.load_options(opt, savedir)
            opt.savedir = savedir

        # print('loaded opt:', opt)

        return opt

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.is_train = self.is_train   # train or test

        # Get savedir from model or select checkpoint dir
        if self.is_train and not opt.resume:
            # Set checkpoint directory to save
            set_savedir = models.get_savedir_setter(opt.model)
            opt.savedir = set_savedir(opt)
            self.save_options(opt)
            opt.log_file = os.path.join(opt.savedir, opt.log_file)
        elif  not self.is_train and opt.url:
            # when url is used, we need to specify all model parameters rather than loading from config.txt
            opt.savedir = os.path.join(opt.checkpoints_dir, opt.model)
        # else:
        #     opt = self.load_options(opt, savedir)

        # print("savedir:", os.path.abspath(opt.savedir))
        if not self.is_train:
            opt.test_results_dir = os.path.join(self.testresult_dir, os.path.basename(opt.savedir))
            os.makedirs(opt.test_results_dir, exist_ok=True)
        # set gpu ids

        self.set_gpus(opt)
        self.print_options(opt)

        self.opt = opt
        return self.opt
    
    def print_options(self, opt):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / train_opt.txt
        """
        

        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: {}]'.format(str(default))
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        # expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        
        if self.is_train:
            file_name = os.path.join(opt.savedir, 'train_opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')

    """
    My methods
    """
    def save_options(self, opt):
        os.makedirs(opt.savedir, exist_ok=True)
        config_file = os.path.join(opt.savedir, "config.txt")
        with open(config_file, 'w') as f:
            json.dump(opt.__dict__, f, indent=2)

    def load_options(self, opt, savedir):
        # print(opt.savedir)
        config_file = os.path.join(savedir, "config.txt")
        with open(config_file, 'r') as f:
            # opt.__dict__ = json.load(f)
            saved_options = json.load(f)

        """
        Set parameters to be controlled
        """
        resume = opt.resume
        test_dataset = opt.test_dataset
        data_dir = opt.data_dir
        for key in saved_options:
            if key in opt:
                # print("saved_options[{}]: {}".format(key, saved_options[key]))
                # print("opt[{}]: {}".format(key, opt.__dict__[key]))
                opt.__dict__[key] = saved_options[key]

        opt.resume = resume
        opt.test_dataset = test_dataset
        opt.data_dir = data_dir
        
        return opt

    def select_checkpoint_dir(self, opt):
        print("checkpoint_dir:", os.path.abspath(opt.checkpoints_dir))
        dirs = sorted(os.listdir(opt.checkpoints_dir))

        for i, d in enumerate(dirs, 0):
            print("({}) {}".format(i, d))
        d_idx = input("Select directory that you want to load: ")

        path_opt = dirs[int(d_idx)]
        savedir = os.path.abspath(os.path.join(self.checkpoints_dir, path_opt))
        print("savedir: {}".format(savedir))

        return savedir

    def set_gpus(self, opt):
        n_gpu = torch.cuda.device_count()
        if opt.multi_gpu and len(opt.gpu_ids) == 0 and torch.cuda.is_available():
            opt.gpu_ids = list(range(torch.cuda.device_count()))
        elif len(opt.gpu_ids) > 0 and torch.cuda.is_available():
            gpu_ids = opt.gpu_ids
            opt.gpu_ids = []
            for id in gpu_ids:
                if id >= 0 and id < n_gpu:
                    opt.gpu_ids.append(id)
            opt.gpu_ids = sorted(opt.gpu_ids)
        else:
            opt.gpu_ids = []
        
        if len(opt.gpu_ids) > 0:
            print("Enabling GPUs", opt.gpu_ids)
            if len(opt.gpu_ids) > 1:
                opt.multi_gpu = True
            else:
                opt.multi_gpu = False
            opt.device = "cuda:{}".format(opt.gpu_ids[0])
        else:
            print("No GPUs use")
            opt.device = "cpu"