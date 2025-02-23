import os
import glob

from data.srdata import SRData

class Pelvis(SRData):
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(n_channels=1)
        parser.set_defaults(pixel_range=1.0)
        parser.set_defaults(scale=1)
        return parser
        
    def __init__(self, args, name='Pelvis', is_train=True, is_valid=False):
        super(Pelvis, self).__init__(
            args, name=name, is_train=is_train, is_valid=is_valid
        )

    def _scan(self):
        videos_lr = {}
        videos_hr = {}
        # videos_maskE1 = {}
        # videos_maskE2 = {}
        # videos_maskNE = {}

        filenames = {}
        
        
        video_list = sorted(
            vid for vid in os.listdir(self.dir_lr) if os.path.isdir(os.path.join(self.dir_lr, vid))
        )

        print("video_list:", len(video_list))   #환자 수 

        for vid in video_list:
            videos_lr[vid] = sorted(
                glob.glob(os.path.join(self.dir_lr, vid, '*' + self.ext))
            )
            videos_hr[vid] = sorted(
                glob.glob(os.path.join(self.dir_hr, vid, '*' + self.ext))
            )
            # videos_maskE1[vid] = sorted(
            #     glob.glob(os.path.join(self.dir_maskE1, vid, '*' + self.ext))
            # )
            # videos_maskE2[vid] = sorted(
            #     glob.glob(os.path.join(self.dir_maskE2, vid, '*' + self.ext))
            # )
            # videos_maskNE[vid] = sorted(
            #     glob.glob(os.path.join(self.dir_maskNE, vid, '*' + self.ext))
            # )
            # videos_maskNE[vid] = sorted(
            #     glob.glob(os.path.join(self.dir_maskNE, vid, '*' + self.ext))
            # )

            assert len(videos_lr[vid]) == len(videos_hr[vid])

        # print(len(videos_maskNE)) #7
   

        video_names = list(videos_lr.keys())
        for vid in video_list:
            filenames[vid] = [self._get_filename(ip) for ip in videos_lr[vid]]

        #videos = (videos_lr, videos_hr, videos_maskE1,videos_maskE2, videos_maskNE)  #dictionary 갯수니까 2개가 맞음 
        videos = (videos_lr, videos_hr) 

        return videos, video_names, filenames

    def _get_filename(self, path):
        filename, _ = os.path.splitext(os.path.basename(path))

        return filename

    def _set_filesystem(self, data_dir):
        self.apath = os.path.join(data_dir, 'ct_pelvis')

        self.dir_lr = os.path.join(self.apath, self.mode, 'quarter_1mm')
        self.dir_hr = os.path.join(self.apath, self.mode, 'full_1mm')
        # self.dir_maskE1 = os.path.join(self.apath, self.mode, 'gauss_mask', 'Edge1')
        # self.dir_maskE2 = os.path.join(self.apath, self.mode,'gauss_mask',  'Edge2')
        # self.dir_maskNE = os.path.join(self.apath, self.mode,'gauss_mask',  'NEdge')


        self.ext = ('.tiff')

 
    # def __getitem__(self, idx):

    #     if not self.in_mem:
    #         frames, videoname, filenames, idx = self._load_file(idx)
      
    #     else:
    #         frames, videoname, filenames, idx = self._load_from_mem(idx)
   
    #     if self.is_train and not self.args.use_img: frames = self.get_patch(*frames)
    #     #print('1', frames[3])
    #     frames = common.set_channel(*frames, n_channels=self.n_channels)
    #     #print('2',frames[3])
    #     frames = common.np2Tensor(*frames, pixel_range=self.args.pixel_range)
    #     #print('3',frames[3])
    #     frames = common.concat_tensor(*frames)
       

    #     if self.olala == 'A':
    #         data_dict = {
    #             'lr': frames[0],
    #             'hr' : frames[1],
    #             'videoname': videoname,
    #             'filenames': filenames, 
    #             #'idx': idx
    #         }

    #     else:   ########## for combine (SPIE2022) ##########
    #         # data_dict = {
    #         #     'lr': frames[0],
    #         #     'hr' : frames[1],
    #         #     'mf' : frames[2],
    #         #     'rf' : frames[3],
    #         #     'videoname': videoname,
    #         #     'filenames': filenames
    #         # }
    #         data_dict = {
    #             'lr': frames[0],
    #             'hr' : frames[1],
    #             # 'mask_E1' : frames[2],
    #             # 'mask_E2' : frames[3],
    #             # 'mask_NE' : frames[4],
    #             # 'mask_E' : frames[2],
    #             # 'mask_NE' : frames[3],
    #             'videoname': videoname,
    #             'filenames': filenames
    #         }
    #     # #if len(frames) == 2:
    #         #data_dict['hr'] = frames[1]
            
            
    #     return data_dict

#     def get_patch(self, *frames):
#         frames = _get_patch(
#             *frames,
#             patch_size=self.args.patch_size,
#             scale=self.scale,
#             center_crop=self.is_valid
#         )
#         if self.args.augment: frames = common.augment(*frames)
        
#         return frames

# def _get_patch(*args, patch_size=160, scale=2, center_crop=False):
#     print('args[0][0].shape:', args[0][0].shape)
#     ih, iw = args[0][0].shape[:2]

#     tp = patch_size
#     ip = tp // scale

#     if center_crop:
#         ix = iw // 2 - ip // 2
#         iy = ih // 2 - ip // 2
    
#     else:
#         ix = random.randrange(0, iw - ip + 1)
#         iy = random.randrange(0, ih - ip + 1)

#     tx, ty = scale * ix, scale * iy

#     args = [
#         [a[iy:iy + ip, ix:ix + ip] for a in args[0]],
#         [a[ty:ty + tp, tx:tx + tp] for a in args[1]],
#         [a[ty:ty + tp, tx:tx + tp] for a in args[2]],
#         [a[ty:ty + tp, tx:tx + tp] for a in args[3]],
#     ]

#     return args