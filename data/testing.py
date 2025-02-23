import os
import glob

from data.srdata import SRData
from . import common

class Testing(SRData):
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(n_channels=1)
        parser.set_defaults(pixel_range=1.0)
        parser.set_defaults(scale=1)
        return parser
        
    def __init__(self, args, name='testing', is_train=True, is_valid=False):
        super(Testing, self).__init__(
            args, name=name, is_train=is_train, is_valid=is_valid
        )

    def _scan(self):
        videos_lr = {}
        videos_hr = {}
        filenames = {}
        videos_mf = {}
        videos_rf = {}

        video_list = sorted(
            vid for vid in os.listdir(self.dir_lr) if os.path.isdir(os.path.join(self.dir_lr, vid))
        )

        for vid in video_list:
            videos_lr[vid] = sorted(
                glob.glob(os.path.join(self.dir_lr, vid, '*' + self.ext[1]))
            )
            videos_hr[vid] = sorted(
                glob.glob(os.path.join(self.dir_hr, vid, '*' + self.ext[0]))
            )
            videos_mf[vid] = sorted(
                glob.glob(os.path.join(self.dir_mf, vid, '*' + self.ext[1]))
            )
            videos_rf[vid] = sorted(
                glob.glob(os.path.join(self.dir_rf, vid, '*' + self.ext[1]))
            )

            #!!
            #assert len(videos_lr[vid]) == len(videos_hr[vid])

            print(len(videos_lr[vid]))
            print(len(videos_hr[vid]))

        video_names = list(videos_lr.keys())
        for vid in video_list:
            filenames[vid] = [self._get_filename(ip) for ip in videos_lr[vid]]

        videos = (videos_lr, videos_hr, videos_mf, videos_rf)

        return videos, video_names, filenames

    def _get_filename(self, path):
        filename, _ = os.path.splitext(os.path.basename(path))

        return filename

    def _set_filesystem(self, data_dir):
        # self.apath = os.path.join(data_dir, 'testing')

        # self.dir_hr = os.path.join(self.apath, self.mode, 'high')
        # self.dir_lr = os.path.join(self.apath, self.mode, 't1')
        # self.dir_mf = os.path.join(self.apath, self.mode, 't2')
        # self.dir_rf = os.path.join(self.apath, self.mode, 't3')

        # self.ext = ('.tiff', '.tiff')
        self.apath = os.path.join(data_dir, 'test')

        self.dir_lr = os.path.join(self.apath,  't1')
        self.dir_hr = os.path.join(self.apath,  'high')
        self.dir_mf = os.path.join(self.apath,  't2')
        self.dir_rf = os.path.join(self.apath,  't3')
    


        self.ext = ('.png', '.png')
    
    def __getitem__(self, idx):
        if not self.in_mem:
            frames, videoname, filenames, idx = self._load_file(idx)
        else:
            frames, videoname, filenames, idx = self._load_from_mem(idx)

        if self.is_train: frames = self.get_patch(*frames)

        print(len(frames))
        frames = common.set_channel(*frames, n_channels=self.n_channels)
        frames = common.np2Tensor(*frames, pixel_range=self.args.pixel_range)
        frames = common.concat_tensor(*frames)
        print(frames[0])

        # !!
        data_dict = {
        't1': frames[0],
        'hr' : frames[1],
        't2' : frames[2],
        't3' : frames[3],
        'videoname': videoname,
        'filenames': filenames
        }
            
        return data_dict