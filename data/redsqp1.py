import os
import glob

from data.srdata import SRData

class REDSQP1(SRData):
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(n_channels=3)
        parser.set_defaults(pixel_range=255)
        parser.set_defaults(scale=1)

        return parser
        
    def __init__(self, args, name='reds', is_train=True, is_valid=False):
        super(REDSQP1, self).__init__(
            args, name=name, is_train=is_train, is_valid=is_valid
        )

    def _scan(self):
        videos_lr = {}
        videos_hr = {}
        filenames = {}

        video_list = sorted(
            vid for vid in os.listdir(self.dir_lr) if os.path.isdir(os.path.join(self.dir_lr, vid))
        )

        for vid in video_list:
            videos_lr[vid] = sorted(
                glob.glob(os.path.join(self.dir_lr, vid, '*' + self.ext[0]))
            )
            videos_hr[vid] = sorted(
                glob.glob(os.path.join(self.dir_hr, vid, '*' + self.ext[1]))
            )

            assert len(videos_hr[vid]) == len(videos_lr[vid])

        video_names = list(videos_lr.keys())

        for vid in video_list:
            filenames[vid] = [self._get_filename(ip) for ip in videos_lr[vid]]
        
        videos = (videos_lr, videos_hr)

        return videos, video_names, filenames

    def _get_filename(self, path):
        fn, _ = os.path.splitext(os.path.basename(path))
        video = os.path.normpath(path).split(os.sep)[-2]
        # print('vn:', video)
        filename = video + '-' + fn

        return filename

    def _set_filesystem(self, data_dir):
        self.apath = os.path.join(data_dir, 'redsqp')

        if self.is_train and not self.is_valid:
            self.dir_hr = os.path.join(self.apath, 'train', 'hr')
            self.dir_lr = os.path.join(self.apath, 'train', 'lr_qp37')
        elif self.is_train and self.is_valid:
            self.dir_hr = os.path.join(self.apath, 'val', 'hr')
            self.dir_lr = os.path.join(self.apath, 'val', 'lr_qp37')
        else:
            self.dir_hr = os.path.join(self.apath, 'test', 'hr')
            self.dir_lr = os.path.join(self.apath, 'test', 'lr_qp37')

        self.ext = ('.png', '.png')