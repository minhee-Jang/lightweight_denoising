import os
import glob

from data.srdata import SRData

class Genoray(SRData):
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(n_channels=1)
        parser.set_defaults(pixel_range=1.0)
        parser.set_defaults(scale=1)
        return parser
        
    def __init__(self, args, name='genoray', is_train=True, is_valid=False):
        super(Genoray, self).__init__(
            args, name=name, is_train=is_train, is_valid=is_valid
        )

    def _scan(self):
        videos_lr = {}
        videos_hr = {}
        filenames = {}
        # videos_mf = {}
        # videos_rf = {}

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
            # videos_mf[vid] = sorted(
            #     glob.glob(os.path.join(self.dir_mf, vid, '*' + self.ext[1]))
            # )
            # videos_rf[vid] = sorted(
            #     glob.glob(os.path.join(self.dir_rf, vid, '*' + self.ext[1]))
            # )

            #assert len(videos_lr[vid]) == len(videos_hr[vid])

        video_names = list(videos_lr.keys())
        for vid in video_list:
            filenames[vid] = [self._get_filename(ip) for ip in videos_lr[vid]]

        videos = (videos_lr, videos_hr)

        return videos, video_names, filenames

    def _get_filename(self, path):
        filename, _ = os.path.splitext(os.path.basename(path))

        return filename

    def _set_filesystem(self, data_dir):
        self.apath = os.path.join(data_dir, 'genoray')

        self.dir_lr = os.path.join(self.apath, self.mode, 'Low')
        self.dir_hr = os.path.join(self.apath, self.mode, 'Low_avg')
        # self.dir_mf = os.path.join(self.apath, self.mode, 'mf')
        # self.dir_rf = os.path.join(self.apath, self.mode, 'rf')
        
        self.ext = ('.tiff', '.tiff')