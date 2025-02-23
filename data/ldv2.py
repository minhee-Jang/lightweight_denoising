import os
import glob

from data.srdata import SRData

class LDV2(SRData):
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--report_ldv', default=False, action='store_true',
            help='load ldv validation data')
        parser.set_defaults(n_channels=3)
        parser.set_defaults(pixel_range=255)
        parser.set_defaults(scale=2)
        return parser
        
    def __init__(self, args, name='ldv2022', is_train=True, is_valid=False):
        self.report_ldv = args.report_ldv
        super(LDV2, self).__init__(
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
            # print('self.report_ldv:', self.report_ldv)
            if not self.report_ldv:
                videos_hr[vid] = sorted(
                    glob.glob(os.path.join(self.dir_hr, vid, '*' + self.ext[1]))
                )

                assert len(videos_hr[vid]) == len(videos_lr[vid])

        video_names = list(videos_lr.keys())

        for vid in video_list:
            filenames[vid] = [self._get_filename(ip) for ip in videos_lr[vid]]

        if self.report_ldv:
            videos = (videos_lr,)
        else:
            videos = (videos_lr, videos_hr)

        return videos, video_names, filenames


    def _get_filename(self, path):
        fn, _ = os.path.splitext(os.path.basename(path))
        video = os.path.normpath(path).split(os.sep)[-2]
        # print('vn:', video)
        filename = video + '-' + fn

        return filename

    def _set_filesystem(self, data_dir):
        self.apath = os.path.join(data_dir, 'ldv2022', 'track2')
        # self.apath = os.path.join(data_dir, 'ldv2022', 'track2_test')

        if self.is_train:
            self.dir_hr = os.path.join(self.apath, 'train_down2_gt')
            self.dir_lr = os.path.join(self.apath, 'train_down2_QP37')
        else:
            self.dir_hr = os.path.join(self.apath, 'validation_track2')
            self.dir_lr = os.path.join(self.apath, 'validation_track2')

        self.ext = ('.png', '.png')