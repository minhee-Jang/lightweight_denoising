import os
import glob

from data.srdata import SRData

class VimeoQP3(SRData):
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(n_channels=3)
        parser.set_defaults(pixel_range=255)
        parser.set_defaults(scale=4)

        return parser
        
    def __init__(self, args, name='vimeo', is_train=True, is_valid=False):
        super(VimeoQP3, self).__init__(
            args, name=name, is_train=is_train, is_valid=is_valid
        )

    def _scan(self):
        videos_lr = {}
        videos_hr = {}
        filenames = {}

        with open(os.path.join(self.apath, self.vid_list_f), 'r') as f:
            video_list = f.readlines()
            video_list = [l[:-1] for l in video_list]

        for vid in video_list:
            # print('vid:', vid)
            s, v = vid.split('/')
            # lrp = os.path.join(self.dir_lr, s, v, '*' + self.ext[0])
            # hrp = os.path.join(self.dir_hr, s, v, '*' + self.ext[1])
            # lrf = glob.glob(lrp)
            # hrf = glob.glob(hrp)
            # print('lr: ', lrp)
            # print('hr: ', hrp)
            # assert len(lrf) == len(hrf)
            videos_lr[vid] = sorted(
                glob.glob(os.path.join(self.dir_lr, s, v, '*' + self.ext[0]))
            )
            videos_hr[vid] = sorted(
                glob.glob(os.path.join(self.dir_hr, s, v, '*' + self.ext[1]))
            )

            assert len(videos_hr[vid]) == len(videos_lr[vid])

        video_names = list(videos_lr.keys())

        for vid in video_list:
            filenames[vid] = [self._get_filename(ip) for ip in videos_lr[vid]]
        
        videos = (videos_lr, videos_hr)

        return videos, video_names, filenames

    def _get_filename(self, path):
        fn, _ = os.path.splitext(os.path.basename(path))
        vp = os.path.normpath(path).split(os.sep)
        seq, video = vp[-3], vp[-2]

        # print('vn:', video)
        filename = seq + '-' + video + '-' + fn

        return filename

    def _set_filesystem(self, data_dir):
        self.apath = os.path.join(data_dir, 'vimeo', 'vimeo_septuplet')

        self.dir_hr = os.path.join(self.apath, 'sequences')
        self.dir_lr = os.path.join(self.apath, 'sequences_qp37_down4')
        if self.is_train and not self.is_valid:
            self.vid_list_f = 'sep_trainlist.txt'
        elif self.is_train and self.is_valid:
            self.vid_list_f = 'sep_vallist.txt'
        else:
            self.vid_list_f = 'sep_testlist.txt'

        self.ext = ('.png', '.png')