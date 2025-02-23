import os
import glob

from data.srdata import SRData

class mayo_forover(SRData):
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(n_channels=1)
        parser.set_defaults(pixel_range=1.0)
        parser.set_defaults(scale=1)
        return parser
        
    def __init__(self, args, name='mayo_forover', is_train=True, is_valid=False):
        super(mayo_forover, self).__init__(
            args, name=name, is_train=is_train, is_valid=is_valid
        )

    def _scan(self):
        videos_lr = {}
        videos_hr = {}
        videos_maskE = {}
        videos_maskNE = {}
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
            videos_maskE[vid] = sorted(
                glob.glob(os.path.join(self.dir_maskE, vid, '*' + self.ext))
            )
            videos_maskNE[vid] = sorted(
                glob.glob(os.path.join(self.dir_maskNE, vid, '*' + self.ext))
            )



            assert len(videos_lr[vid]) == len(videos_hr[vid])


        # print(len(videos_maskNE)) #7

        video_names = list(videos_lr.keys())
        for vid in video_list:
            filenames[vid] = [self._get_filename(ip) for ip in videos_lr[vid]]

        videos = (videos_lr, videos_hr, videos_maskE, videos_maskNE)  #dictionary 갯수니까 2개가 맞음 
        #videos = (videos_lr, videos_hr) 

        return videos, video_names, filenames

    def _get_filename(self, path):
        filename, _ = os.path.splitext(os.path.basename(path))

        return filename

    def _set_filesystem(self, data_dir):
        #self.apath = os.path.join(data_dir, 'mayo2d', 'mayo2d')
        self.apath = os.path.join(data_dir, 'mayo2d', 'mayo_forover')

        self.dir_lr = os.path.join(self.apath, self.mode, 'quarter_1mm')
        self.dir_hr = os.path.join(self.apath, self.mode, 'full_1mm')
        self.dir_maskE = os.path.join(self.apath, self.mode, 'Edge')
        self.dir_maskNE = os.path.join(self.apath, self.mode, 'NEdge')
    

        self.ext = ('.tiff')