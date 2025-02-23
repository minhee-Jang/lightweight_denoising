from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options

        # parser.set_defaults(load_epoch='best')
        parser.add_argument('--url', default=False, action='store_true',
            help='download checkpoint from url')
        parser.add_argument('--ensemble', default=False, action='store_true',
            help='do self ensemble')
        parser.add_argument('--compare', default=False, action='store_true',
            help='generate compare images')
            
        parser.add_argument('--test_results_dir', type=str, default=None,
            help='saves results here.')
        parser.add_argument('--patch_offset', type=int, default=5,
            help='size of patch offset')
            
        self.is_train = False
        return parser
