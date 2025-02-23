"""
Transformer for Video
see: https://pytorch.org/tutorials/beginner/translation_transformer.html
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from models.base_model import BaseModel
from models.common.position_encoding import PositionEmbeddingSineTokens
from utils.tester import Tensor2PatchDataset
from torch.utils import data

class Vid2Vid(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # * Transformer
        parser.add_argument('--num_encoder_layers', default=6, type=int,
            help="Number of encoding layers in the transformer")
        # parser.add_argument('--num_decoder_layers', default=6, type=int,
        #     help="Number of decoding layers in the transformer")
        parser.add_argument('--dim_feedforward', default=2048, type=int,
            help="Intermediate size of the feedforward layers in the transformer blocks")
        # parser.add_argument('--emb_size', default=256, type=int,
        #     help="Size of the embeddings (dimension of the transformer)")
        parser.add_argument('--dropout', default=0.1, type=float,
            help="Dropout applied in the transformer")
        parser.add_argument('--activation', default='relu',
            help='the activation function of the intermediate layer (“relu” or “gelu”)')
        parser.add_argument('--nhead', default=4, type=int,
            help="Number of attention heads inside the transformer's attentions")
        parser.add_argument('--pre_norm', action='store_true')

        parser.add_argument('--n_token', default=1, type=int,
            help="Number of tokens in input")

        parser.set_defaults(patch_size=32)
        parser.set_defaults(n_frames=7)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.model_names = ['transformer']
        self.transformer = create_model(opt).to(self.device)
        self.n_frames = opt.n_frames
        self.opt = opt

        # Define losses and optimizers
        if self.is_train:
            self.loss_criterion = nn.MSELoss()
            self.mse_loss_criterion = nn.MSELoss()

            self.optimizer_names = ['optimizer']
            self.optimizer = torch.optim.Adam(
                self.transformer.parameters(),
                lr=opt.lr,
                betas=(opt.b1, opt.b2),
                eps=1e-8,
                weight_decay=0
            )
            self.optimizers.append(self.optimizer)

        self.forward_ensemble = None
        if not self.is_train and opt.ensemble:
            from .common.ensemble import SpatialTemporalEnsemble
            self.forward_ensemble = SpatialTemporalEnsemble(is_temporal_ensemble=False)

    def set_input(self, input):
        # b, c, n, h, w = self.x.shape
        self.x = input['lr'].to(self.device)
        if 'hr' in input:
            self.target = input['hr'].to(self.device)

    def forward(self):
        self.out = self.transformer(self.x)

    def test_ensemble(self):
        with torch.no_grad():
            self.out = self.forward_ensemble(self.x, self.net)
    def backward(self):
        self.loss.backward()

    def calc_loss(self):
        self.loss = self.loss_criterion(self.out, self.target)
        mse_loss = self.mse_loss_criterion(self.out.detach(), self.target.detach())
        self.psnr = 10 * torch.log10(1 / mse_loss)

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        self.calc_loss()
        self.backward()
        self.optimizer.step()

    def test(self):
        self.out = self.transformer(self.x)

    def get_logs(self):
        log_dict = {
            'loss': '{:.8f}'.format(self.loss),
            # 'mse_loss': '{:.8f}'.format(mse_loss),
            'psnr': '{:.8f}'.format(self.psnr)
        }

        return log_dict

    def get_batch_measure(self):
        return self.loss.detach(), self.psnr.detach()

    def predict(self, video):
        n_frames = self.n_frames
        x = video['lr']
        _, c, n, h, w = x.shape # here n is the number of whole images in the video

        predicted_imgs = []
        # predicted_idxs = []

        for d in range(0, n, n_frames):
            mp_start_time = time.time()

            xd = x[:, :, d:d+n_frames]
            tensor_x = xd

            print('[*] Frames[{}:{}]'.format(d, d+xd.shape[2]))

            img_patch_dataset = Tensor2PatchDataset(self.opt, tensor_x)
            img_patch_dataloader = data.DataLoader(
                dataset=img_patch_dataset,
                batch_size=self.opt.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=self.opt.n_threads
            )
            
            img_shape = img_patch_dataset.get_img_shape()
            pad_img_shape = img_patch_dataset.get_padded_img_shape()

            mp_end_time = time.time()
            print("[*] Input shape", tensor_x.shape)
            print("[*] Making patches: {:4f}s".format((mp_end_time - mp_start_time)))

            net_start_time = time.time()
            print("[*] Network started")
            out_list = []

            for img_patch in img_patch_dataloader:
                # print("batch:", batch.shape)
                input = {
                    'lr': img_patch
                }

                self.set_input(input)
                with torch.no_grad():
                    if self.forward_ensemble is not None:
                        print("Forward ensemble")
                        self.test_ensemble()
                    else:
                        self.test()


                out = self.out
                out_list.append(out)
                
            net_end_time = time.time()
            print("[*] Network process: {:4f}s".format((net_end_time - net_start_time)))

            out = torch.cat(out_list, dim=0)
            
            recon_start_time = time.time()
            
            out_img = img_patch_dataset.recon_tensor_arr_patches(out)
            # out_img = unpad_tensor(out_img, opt.patch_offset, img_shape)
            
            recon_end_time = time.time()
            print('[*] Reconstructed out_img.shape:', out_img.shape)
            print("[*] Reconstruction time: {:4f}s".format((recon_end_time - recon_start_time)))
            print("[*] Total time {:.4f}s".format(recon_end_time - mp_start_time))

            # for i in range(out_img.shape[2]):
            #     predicted_idx = d + i
            #     predicted_idxs.append(predicted_idx)
            
            predicted_imgs.append(out_img)

        predicted_imgs = torch.cat(predicted_imgs, dim=2)
        print('[*] Output shape:', predicted_imgs.shape)
        return predicted_imgs


def create_model(opt):
    return Vid2VidTransformer(opt)


# Learned perceptual metric
class Vid2VidTransformer(nn.Module):
    def __init__(self, opt):
        super(Vid2VidTransformer, self).__init__()
        self.dev = opt.device
        self.img_sz = opt.patch_size
        self.nc = opt.n_channels

        
        emb_size = opt.patch_size * opt.patch_size * opt.n_channels # image vector size
        self.n_token = opt.n_token
        encoder_layer = nn.TransformerEncoderLayer(
                            d_model=emb_size,
                            nhead=opt.nhead,
                            dim_feedforward=opt.dim_feedforward,
                            dropout=opt.dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, opt.num_encoder_layers, norm=None)
        self.generator = nn.Linear(emb_size, emb_size)
        self.positional_encoding = PositionEmbeddingSineTokens(
            emb_size, dropout=opt.dropout, maxlen=opt.n_frames, n_token=self.n_token)

    def forward(self, src: Tensor):
        # print('input', src.shape)
        src_emb = self.positional_encoding(self._flatten_img(src))
        # print('end position_embedding:', src_emb.shape)
        src_mask = self._create_mask(src_emb)
        # print('src_mask.shape:', src_mask.shape)
        # print('tgt_mask.shape:', tgt_mask.shape)
        outs = self.encoder(src_emb, src_mask)
        outs = self.generator(outs)
        outs = self._recon_img(outs)
        outs = outs[:, :, self.n_token:]

        outs = outs + src
        return outs

    def _flatten_img(self, x: Tensor):
        # bs, c, n, h, w -> n, bs, (c*h*w)
        bs, c, n, h, w = x.shape
        # print('x.shape:', x.shape)
        x = x.transpose(1, 2)
        # print('x.shape:', x.shape)
        x = x.reshape(bs, n, -1)
        x = x.transpose(0, 1)
        # print('x_flatten.shape:', x.shape)
        return x

    def _recon_img(self, x: Tensor):
        # n, bs, (n*h*w)
        n, bs, c = x.shape
        # print('x.shape:', x.shape)
        x = x.transpose(0, 1)
        # print('x.shape:', x.shape)
        x = x.reshape(bs, self.nc, n, self.img_sz, self.img_sz)
        # print('recon_x.shape:', x.shape)
        return x


    def _create_mask(self, src: Tensor):
        src_seq_len = src.shape[0]
        src_mask = torch.zeros((src_seq_len, src_seq_len),device=src.device).type(torch.bool)

        # src_padding_mask = (src == PAD_IDX).transpose(0, 1) # NLP 
        return src_mask
