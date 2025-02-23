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
        parser.add_argument('--num_decoder_layers', default=6, type=int,
            help="Number of decoding layers in the transformer")
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

        parser.set_defaults(patch_size=16)
        parser.set_defaults(n_frames=64)

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

    def set_input(self, input):
        # b, c, n, h, w = self.x.shape
        self.x = input['lr'].to(self.device)
        if 'hr' in input:
            self.target = input['hr'].to(self.device)

    def forward(self):
        self.out = self.transformer(self.x, self.target)

    def backward(self):
        self.loss.backward()

    def calc_loss(self):
        # print('calc_loss')
        self.loss = self.loss_criterion(self.out, self.target)
        mse_loss = self.mse_loss_criterion(self.out.detach(), self.target.detach())
        self.psnr = 10 * torch.log10(1 / mse_loss)
        # print('end calc_loss')

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        self.calc_loss()
        self.backward()
        self.optimizer.step()

    def test(self):
        self.out = self.transformer(self.x, self.target)

    def get_logs(self):
        log_dict = {
            'loss': '{:.8f}'.format(self.loss),
            # 'mse_loss': '{:.8f}'.format(mse_loss),
            'psnr': '{:.8f}'.format(self.psnr)
        }

        return log_dict

    def get_batch_measure(self):
        return self.loss.detach(), self.psnr.detach()

    def greedy_decode(self):
        # print('self.x.shape:', self.x.shape)
        bs, c, n, h, w = self.x.shape
        memory = self.transformer.encode(
            self.transformer.positional_encoding(self.transformer._flatten_img(self.x))
        )
        ys = self.transformer.positional_encoding.embedding(
            torch.tensor([0], dtype=torch.long, device=torch.device(self.x.device))
        )
        ys = ys.expand(1, bs, c*h*w)
        # print('initial ys.shape:', ys.shape)

        for i in range(n):
            # print('ys.shape:', ys.shape)
            tgt_mask = (self.transformer._generate_square_subsequent_mask(ys)).type(torch.bool).to(ys.device)
            out = self.transformer.decode(ys, memory)
            # print('decode out.shape:', out.shape)
            # out = out.transpose(0, 1)
            # print('transpose out.shape:', out.shape)
            out = self.transformer.generator(out)
            # print('generator out.shape:', out.shape)
            # print('new embedding.shape:', out[i:i+1].shape)

            ys = torch.cat([ys, out[i:i+1]], dim=0) # ys += new out or ys = initial ys + out[1:] ?

        # print('**************** final self.out.shape:', out.shape)
        self.out = self.transformer._recon_img(out)



    def predict_ldv(self, batch):
        n_frames = self.n_frames
        x = batch['lr']
        _, c, n, h, w = x.shape # here n is the number of whole images in the video

        predicted_video = []
        predicted_idxs = []

        for d in range(0, n, n_frames):
            mp_start_time = time.time()

            xd = x[:, :, d:d+n_frames]
            tensor_x = xd

            print('[*] frames[{}:{}]'.format(d, d+xd.shape[2]))

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
            print("[*] input shape", tensor_x.shape)
            print("[*] Making patches: {:4f}s".format((mp_end_time - mp_start_time)))

            net_start_time = time.time()
            print("[*] Network started")
            out_list = []

            for img_patch in img_patch_dataloader:
                # print("batch:", batch.shape)
                input = {
                    'lr': img_patch
                }
                with torch.no_grad():
                    self.set_input(input)
                    self.greedy_decode()

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

            out_np = out_img.detach().to('cpu').numpy()
            # print('[*] out.shape:', out.shape)

            out_np[out_np > 1.0] = 1.0
            out_np[out_np < 0.0] = 0.
            out_np = out_np * 255
            out_np = np.rint(out_np)
            out_np  = out_np.astype(np.uint8)

            for i in range(out_np.shape[2]):
                out_x = out_np[:, :, i].squeeze().transpose(1, 2, 0)

                predicted_idx = d + i + 1
                print('predicted file {:03d}'.format(predicted_idx))
                predicted_video.append(out_x)
                predicted_idxs.append(predicted_idx)

        return predicted_video, predicted_idxs


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
        self.transformer = nn.Transformer(d_model=emb_size,
                                       nhead=opt.nhead,
                                       num_encoder_layers=opt.num_encoder_layers,
                                       num_decoder_layers=opt.num_decoder_layers,
                                       dim_feedforward=opt.dim_feedforward,
                                       dropout=opt.dropout)
        self.generator = nn.Linear(emb_size, emb_size)
        self.positional_encoding = PositionEmbeddingSineTokens(
            emb_size, dropout=opt.dropout, maxlen=opt.n_frames, n_token=self.n_token)

    def forward(self,
                src: Tensor,
                trg: Tensor):
        # print('input', src.shape)
        # print('src.device:', src.device)
        src_emb = self.positional_encoding(self._flatten_img(src))
        tgt_emb = self.positional_encoding(self._flatten_img(trg))
        # print('end position_embedding:', src_emb.shape)
        src_mask, tgt_mask = self._create_mask(src_emb, tgt_emb)
        # print('src_mask.shape:', src_mask.shape)
        # print('tgt_mask.shape:', tgt_mask.shape)
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, 
                                None, None, None)
        # outs = self.transformer(src_emb, tgt_emb, None, None, None, 
                                # None, None, None)
        # print('outs.shape:', outs.shape)
        outs = self.generator(outs)
        # return self.generator(outs)
        outs = self._recon_img(outs)
        outs = outs[:, :, self.n_token:]
        # outs = outs[:, :, :-self.n_token]
        # print('foward ret:', outs.shape)
        outs = outs + src
        return outs

    def encode(self, src_emb: Tensor):
        # src_emb = self.positional_encoding(self._flatten_img(src))
        src_seq_len = src_emb.shape[0]
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=src_emb.device).type(torch.bool)
        return self.transformer.encoder(src_emb, src_mask)

    def decode(self, tgt_emb: Tensor, memory: Tensor):
        # tgt_emb = self.positional_encoding(self._flatten_img(tgt))
        # tgt_seq_len = tgt_emb.shape[0]
        tgt_mask = self._generate_square_subsequent_mask(tgt_emb)
        # tgt_mask = (torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=tgt.device)) == 1).transpose(0, 1)
        # print('tgt_mask.device:', tgt_mask.device)
        # tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))

        return self.transformer.decoder(tgt_emb, memory, tgt_mask)

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

    def _generate_square_subsequent_mask(self, tgt: Tensor):
        # print('self.device:', self.device)
        # print('self.dev:', self.dev)
        tgt_seq_len = tgt.shape[0]
        mask = (torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=tgt.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        # print('mask.device:', mask.device)
        return mask

    def _create_mask(self, src: Tensor, tgt: Tensor):
        # print('src.device:', src.device)
        # tgt_seq_len = tgt.shape[0]

        tgt_mask = self._generate_square_subsequent_mask(tgt)
        # tgt_mask = (torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=tgt.device)) == 1).transpose(0, 1)
        # print('tgt_mask.device:', tgt_mask.device)
        # tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))

        src_seq_len = src.shape[0]
        src_mask = torch.zeros((src_seq_len, src_seq_len),device=src.device).type(torch.bool)

        # src_padding_mask = (src == PAD_IDX).transpose(0, 1) # NLP 
        # tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1) 
        return src_mask, tgt_mask #, src_padding_mask, tgt_padding_mask
