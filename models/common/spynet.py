"""
https://github.com/sniklaus/pytorch-spynet
"""

import math
import numpy as np

import torch

class SpyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        class Preprocess(torch.nn.Module):
            def __init__(self):
                super().__init__()
            # end

            def forward(self, tenInput):
                if tenInput.shape[1] == 1:
                    tenInput = torch.cat((tenInput, tenInput, tenInput), 1)
                tenInput = tenInput.flip([1])
                tenInput = tenInput - torch.tensor(data=[0.485, 0.456, 0.406], dtype=tenInput.dtype, device=tenInput.device).view(1, 3, 1, 1)
                tenInput = tenInput * torch.tensor(data=[1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225], dtype=tenInput.dtype, device=tenInput.device).view(1, 3, 1, 1)

                return tenInput
            # end
        # end

        class Basic(torch.nn.Module):
            def __init__(self, intLevel):
                super().__init__()

                self.netBasic = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
                )
            # end

            def forward(self, tenInput):
                return self.netBasic(tenInput)
            # end
        # end

        self.netPreprocess = Preprocess()

        self.netBasic = torch.nn.ModuleList([ Basic(intLevel) for intLevel in range(6) ])

        arguments_strModel = 'sintel-final'
        self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-spynet/network-' + arguments_strModel + '.pytorch', file_name='spynet-' + arguments_strModel).items() })

        self.backwarp_tenGrid = {}
    # end

    def forward(self, tenOne, tenTwo):
        print('tenOne.dim:', tenOne.dim())
        assert(tenOne.dim() == 4)
        assert(tenOne.shape[2] == tenTwo.shape[2])
        assert(tenOne.shape[3] == tenTwo.shape[3])

        bs, c, intHeight, intWidth = tenOne.shape
            
        # intWidth, intHeight = tenOne.shape
        # c = tenOne.shape[0]
        # intWidth = tenOne.shape[2]
        # intHeight = tenOne.shape[1]

        tenPreprocessedOne = tenOne.view(bs, c, intHeight, intWidth)
        tenPreprocessedTwo = tenTwo.view(bs, c, intHeight, intWidth)

        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

        tenPreprocessedOne = torch.nn.functional.interpolate(input=tenPreprocessedOne, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
        tenPreprocessedTwo = torch.nn.functional.interpolate(input=tenPreprocessedTwo, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)


        tenFlow = []

        tenOne = [ self.netPreprocess(tenPreprocessedOne) ]
        tenTwo = [ self.netPreprocess(tenPreprocessedTwo) ]

        for intLevel in range(5):
            if tenOne[0].shape[2] > 32 or tenOne[0].shape[3] > 32:
                tenOne.insert(0, torch.nn.functional.avg_pool2d(input=tenOne[0], kernel_size=2, stride=2, count_include_pad=False))
                tenTwo.insert(0, torch.nn.functional.avg_pool2d(input=tenTwo[0], kernel_size=2, stride=2, count_include_pad=False))
            # end
        # end

        tenFlow = tenOne[0].new_zeros([ tenOne[0].shape[0], 2, int(math.floor(tenOne[0].shape[2] / 2.0)), int(math.floor(tenOne[0].shape[3] / 2.0)) ])

        for intLevel in range(len(tenOne)):
            tenUpsampled = torch.nn.functional.interpolate(input=tenFlow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            if tenUpsampled.shape[2] != tenOne[intLevel].shape[2]: tenUpsampled = torch.nn.functional.pad(input=tenUpsampled, pad=[ 0, 0, 0, 1 ], mode='replicate')
            if tenUpsampled.shape[3] != tenOne[intLevel].shape[3]: tenUpsampled = torch.nn.functional.pad(input=tenUpsampled, pad=[ 0, 1, 0, 0 ], mode='replicate')

            tenFlow = self.netBasic[intLevel](torch.cat([ tenOne[intLevel], self.backwarp(tenInput=tenTwo[intLevel], tenFlow=tenUpsampled), tenUpsampled ], 1)) + tenUpsampled
        # end

        # return tenFlow
        tenFlow = torch.nn.functional.interpolate(input=tenFlow, size=(intHeight, intWidth), mode='bilinear', align_corners=False)
        tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
        tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)
        return tenFlow

    def backwarp(self, tenInput, tenFlow):
        if str(tenFlow.shape) not in self.backwarp_tenGrid:
            tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
            tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])

            self.backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()
        # end

        tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

        return torch.nn.functional.grid_sample(input=tenInput, grid=(self.backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=False)
    # end

def estimate(tenOne, tenTwo, netNetwork=None):
    # if netNetwork is None:
        # netNetwork = Network().cuda().eval()
    netNetwork = netNetwork.cuda().eval()
    # end

    assert(tenOne.shape[1] == tenTwo.shape[1])
    assert(tenOne.shape[2] == tenTwo.shape[2])

    c = tenOne.shape[0]
    intWidth = tenOne.shape[2]
    intHeight = tenOne.shape[1]

    # assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    # assert(intHeight == 416) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    tenPreprocessedOne = tenOne.cuda().view(1, c, intHeight, intWidth)
    tenPreprocessedTwo = tenTwo.cuda().view(1, c, intHeight, intWidth)

    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

    tenPreprocessedOne = torch.nn.functional.interpolate(input=tenPreprocessedOne, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
    tenPreprocessedTwo = torch.nn.functional.interpolate(input=tenPreprocessedTwo, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

    tenFlow = torch.nn.functional.interpolate(input=netNetwork(tenPreprocessedOne, tenPreprocessedTwo), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

    tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    return tenFlow[0, :, :, :].cpu()


if __name__ == '__main__':
    import PIL
    import PIL.Image
    import matplotlib.pyplot as plt
    from skimage.io import imread, imsave
    from skimage.color import rgb2gray

    import os, sys
    utils_path = os.path.join(sys.path[0], '../..')
    utlis_path = os.path.abspath(utils_path)
    sys.path.append(utils_path)

    from models.common.flow_vis import flow_to_color
    from models.mmedit.models.common.flow_warp import flow_warp

    spynet = SpyNet()
    # print(spynet)
    
    
    # p1 = r'E:\data\tests\one.png'
    # p2 = r'E:\data\tests\two.png'
    # one_np = np.array(PIL.Image.open(p1))
    # two_np = np.array(PIL.Image.open(p2))
    # pm = 255.0

    
    # p1 = r'E:\data\video-sr\moving700\train\low\rotate_set1\rotate_set1-000.tiff'
    # p2 = r'E:\data\video-sr\moving700\train\low\rotate_set1\rotate_set1-001.tiff'

    p1 = r'E:\data\video-sr\moving700\train\low\hand_linear_set2\hand_linear_set2-000.tiff'
    p2 = r'E:\data\video-sr\moving700\train\low\hand_linear_set2\hand_linear_set2-011.tiff'

    one_np = imread(p1)
    two_np = imread(p2)
    if one_np.ndim == 2:
        one_np = np.expand_dims(one_np, axis=2)
    if two_np.ndim == 2:
        two_np = np.expand_dims(two_np, axis=2)
    pm = 1.0


    # tenOne = torch.FloatTensor(np.ascontiguousarray(np.array(PIL.Image.open(p1))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
    # tenTwo = torch.FloatTensor(np.ascontiguousarray(np.array(PIL.Image.open(p2))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
    # tenOne = torch.FloatTensor(np.ascontiguousarray(one_np[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
    # tenTwo = torch.FloatTensor(np.ascontiguousarray(two_np[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
    tenOne = torch.FloatTensor(np.ascontiguousarray(one_np.transpose(2, 0, 1).astype(np.float32) * (1.0 / pm)))
    tenTwo = torch.FloatTensor(np.ascontiguousarray(two_np.transpose(2, 0, 1).astype(np.float32) * (1.0 / pm)))


    # tenOutput = estimate(tenOne, tenTwo, netNetwork=spynet)
    tenOutput = estimate(tenTwo, tenOne, netNetwork=spynet)
    print('tenOutput.shape:', tenOutput.shape)

    # arguments_strOut = './out.flo'
    # objOutput = open(arguments_strOut, 'wb')

    # np.array([ 80, 73, 69, 72 ], np.uint8).tofile(objOutput)
    # np.array([ tenOutput.shape[2], tenOutput.shape[1] ], np.int32).tofile(objOutput)
    # np.array(tenOutput.detach().numpy().transpose(1, 2, 0), np.float32).tofile(objOutput)

    # objOutput.close()

    flow_np = tenOutput.detach().numpy().transpose(1, 2, 0)
    flow_img = flow_to_color(flow_np)

    print(flow_np)
    plt.imshow(flow_img)
    plt.show()

    c, h, w = tenOne.shape
    print('tenOne.shape:', tenOne.shape)
    one_t = tenOne.view(1, c, h, w)
    two_t = tenTwo.view(1, c, h, w)
    flow_t = tenOutput.permute(1, 2, 0).view(1, h, w, 2)

    warped = flow_warp(one_t, flow_t)
    

    warped_np = warped.detach().numpy().transpose(0, 2, 3, 1)
    warped_img = warped_np.squeeze()
    # warped_img = rgb2gray(warped_img)

    # concat_img = warped_img
    if c == 3:
        print('3 channels')
        warped_img = warped_img * 255
        warped_img = warped_img.astype(np.uint8)
        print('one_np.dtype:', one_np.dtype)
        print('two_np.dtype:', two_np.dtype)
        print('warped_img.dtype:', warped_img.dtype)
        concat_img = np.concatenate((one_np, two_np, warped_img), axis=1)
        print('concat_img.dtype:', concat_img.dtype)
        plt.imshow(concat_img)
    else:
        one_np = one_np.squeeze()
        two_np = two_np.squeeze()
        concat_img = np.concatenate((one_np, two_np, warped_img), axis=1)
        plt.imshow(concat_img, cmap=plt.cm.gray)
        
    plt.show()

    imsave('warping.png', warped_img)
    imsave('flow.png', flow_img)

