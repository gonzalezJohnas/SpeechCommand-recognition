import torch.nn as nn
import torch
import torch.nn.functional as F
import math

import torchaudio

import utils.logger as logger

class MFCCEncoder(nn.Module):
    """
        Input: (N, samples_i) numeric tensor
        Output: (N, samples_o, channels) numeric tensor
    """
    def __init__(self, channels, layer_specs):
        super().__init__()

        self.mfccconvs = nn.ModuleList()

        self.convs_wide = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.layer_specs = layer_specs
        self.mfcc = torchaudio.transforms.MFCC(sample_rate=22050)

        prev_channels = 1
        total_scale = 1
        pad_left = 0
        self.skips = []
        for stride, ksz, dilation_factor in layer_specs:
            conv_wide = nn.Conv1d(prev_channels, 2 * channels, ksz, stride=stride, dilation=dilation_factor)
            wsize = 2.967 / math.sqrt(ksz * prev_channels)
            conv_wide.weight.data.uniform_(-wsize, wsize)
            conv_wide.bias.data.zero_()
            self.convs_wide.append(conv_wide)


            conv_1x1 = nn.Conv1d(channels, channels, 1)
            conv_1x1.bias.data.zero_()
            self.convs_1x1.append(conv_1x1)

            prev_channels = channels
            skip = (ksz - stride) * dilation_factor
            pad_left += total_scale * skip
            logger.log(f'pad += {total_scale} * {ksz-stride} * {dilation_factor}')
            self.skips.append(skip)
            total_scale *= stride
        self.pad_left = pad_left
        self.total_scale = total_scale

        prev_mfcc_channels = 1

        for i in range(2):
            mfcc_iconv = nn.Conv2d(prev_mfcc_channels, channels, kernel_size=4)
            mfcc_iconv.bias.data.zero_()

            prev_mfcc_channels = channels
            self.mfccconvs.append(mfcc_iconv)

        mfcc_iconv = nn.Conv2d(prev_mfcc_channels, channels, kernel_size=5, stride=2)
        mfcc_iconv.bias.data.zero_()

        self.mfccconvs.append(mfcc_iconv)

        for i in range(2):
            mfcc_iconv = nn.Conv2d(prev_mfcc_channels, channels, kernel_size=3)
            mfcc_iconv.bias.data.zero_()

            prev_mfcc_channels = channels
            self.mfccconvs.append(mfcc_iconv)

        for i in range(4):
             self.mfccconvs.append(nn.Conv2d(channels, channels, kernel_size=1))

        self.mfcc_conv_0 = nn.Conv2d(channels, channels, 1)
        self.mfcc_conv_0.bias.data.zero_()
        self.mfcc_conv_1 = nn.Conv2d(channels, channels, 1)

        self.final_conv_0 = nn.Conv1d(channels, channels, 1)
        self.final_conv_0.bias.data.zero_()
        self.final_conv_1 = nn.Conv1d(channels, channels, 1)
        # We don't set the bias to 0 here because otherwise the initial model
        # would produce the 0 vector when the input is 0, and it will make
        # the vq layer unhappy.

    def forward(self, samples):
        print("samples shape!: ", samples.shape)
        mfccres = self.mfcc.forward(samples).unsqueeze(1)
        print("mfcc shape: ", mfccres.shape)

        for i, mfcc_iconv in enumerate(self.mfccconvs):
            mfccres = mfcc_iconv(mfccres)


        mfccres = mfccres.view(16, 128, -1).transpose(1, 2)

        print("mfcc conv shape: ", mfccres.shape)

        mfccres = F.pad(mfccres, [0, 0, 0, 2, 0, 0], value=0)

        print("padded shape: ", mfccres.shape)

        x = samples.unsqueeze(1)

        #logger.log(f'sd[samples] {x.std()}')
        for i, stuff in enumerate(zip(self.convs_wide, self.convs_1x1, self.layer_specs, self.skips)):
            conv_wide, conv_1x1, layer_spec, skip = stuff
            stride, ksz, dilation_factor = layer_spec

            x1 = conv_wide(x)
            #logger.log(f'sd[conv.s] {x1.std()}')
            x1_a, x1_b = x1.split(x1.size(1) // 2, dim=1)
            x2 = torch.tanh(x1_a) * torch.sigmoid(x1_b)
            #logger.log(f'sd[act] {x2.std()}')
            x3 = conv_1x1(x2)
            #logger.log(f'sd[conv.1] {x3.std()}')
            if i == 0:
                x = x3
            else:
                x = x3 + x[:, :, skip:skip+x3.size(2)*stride].view(x.size(0), x3.size(1), x3.size(2), -1)[:, :, :, -1]
            #logger.log(f'sd[out] {x.std()}')
        x = self.final_conv_1(F.relu(self.final_conv_0(x)))
        #logger.log(f'sd[final] {x.std()}')

        print("final shape!: ", x.transpose(1, 2).shape)
        return mfccres
