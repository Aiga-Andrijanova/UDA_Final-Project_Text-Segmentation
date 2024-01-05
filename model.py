import math
import os
import cv2
import torch
import torchvision
import torch.utils.data
import seg_helpers

from torch import nn

import numpy as np
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_conv_bias=False, kernel_size=3, stride=1):
        super().__init__()

        kernel_size_first = kernel_size
        padding = int(kernel_size/2)
        if kernel_size % 2 == 0:
            kernel_size_first = kernel_size - 1
            padding -= 1

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size_first,
                               stride=1,
                               padding=int(kernel_size_first/2), bias=is_conv_bias)
        if in_channels%2 != 0:
            self.gn1 = nn.GroupNorm(num_channels=in_channels, num_groups=1)
        else:
            self.gn1 = nn.GroupNorm(num_channels=in_channels, num_groups=math.ceil(in_channels/2))


        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride,
                               padding=padding, bias=is_conv_bias)
        if in_channels%2 != 0:
            self.gn2 = nn.GroupNorm(num_channels=out_channels, num_groups=1)
        else:
            self.gn2 = nn.GroupNorm(num_channels=out_channels, num_groups=math.ceil(in_channels/2))

        

        self.is_projection = False
        if stride > 1 or in_channels != out_channels:
            self.is_projection = True
            self.conv_res = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                                      padding=1, bias=is_conv_bias)

    def forward(self, x):
        # Batch, Channel, W
        residual = x

        out = self.conv1(x)
        out = F.leaky_relu(out, 0.1, False)
        out = self.gn1(out)

        out = self.conv2(out)
        if self.is_projection:
            residual = self.conv_res(x)

        out += residual
        out = F.leaky_relu(out, 0.1, False)
        out = self.gn2(out)

        return out


# unet3+ with resblocks
class Unet3Plus(nn.Module):
    def __init__(self):
        super(Unet3Plus, self).__init__()

        self.deep_supervision = False

        self.depth = 6
        self.first_layer_channels = 3
        self.expand_rate = 2

        self.channels = [self.first_layer_channels, self.first_layer_channels]
        for d in range(self.depth-1):
            self.channels.append(self.channels[-1]*self.expand_rate)

        # make encoder
        self.encoder = nn.ModuleList()
        for idx in range(self.depth):
            enc_modules = []
            if idx > 0:
                enc_modules.append(nn.MaxPool2d(kernel_size=2, stride=2))
            enc_modules.append(ResBlock(self.channels[idx], self.channels[idx+1]))
            self.encoder.append(nn.Sequential(*enc_modules))

        self.channels_for_concat = 3
        self.concat_channels = self.channels_for_concat * self.depth
        self.output_channels = 1

        # make decoder
        self.decoder = nn.ModuleList()
        # reverse order, start from deepest layer
        for idx_step in range(self.depth-1)[::-1]:
            decoder_step = nn.ModuleList()
            for idx in range(self.depth):
                seq_list = []
                delta = abs(idx - idx_step)
                in_channels = self.channels[idx+1]
                # values from deeper layers upsample
                if idx > idx_step:
                    seq_list.append(nn.Upsample(scale_factor=2**delta, mode='bilinear', align_corners=True))
                    # all except last have self.concat_channels
                    if idx != self.depth -1:
                        in_channels = self.concat_channels
                # values from shallower layers downsample
                elif idx < idx_step:
                    seq_list.append(nn.MaxPool2d(kernel_size=2**delta, stride=2**delta))
                if self.channels_for_concat%2 != 0:
                    seq_list.extend([
                        nn.Conv2d(in_channels, self.channels_for_concat, 3, padding=1),
                        nn.GroupNorm(num_channels=self.channels_for_concat, num_groups=1),
                        nn.LeakyReLU(negative_slope=0.1, inplace=True)
                    ])
                else:
                    seq_list.extend([
                        nn.Conv2d(in_channels, self.channels_for_concat, 3, padding=1),
                        
                        nn.GroupNorm(num_channels=self.channels_for_concat, num_groups=math.ceil(self.channels_for_concat/2)),
                        nn.LeakyReLU(negative_slope=0.1, inplace=True)
                    ])
                decoder_step.append(nn.Sequential(*seq_list))

            decoder_step.append(nn.Sequential(
                nn.Conv2d(self.concat_channels, self.concat_channels, kernel_size=3, padding=1),
                nn.GroupNorm(num_channels=self.concat_channels, num_groups=math.ceil(self.concat_channels/2)),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
            ))
            self.decoder.append(decoder_step)

        # reverse back the order
        self.decoder = self.decoder[::-1]

        # make output layers
        self.output_count = 1
        if self.deep_supervision:
            self.output_count = self.depth

        self.output_layers = nn.ModuleList()
        # make starting from last output
        for idx in range(self.output_count)[::-1]:
            output_layer = []
            delta = abs(self.output_count - idx -1)
            # channels[-1] only for last if output_count not 1
            in_channels = self.concat_channels if idx != 0 or self.output_count == 1 else self.channels[-1]
            if delta > 0:
                output_layer.append(nn.Upsample(scale_factor=2**delta, mode='bilinear', align_corners=True))
            output_layer.append(nn.Conv2d(in_channels, self.output_channels, 3, padding=1))
            self.output_layers.append(nn.Sequential(*output_layer))

    def forward(self, input):
        value = input

        # if no batch dim then unsqueeze
        if len(input.shape) == 3:
            value = input.unsqueeze(0) #Batch, Channel, Width, Height

        # container for encoder output values
        encoder_values = []
        for idx in range(self.depth):
            encoder_step = self.encoder[idx]
            value = encoder_step.forward(value)
            encoder_values.append(value)

        # last encoder value is first decoder value
        # order reversed so that can be filled with idx_step later
        decoder_values = [0] * self.depth
        decoder_values[-1] = encoder_values[-1]

        # reverse order, start from deepest layer
        for idx_step in range(self.depth-1)[::-1]:
            decoder_step = self.decoder[idx_step]
            step_outputs = []
            for idx in range(self.depth):
                if idx > idx_step:
                    # take from decoder values
                    value = decoder_values[idx]
                else:
                    # take from encoder values
                    value = encoder_values[idx]

                module = decoder_step[idx]
                value = module.forward(value)
                step_outputs.append(value)
            value = decoder_step[-1].forward(torch.cat(step_outputs, dim=1))
            decoder_values[idx_step] = value

        output = []
        for idx in range(self.output_count):
            module = self.output_layers[idx]
            value = module.forward(decoder_values[idx])

            value = torch.sigmoid(value)

            # if needed remove channel dim
            if len(input.size()) == 3:
                value = value.squeeze(1) #Batch, Width, Height

            output.append(value)

        if self.output_count == 1:
            output = output[0]

        return output


class DiceLoss(nn.Module):
    """
    Dice loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        """
        Forward pass.
        """

        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice
