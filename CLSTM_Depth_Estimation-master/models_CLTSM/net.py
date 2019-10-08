import torch
import torch.nn as nn
from models_CLTSM import modules
from models_CLTSM.refinenet_dict import refinenet_dict


def cubes_2_maps(cubes):
    b, c, d, h, w = cubes.shape
    cubes = cubes.permute(0, 2, 1, 3, 4)

    return cubes.contiguous().view(b*d, c, h, w), b, d


class model(nn.Module):
    def __init__(self, Encoder, num_features, block_channel, refinenet):

        super(model, self).__init__()

        self.E = Encoder
        self.D = modules.D(num_features)
        self.MFF = modules.MFF(block_channel)
        self.R = refinenet_dict[refinenet](block_channel)

    def forward(self, x):
        x, b, d = cubes_2_maps(x)
        x_block1, x_block2, x_block3, x_block4 = self.E(x)
        x_decoder = self.D(x_block1, x_block2, x_block3, x_block4)
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4,[x_decoder.size(2),x_decoder.size(3)])
        out = self.R(torch.cat((x_decoder, x_mff), 1), b, d)

        return out
