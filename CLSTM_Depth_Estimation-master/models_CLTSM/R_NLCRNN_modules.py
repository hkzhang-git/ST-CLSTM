import torch.nn as nn
import torch
from models_CLTSM.non_local import Non_local


class ConvRNNCell(nn.Module):

    def __init__(self, input_dim, inter_dim, hidden_dim):
        super(ConvRNNCell, self).__init__()

        self.non_local = Non_local(in_channels=input_dim, inter_channels=inter_dim, out_channels=inter_dim)
        num_features_in = 128 + inter_dim
        num_features = 128

        self.refine = nn.Sequential(
            nn.Conv2d(num_features_in, num_features,
                      kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(num_features),
            nn.Conv2d(num_features, num_features,
                      kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(num_features),
            nn.Conv2d(num_features, 1, kernel_size=5, stride=1, padding=2, bias=True)
        )

    def forward(self, pre_h, pre_d, cur_h):
        attention_pre_d = self.non_local(pre_h, pre_d, cur_h)
        combined_info = torch.cat((pre_h, attention_pre_d), dim=1)

        out = self.refine(combined_info)

        return out


class ConvRNN(nn.Module):

    def __init__(self, input_dim, inter_dim,  hidden_dim):
        super(ConvRNN, self).__init__()

        self.Cell = ConvRNNCell(input_dim, inter_dim, hidden_dim)

    def forward(self, input_tensor):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        b, c, d, h, w = input_tensor.shape
        pre_d_init = torch.zeros(b, 1, h, w).to('cuda')
        pre_h_init = torch.zeros(b, c, h, w).to('cuda')

        seq_len = input_tensor.size(2)

        output_inner = []
        for t in range(seq_len):
            if t == 0:
                pre_d = self.Cell(pre_h=pre_h_init, pre_d=pre_d_init, cur_h=input_tensor[:, :, t, :, :])
                pre_h = input_tensor[:, :, t, :, :]
                output_inner.append(pre_d)
            else:
                pre_d = self.Cell(pre_h=pre_h, pre_d=pre_d, cur_h=input_tensor[:, :, t, :, :])
                pre_h = input_tensor[:, :, t, :, :]
                output_inner.append(pre_d)

        cell_output = torch.stack(output_inner, dim=2)

        return cell_output


class R_NLCRNN_1(nn.Module):
    def __init__(self, block_channel):
        super(R_NLCRNN_1, self).__init__()

        num_features = 64 + block_channel[3] // 32

        self.CRNN = ConvRNN(num_features, 3, num_features)

    def forward(self, x, b, d):
        x_cubes = self.maps_2_cubes(x, b, d)

        x_clstm = self.CRNN(x_cubes)

        #  x2 = self.conv(self.cubes_2_maps(x_clstm))

        return self.cubes_2_maps(x_clstm)

    def maps_2_cubes(self, maps, b, d):
        x_b, x_c, x_h, x_w = maps.shape
        maps = maps.contiguous().view(b, d, x_c, x_h, x_w)

        return maps.permute(0, 2, 1, 3, 4)

    def cubes_2_maps(self, cubes):
        b, c, d, h, w = cubes.shape
        cubes = cubes.permute(0, 2, 1, 3, 4)

        return cubes.contiguous().view(b * d, c, h, w)


