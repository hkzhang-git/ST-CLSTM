import torch.nn as nn
from torch.autograd import Variable
import torch
from models_CLTSM.non_local import Non_local


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, padding, dilation):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding,
                              dilation=dilation
                              )

        self.non_local = Non_local(in_channels=input_dim, out_channels=hidden_dim)

    def forward(self, pre_h, pre_d, cur_h, cell_state):

        attention_pre_d = self.non_local(pre_h, pre_d, cur_h, cell_state)

        combined_info = torch.cat((pre_h, cur_h), dim=1)
        cc_ft = torch.sigmoid(self.cc_ft(combined_info))
        cc_it = torch.sigmoid(self.cc_it(combined_info))


        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, padding, dilation, num_layers):
        super(ConvLSTM, self).__init__()

        self.num_layers = num_layers

        cell_list = []
        for i in range(0, num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=hidden_dim,
                                          kernel_size=kernel_size,
                                          padding=padding,
                                          dilation=dilation
                                          ))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
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
        h_state_init = torch.zeros(b, c, h, w).to('cuda')
        c_state_init = torch.zeros(b, c, h, w).to('cuda')

        seq_len = input_tensor.size(2)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = h_state_init, c_state_init
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, :, t, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=2)
            cur_layer_input = layer_output

        return layer_output


class R_CLSTM_1(nn.Module):
    def __init__(self, block_channel):
        super(R_CLSTM_1, self).__init__()

        num_features = 64 + block_channel[3] // 32

        self.shortcut = nn.Conv2d(num_features, num_features, kernel_size=3,
                                  stride=1, padding=1, bias=False)

        self.CLSTM = ConvLSTM(num_features, num_features, kernel_size=3, padding=2, dilation=2, num_layers=2)

        self.conv = nn.Sequential(
            nn.Conv2d(num_features, 1,
                      kernel_size=5, stride=1, padding=2, bias=False),
        )

    def forward(self, x, b, d):
        x_cubes = self.maps_2_cubes(x, b, d)

        xs = self.shortcut(x)

        x_clstm = self.CLSTM(x_cubes)

        x2 = self.conv(self.cubes_2_maps(x_clstm) + xs)

        return x2

    def maps_2_cubes(self, maps, b, d):
        x_b, x_c, x_h, x_w = maps.shape
        maps = maps.contiguous().view(b, d, x_c, x_h, x_w)

        return maps.permute(0, 2, 1, 3, 4)

    def cubes_2_maps(self, cubes):
        b, c, d, h, w = cubes.shape
        cubes = cubes.permute(0, 2, 1, 3, 4)

        return cubes.contiguous().view(b * d, c, h, w)


