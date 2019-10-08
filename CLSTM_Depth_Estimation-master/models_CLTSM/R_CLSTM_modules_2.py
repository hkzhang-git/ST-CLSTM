import torch.nn.functional as F
import torch.nn as nn
import torch
import time

def maps_2_cubes(x, b, d):
    x_b, x_c, x_h, x_w = x.shape
    x = x.contiguous().view(b, d, x_c, x_h, x_w)

    return x.permute(0, 2, 1, 3, 4)


def maps_2_maps(x, b, d):
    x_b, x_c, x_h, x_w = x.shape
    x = x.contiguous().view(b, d * x_c, x_h, x_w)

    return x


class R(nn.Module):
    def __init__(self, block_channel):
        super(R, self).__init__()

        num_features = 64 + block_channel[3] // 32
        self.conv0 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features)

        self.conv1 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)

        self.conv2 = nn.Conv2d(
            num_features, 1, kernel_size=5, stride=1, padding=2, bias=True)

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.bn0(x0)
        x0 = F.relu(x0)

        x1 = self.conv1(x0)
        x1 = self.bn1(x1)
        h = F.relu(x1)

        pred_depth = self.conv2(h)

        return h, pred_depth


class R_2(nn.Module):
    def __init__(self, block_channel):
        super(R_2, self).__init__()

        num_features = 64 + block_channel[3] // 32 + 4
        self.conv0 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features)

        self.conv1 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)

        self.conv2 = nn.Conv2d(
            num_features, 1, kernel_size=5, stride=1, padding=2, bias=True)

        self.convh = nn.Conv2d(
            num_features, 4, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.bn0(x0)
        x0 = F.relu(x0)

        x1 = self.conv1(x0)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)

        h = self.convh(x1)
        pred_depth = self.conv2(x1)

        return h, pred_depth


class R_d(nn.Module):
    def __init__(self, block_channel):
        super(R_d, self).__init__()

        num_features = 64 + block_channel[3] // 32 + 4
        self.conv0 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features)

        self.conv1 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)

        self.dropout = nn.Dropout2d(p=0.5)

        self.conv2 = nn.Conv2d(
            num_features, 1, kernel_size=5, stride=1, padding=2, bias=True)

        self.convh = nn.Conv2d(
            num_features, 4, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.bn0(x0)
        x0 = F.relu(x0)

        x1 = self.conv1(x0)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)

        h = self.convh(x1)
        x1 = self.dropout(x1)
        pred_depth = self.conv2(x1)

        return h, pred_depth


class R_3(nn.Module):
    def __init__(self, block_channel):
        super(R_3, self).__init__()

        num_features = 64 + block_channel[3] // 32 + 8
        self.conv0 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features)

        self.conv1 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)

        self.conv2 = nn.Conv2d(
            num_features, 1, kernel_size=5, stride=1, padding=2, bias=True)

        self.convh = nn.Conv2d(
            num_features, 8, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.bn0(x0)
        x0 = F.relu(x0)

        x1 = self.conv1(x0)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)

        h = self.convh(x1)
        pred_depth = self.conv2(x1)

        return h, pred_depth


class R_10(nn.Module):
    def __init__(self, block_channel):
        super(R_10, self).__init__()

        num_features = 64 + block_channel[3] // 32 + 8
        self.conv0 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features)

        self.conv1 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)

        self.conv2 = nn.Conv2d(
            num_features, 1, kernel_size=5, stride=1, padding=2, bias=True)

        self.convh = nn.Conv2d(
            num_features, 8, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.bn0(x0)
        x0 = F.relu(x0)

        x1 = self.conv1(x0)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)

        h = self.convh(x1)
        pred_depth = F.tanh(self.conv2(x1))

        return h, pred_depth


class R_CLSTM_1(nn.Module):
    def __init__(self, block_channel):
        super(R_CLSTM_1, self).__init__()
        num_features = 64 + block_channel[3] // 32
        self.Refine = R(block_channel)
        self.F_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features+num_features,
                      out_channels=num_features,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )
        self.I_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + num_features,
                      out_channels=num_features,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )
        self.C_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + num_features,
                      out_channels=num_features,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Tanh()
        )
        self.Q_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + num_features,
                      out_channels=num_features,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )

    def forward(self, input_tensor, b, d):
        input_tensor = maps_2_cubes(input_tensor, b, d)
        b, c, d, h, w = input_tensor.shape
        h_state_init = torch.zeros(b, c, h, w).to('cuda')
        c_state_init = torch.zeros(b, c, h, w).to('cuda')

        seq_len = d

        h_state, c_state = h_state_init, c_state_init
        output_inner = []
        for t in range(seq_len):
            input_cat = torch.cat((input_tensor[:,:,t,:,:], h_state), dim=1)
            c_state = self.F_t(input_cat) * c_state + self.I_t(input_cat) * self.C_t(input_cat)

            h_state, p_depth = self.Refine(c_state * self.Q_t(input_cat))

            output_inner.append(p_depth)

        layer_output = torch.stack(output_inner, dim=2)

        return layer_output


class R_CLSTM_2(nn.Module):
    def __init__(self, block_channel):
        super(R_CLSTM_2, self).__init__()
        num_features = 64 + block_channel[3] // 32
        self.Refine = R(block_channel)
        self.F_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + num_features,
                      out_channels=num_features // 2,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )
        self.I_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + num_features,
                      out_channels=num_features // 2,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )
        self.C_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + num_features,
                      out_channels=num_features // 2,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Tanh()
        )
        self.Q_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + num_features,
                      out_channels=num_features // 2,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )

    def forward(self, input_tensor, b, d):
        input_tensor = maps_2_cubes(input_tensor, b, d)
        b, c, d, h, w = input_tensor.shape
        h_state_init = torch.zeros(b, c, h, w).to('cuda')
        c_state_init = torch.zeros(b, c // 2, h, w).to('cuda')

        seq_len = d

        h_state, c_state = h_state_init, c_state_init
        output_inner = []
        for t in range(seq_len):
            input_cat = torch.cat((input_tensor[:, :, t, :, :], h_state), dim=1)
            c_state = self.F_t(input_cat) * c_state + self.I_t(input_cat) * self.C_t(input_cat)

            h_state, p_depth = self.Refine(torch.cat((c_state, self.Q_t(input_cat)), 1))

            output_inner.append(p_depth)

        layer_output = torch.stack(output_inner, dim=2)

        return layer_output


class R_CLSTM_3(nn.Module):
    def __init__(self, block_channel):
        super(R_CLSTM_3, self).__init__()
        num_features = 64 + block_channel[3] // 32
        self.Refine = R_2(block_channel)
        self.F_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features+4,
                      out_channels=4,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )
        self.I_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + 4,
                      out_channels=4,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )
        self.C_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + 4,
                      out_channels=4,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Tanh()
        )
        self.Q_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + 4,
                      out_channels=num_features,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )

    def forward(self, input_tensor, b, d):
        input_tensor = maps_2_cubes(input_tensor, b, d)
        b, c, d, h, w = input_tensor.shape
        h_state_init = torch.zeros(b, 4, h, w).to('cuda')
        c_state_init = torch.zeros(b, 4, h, w).to('cuda')

        seq_len = d

        h_state, c_state = h_state_init, c_state_init
        output_inner = []
        for t in range(seq_len):
            input_cat = torch.cat((input_tensor[:,:,t,:,:], h_state), dim=1)
            c_state = self.F_t(input_cat) * c_state + self.I_t(input_cat) * self.C_t(input_cat)

            h_state, p_depth = self.Refine(torch.cat((c_state, self.Q_t(input_cat)), 1))

            output_inner.append(p_depth)

        layer_output = torch.stack(output_inner, dim=2)

        return layer_output


class R_CLSTM_4(nn.Module):
    def __init__(self, block_channel):
        super(R_CLSTM_4, self).__init__()
        num_features = 64 + block_channel[3] // 32
        self.Refine = R_d(block_channel)
        self.F_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features+4,
                      out_channels=4,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )
        self.I_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + 4,
                      out_channels=4,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )
        self.C_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + 4,
                      out_channels=4,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Tanh()
        )
        self.Q_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + 4,
                      out_channels=num_features,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )

    def forward(self, input_tensor, b, d):
        input_tensor = maps_2_cubes(input_tensor, b, d)
        b, c, d, h, w = input_tensor.shape
        h_state_init = torch.zeros(b, 4, h, w).to('cuda')
        c_state_init = torch.zeros(b, 4, h, w).to('cuda')

        seq_len = d

        h_state, c_state = h_state_init, c_state_init
        output_inner = []
        for t in range(seq_len):
            input_cat = torch.cat((input_tensor[:,:,t,:,:], h_state), dim=1)
            c_state = self.F_t(input_cat) * c_state + self.I_t(input_cat) * self.C_t(input_cat)

            h_state, p_depth = self.Refine(torch.cat((c_state, self.Q_t(input_cat)), 1))

            output_inner.append(p_depth)

        layer_output = torch.stack(output_inner, dim=2)

        return layer_output


class R_CLSTM_5(nn.Module):
    def __init__(self, block_channel):
        super(R_CLSTM_5, self).__init__()
        num_features = 64 + block_channel[3] // 32
        self.Refine = R_3(block_channel)
        self.F_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features+8,
                      out_channels=8,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )
        self.I_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + 8,
                      out_channels=8,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )
        self.C_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + 8,
                      out_channels=8,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Tanh()
        )
        self.Q_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + 8,
                      out_channels=num_features,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )

    def forward(self, input_tensor, b, d):
        input_tensor = maps_2_cubes(input_tensor, b, d)
        b, c, d, h, w = input_tensor.shape
        h_state_init = torch.zeros(b, 8, h, w).to('cuda')
        c_state_init = torch.zeros(b, 8, h, w).to('cuda')

        seq_len = d

        h_state, c_state = h_state_init, c_state_init
        output_inner = []
        start = time.time()
        for t in range(seq_len):
            input_cat = torch.cat((input_tensor[:,:,t,:,:], h_state), dim=1)
            c_state = self.F_t(input_cat) * c_state + self.I_t(input_cat) * self.C_t(input_cat)

            h_state, p_depth = self.Refine(torch.cat((c_state, self.Q_t(input_cat)), 1))

            output_inner.append(p_depth)

        layer_output = torch.stack(output_inner, dim=2)
        torch.cuda.synchronize()
        # print(time.time()-start)

        return layer_output


class R_cell(nn.Module):
    def __init__(self, block_channel, cell_width=16):
        super(R_cell, self).__init__()

        num_features = 64 + block_channel[3] // 32
        self.conv0 = nn.Conv2d(num_features + cell_width, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features)

        self.conv1 = nn.Conv2d(num_features, num_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features)

        self.conv2 = nn.Conv2d(
            num_features, 1, kernel_size=5, stride=1, padding=2, bias=True)

        self.convh = nn.Conv2d(
            num_features, cell_width, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.bn0(x0)
        x0 = F.relu(x0)

        x1 = self.conv1(x0)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)

        h = self.convh(x1)
        pred_depth = self.conv2(x1)

        return h, pred_depth


class R_CLSTM_6(nn.Module):
    def __init__(self, block_channel):
        super(R_CLSTM_6, self).__init__()
        num_features = 64 + block_channel[3] // 32
        self.cell_width = 8
        self.Refine = R_cell(block_channel, self.cell_width)
        self.F_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + self.cell_width,
                      out_channels=self.cell_width,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )
        self.I_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + self.cell_width,
                      out_channels=self.cell_width,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )
        self.C_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + self.cell_width,
                      out_channels=self.cell_width,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Tanh()
        )
        self.Q_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + self.cell_width,
                      out_channels=num_features,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )

    def forward(self, input_tensor, b, d):
        input_tensor = maps_2_cubes(input_tensor, b, d)
        b, c, d, h, w = input_tensor.shape
        h_state_init = torch.zeros(b, self.cell_width, h, w).to('cuda')
        c_state_init = torch.zeros(b, self.cell_width, h, w).to('cuda')

        seq_len = d

        h_state, c_state = h_state_init, c_state_init
        output_inner = []
        for t in range(seq_len):
            input_cat = torch.cat((input_tensor[:, :, t, :, :], h_state), dim=1)
            c_state = self.F_t(input_cat) * c_state + self.I_t(input_cat) * self.C_t(input_cat)

            h_state, p_depth = self.Refine(torch.cat((c_state, self.Q_t(input_cat)), 1))

            output_inner.append(p_depth)

        layer_output = torch.stack(output_inner, dim=2)

        return layer_output


class R_CLSTM_7(nn.Module):
    def __init__(self, block_channel):
        super(R_CLSTM_7, self).__init__()
        num_features = 64 + block_channel[3] // 32
        self.cell_width = 12
        self.Refine = R_cell(block_channel, self.cell_width)
        self.F_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + self.cell_width,
                      out_channels=self.cell_width,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )
        self.I_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + self.cell_width,
                      out_channels=self.cell_width,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )
        self.C_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + self.cell_width,
                      out_channels=self.cell_width,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Tanh()
        )
        self.Q_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + self.cell_width,
                      out_channels=num_features,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )

    def forward(self, input_tensor, b, d):
        input_tensor = maps_2_cubes(input_tensor, b, d)
        b, c, d, h, w = input_tensor.shape
        h_state_init = torch.zeros(b, self.cell_width, h, w).to('cuda')
        c_state_init = torch.zeros(b, self.cell_width, h, w).to('cuda')

        seq_len = d

        h_state, c_state = h_state_init, c_state_init
        output_inner = []
        for t in range(seq_len):
            input_cat = torch.cat((input_tensor[:, :, t, :, :], h_state), dim=1)
            c_state = self.F_t(input_cat) * c_state + self.I_t(input_cat) * self.C_t(input_cat)

            h_state, p_depth = self.Refine(torch.cat((c_state, self.Q_t(input_cat)), 1))

            output_inner.append(p_depth)

        layer_output = torch.stack(output_inner, dim=2)

        return layer_output


class R_CLSTM_8(nn.Module):
    def __init__(self, block_channel):
        super(R_CLSTM_8, self).__init__()
        num_features = 64 + block_channel[3] // 32
        self.cell_width = 16
        self.Refine = R_cell(block_channel, self.cell_width)
        self.F_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + self.cell_width,
                      out_channels=self.cell_width,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )
        self.I_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + self.cell_width,
                      out_channels=self.cell_width,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )
        self.C_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + self.cell_width,
                      out_channels=self.cell_width,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Tanh()
        )
        self.Q_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + self.cell_width,
                      out_channels=num_features,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )

    def forward(self, input_tensor, b, d):
        input_tensor = maps_2_cubes(input_tensor, b, d)
        b, c, d, h, w = input_tensor.shape
        h_state_init = torch.zeros(b, self.cell_width, h, w).to('cuda')
        c_state_init = torch.zeros(b, self.cell_width, h, w).to('cuda')

        seq_len = d

        h_state, c_state = h_state_init, c_state_init
        output_inner = []
        for t in range(seq_len):
            input_cat = torch.cat((input_tensor[:, :, t, :, :], h_state), dim=1)
            c_state = self.F_t(input_cat) * c_state + self.I_t(input_cat) * self.C_t(input_cat)

            h_state, p_depth = self.Refine(torch.cat((c_state, self.Q_t(input_cat)), 1))

            output_inner.append(p_depth)

        layer_output = torch.stack(output_inner, dim=2)

        return layer_output


class R_CLSTM_9(nn.Module):
    def __init__(self, block_channel):
        super(R_CLSTM_9, self).__init__()
        num_features = 64 + block_channel[3] // 32
        self.cell_width = 32
        self.Refine = R_cell(block_channel, self.cell_width)
        self.F_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + self.cell_width,
                      out_channels=self.cell_width,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )
        self.I_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + self.cell_width,
                      out_channels=self.cell_width,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )
        self.C_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + self.cell_width,
                      out_channels=self.cell_width,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Tanh()
        )
        self.Q_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + self.cell_width,
                      out_channels=num_features,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )

    def forward(self, input_tensor, b, d):
        input_tensor = maps_2_cubes(input_tensor, b, d)
        b, c, d, h, w = input_tensor.shape
        h_state_init = torch.zeros(b, self.cell_width, h, w).to('cuda')
        c_state_init = torch.zeros(b, self.cell_width, h, w).to('cuda')

        seq_len = d

        h_state, c_state = h_state_init, c_state_init
        output_inner = []
        for t in range(seq_len):
            input_cat = torch.cat((input_tensor[:, :, t, :, :], h_state), dim=1)
            c_state = self.F_t(input_cat) * c_state + self.I_t(input_cat) * self.C_t(input_cat)

            h_state, p_depth = self.Refine(torch.cat((c_state, self.Q_t(input_cat)), 1))

            output_inner.append(p_depth)

        layer_output = torch.stack(output_inner, dim=2)

        return layer_output


class R_CLSTM_10(nn.Module):
    def __init__(self, block_channel):
        super(R_CLSTM_10, self).__init__()
        num_features = 64 + block_channel[3] // 32
        self.Refine = R_10(block_channel)
        self.F_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features+8,
                      out_channels=8,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )
        self.I_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + 8,
                      out_channels=8,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )
        self.C_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + 8,
                      out_channels=8,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Tanh()
        )
        self.Q_t = nn.Sequential(
            nn.Conv2d(in_channels=num_features + 8,
                      out_channels=num_features,
                      kernel_size=3,
                      padding=1,
                      ),
            nn.Sigmoid()
        )

    def forward(self, input_tensor, b, d):
        input_tensor = maps_2_cubes(input_tensor, b, d)
        b, c, d, h, w = input_tensor.shape
        h_state_init = torch.zeros(b, 8, h, w).to('cuda')
        c_state_init = torch.zeros(b, 8, h, w).to('cuda')

        seq_len = d

        h_state, c_state = h_state_init, c_state_init
        output_inner = []
        for t in range(seq_len):
            input_cat = torch.cat((input_tensor[:,:,t,:,:], h_state), dim=1)
            c_state = self.F_t(input_cat) * c_state + self.I_t(input_cat) * self.C_t(input_cat)

            h_state, p_depth = self.Refine(torch.cat((c_state, self.Q_t(input_cat)), 1))

            output_inner.append(p_depth)

        layer_output = torch.stack(output_inner, dim=2)

        return layer_output


