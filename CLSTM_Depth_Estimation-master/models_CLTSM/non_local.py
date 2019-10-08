import torch
from torch import nn
from torch.nn import functional as F


class Non_local_d(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels):
        super(Non_local_d, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_channels = out_channels

        max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.g = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1),
            max_pool
        )

        self.theta = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=inter_channels,
                      kernel_size=3, stride=1, padding=1),
            max_pool
        )

        self.phi = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=inter_channels,
                      kernel_size=3, stride=1, padding=1),
            max_pool
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)


    def forward(self, pre_h, pre_d, cur_h):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = pre_h.size(0)
        h, w = pre_h.size(-2), pre_h.size(-1)

        pre_d_f = self.g(pre_d).view(batch_size, self.out_channels, -1)
        pre_d_f = pre_d_f.permute(0, 2, 1)

        theta_x = self.theta(pre_h).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(cur_h).view(batch_size, self.inter_channels, -1)

        depth_attetion = F.softmax(torch.matmul(theta_x, phi_x), dim=-1)

        y = torch.matmul(depth_attetion, pre_d_f)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.out_channels, h // 2, w // 2)
        y = F.upsample_bilinear(y, scale_factor=2)
        W_y = self.W(y)
        # z = torch.cat((W_y, pre_d), dim=1)

        return W_y


class Non_local_h(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels):
        super(Non_local_h, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_channels = out_channels

        max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.g = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1),
            max_pool
        )

        self.theta = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=inter_channels,
                      kernel_size=3, stride=1, padding=1),
            max_pool
        )

        self.phi = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=inter_channels,
                      kernel_size=3, stride=1, padding=1),
            max_pool
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)


    def forward(self, pre_h, pre_d, cur_h):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = pre_h.size(0)
        h, w = pre_h.size(-2), pre_h.size(-1)

        pre_d_f = self.g(pre_d).view(batch_size, self.out_channels, -1)
        pre_d_f = pre_d_f.permute(0, 2, 1)

        theta_x = self.theta(pre_h).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(cur_h).view(batch_size, self.inter_channels, -1)

        depth_attetion = F.softmax(torch.matmul(theta_x, phi_x), dim=-1)

        y = torch.matmul(depth_attetion, pre_d_f)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.out_channels, h // 2, w // 2)
        y = F.upsample_bilinear(y, scale_factor=2)
        W_y = self.W(y)
        # z = torch.cat((W_y, pre_d), dim=1)

        return W_y




if __name__ == '__main__':
    pre_h = torch.randn(10, 128, 114, 108)
    pre_d = torch.randn(10, 1, 114, 108)
    cur_h = torch.randn(10, 128, 114, 108)

    non_local = Non_local_h(128, 16, 8).cuda()
    pre_h, pre_d, cur_h = pre_h.to('cuda'), pre_d.to('cuda'), cur_h.to('cuda')

    out = non_local(pre_h.to('cuda'), pre_d.to('cuda'), cur_h.to('cuda'))









