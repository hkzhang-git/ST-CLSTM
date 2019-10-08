import torch.nn as nn


class C_C2D_1(nn.Module):

    def __init__(self, num_classes=2, init_width=32, input_channels=1):
        self.inplanes = init_width
        super(C_C2D_1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, init_width, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(init_width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(init_width, init_width*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(init_width*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        init_width = init_width * 2
        self.conv3 = nn.Sequential(
            nn.Conv2d(init_width, init_width * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(init_width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        init_width = init_width * 2
        self.conv4 = nn.Sequential(
            nn.Conv2d(init_width, init_width * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(init_width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(init_width * 2, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x