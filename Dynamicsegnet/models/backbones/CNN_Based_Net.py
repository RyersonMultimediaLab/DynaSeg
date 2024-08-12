import torch.nn as nn
import torch.nn.functional as F


class MyNet(nn.Module):

    def __init__(self, input_dim):
        super(MyNet, self).__init__()

        self.input_dim = input_dim
        self.output_dim = 100
        self.dilation = 1
        self.conv1 = nn.Conv2d(self.input_dim, self.output_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.output_dim)
        self.conv2 = nn.Conv2d(self.output_dim, self.output_dim, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.ModuleList()
        # for i in range(2):
        #     self.conv2.append(nn.Conv2d(self.output_dim, self.output_dim, kernel_size=3, stride=1, padding=1))
        #     self.bn2.append(nn.BatchNorm2d(self.output_dim))

        self.conv3 = nn.Conv2d(self.output_dim, self.output_dim, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(self.output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x.size()
        x = F.relu(x)
        print(x.size())
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)
        # for i in range(2 - 1):
        #     x = self.conv2[i](x)
        #     x = F.relu(x)
        #     x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)

        return x

    def mynet(arch):
        model = MyNet(3)
        return model


