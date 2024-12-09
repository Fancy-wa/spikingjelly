import torch.nn as nn



class LeNet(nn.Module):
    """
    经典ANN的LeNet网络模型
    """
    def __init__(self, is_train=False):
        super().__init__()
        self.is_train = is_train

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=16*4 *4, out_features=120)  # 4*4 是经过两次池化后的特征图大小
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, input_tensor):
        self.input = input_tensor
        if (self.is_train):
            self.input.retain_grad()
        x = self.conv1(self.input)                  # [N, 1, 28, 28] -> [N, 6, 24, 24]
        x = self.relu1(x)                           # [N, 6, 24, 24] -> [N, 6, 24, 24]
        x = self.maxpool1(x)                        # [N, 6, 24, 24] -> [N, 6, 12, 12]
        x = self.conv2(x)                           # [N, 6, 12, 12] -> [N, 16, 8, 8]
        x = self.relu2(x)                           # [N, 16, 8, 8] -> [N, 16, 8, 8]
        x = self.maxpool2(x)                        # [N, 16, 8, 8] -> [N, 16, 4, 4]
        x = self.flatten(x)                         # [N, 16, 4, 4] -> [N, 16*4*4]
        x = self.fc1(x)                             # [N, 256] -> [N, 120]
        x = self.relu3(x)                           # [N, 120] -> [N, 120]
        x = self.fc2(x)                             # [N, 120] -> [N, 84]
        x = self.relu4(x)                           # [N, 84] -> [N, 84]
        self.output = self.fc3(x)                   # [N, 84] -> [N, 10]
        if (self.is_train):
            self.output.retain_grad()
        return self.output



class _BasicBlock(nn.Module):
    """
    ResNet的基本残差块
    """
    expansion = 1
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(_BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        x = self.relu2(x)
        return x


class ResNet1(nn.Module):
    """
    经典ANN的ResNet网络模型(只用一层残差块，训练快一点)
    """
    def __init__(self, block=_BasicBlock, blocks_num=[2,2,2,2], num_classes=10, is_train=False):
        super(ResNet1, self).__init__()
        self.is_train = is_train
        self.in_channel = 64

        # self.conv1 = nn.Conv2d(1, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = nn.Conv2d(1, self.in_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.avgpool = nn.AvgPool2d(kernel_size=14, stride=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, channel, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channel, channel))
        return nn.Sequential(*layers)

    def forward(self, input_tensor):
        self.input = input_tensor
        if (self.is_train):
            self.input.retain_grad()
        x = self.conv1(self.input)                  # [N, 1, 28, 28] -> [N, 64, 28, 28]
        x = self.bn1(x)                             # [N, 64, 28, 28] -> [N, 64, 28, 28]
        x = self.relu(x)                            # [N, 64, 28, 28] -> [N, 64, 28, 28]
        x = self.maxpool(x)                         # [N, 64, 28, 28] -> [N, 64, 14, 14]
        x = self.layer1(x)                          # [N, 64, 14, 14] -> [N, 64, 14, 14]
        x = self.avgpool(x)                         # [N, 128, 14, 14] -> [N, 128, 1, 1]
        x = self.flatten(x)                         # [N, 128, 1, 1] -> [N, 128*1*1]
        self.output = self.fc(x)                    # [N, 128] -> [N, 10]
        if (self.is_train):
            self.output.retain_grad()
        return self.output


class ResNet2(nn.Module):
    """
    经典ANN的ResNet网络模型(用二层残差块，效果好一点)
    """
    def __init__(self, block=_BasicBlock, blocks_num=[2,2,2,2], num_classes=10, is_train=False):
        super(ResNet2, self).__init__()
        self.is_train = is_train
        self.in_channel = 64

        # self.conv1 = nn.Conv2d(1, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = nn.Conv2d(1, self.in_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * block.expansion, num_classes)

    def _make_layer(self, block, channel, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channel, channel))
        return nn.Sequential(*layers)

    def forward(self, input_tensor):
        self.input = input_tensor
        if (self.is_train):
            self.input.retain_grad()
        x = self.conv1(self.input)                  # [N, 1, 28, 28] -> [N, 64, 28, 28]
        x = self.bn1(x)                             # [N, 64, 28, 28] -> [N, 64, 28, 28]
        x = self.relu(x)                            # [N, 64, 28, 28] -> [N, 64, 28, 28]
        x = self.maxpool(x)                         # [N, 64, 28, 28] -> [N, 64, 14, 14]
        x = self.layer1(x)                          # [N, 64, 14, 14] -> [N, 64, 14, 14]
        x = self.layer2(x)                          # [N, 64, 14, 14] -> [N, 128, 7, 7]
        # x = self.layer3(x)
        # x = self.layer4(x)
        x = self.avgpool(x)                         # [N, 128, 7, 7] -> [N, 128, 1, 1]
        x = self.flatten(x)                         # [N, 128, 1, 1] -> [N, 128*1*1]
        self.output = self.fc(x)                    # [N, 128] -> [N, 10]
        if (self.is_train):
            self.output.retain_grad()
        return self.output



class SimpleLSTM(nn.Module):
    def __init__(self, is_train=False):
        super(SimpleLSTM, self).__init__()
        self.is_train = is_train

        self.lstm = nn.LSTM(input_size=28, hidden_size=128, num_layers=1, bias=True, batch_first=True)
        self.fc = nn.Linear(in_features=128, out_features=10)

    def forward(self, input_tensor):
        self.input = input_tensor
        if (self.is_train):
            self.input.retain_grad()
        x = self.input.squeeze(1)                   # [N, 1, 28, 28] -> [N, 28, 28]
        x, _ = self.lstm(x)                         # [N, 28, 28] -> [N, 28, 128]
        x = x[:, -1, :]                             # [N, 28, 128] -> [N, 128]
        x = self.fc(x)                              # [N, 128] -> [N, 10]
        self.output = x
        if (self.is_train):
            self.output.retain_grad()
        return self.output

