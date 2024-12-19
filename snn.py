import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer



class LeNet(nn.Module):
    """
    从ANN的LeNet修改的SNN的LeNet网络模型
    """
    def __init__(self, tau=2.0, is_train=False):
        super().__init__()
        self.is_train = is_train
        
        self.conv1 = layer.Conv2d(in_channels=1, out_channels=6, kernel_size=5, bias=False)
        self.lifnode1 = neuron.LIFNode(v_threshold=0.1)
        self.maxpool1 = layer.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = layer.Conv2d(in_channels=6, out_channels=16, kernel_size=5, bias=False)
        self.lifnode2 = neuron.LIFNode(v_threshold=0.1)
        self.maxpool2 = layer.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = layer.Flatten()
        self.fc1 = layer.Linear(in_features=16*4*4, out_features=120, bias=False)  # 4*4 是经过两次池化后的特征图大小
        self.lifnode3 = neuron.LIFNode(v_threshold=0.1)
        self.fc2 = layer.Linear(in_features=120, out_features=84, bias=False)
        self.lifnode4 = neuron.LIFNode(v_threshold=0.1)
        self.fc3 = layer.Linear(in_features=84, out_features=10, bias=False)

    def forward(self, input_tensor):
        self.input = input_tensor
        if (self.is_train):
            self.input.retain_grad()
        x = self.conv1(self.input)                  # [N, 1, 28, 28] -> [N, 6, 24, 24]
        x = self.lifnode1(x)                        # [N, 6, 24, 24] -> [N, 6, 24, 24]
        x = self.maxpool1(x)                        # [N, 6, 24, 24] -> [N, 6, 12, 12]
        x = self.conv2(x)                           # [N, 6, 12, 12] -> [N, 16, 8, 8]
        x = self.lifnode2(x)                        # [N, 16, 8, 8] -> [N, 16, 8, 8]
        x = self.maxpool2(x)                        # [N, 16, 8, 8] -> [N, 16, 4, 4]
        x = self.flatten(x)                         # [N, 16, 4, 4] -> [N, 16*4*4]
        x = self.fc1(x)                             # [N, 256] -> [N, 120]
        x = self.lifnode3(x)                        # [N, 120] -> [N, 120]
        x = self.fc2(x)                             # [N, 120] -> [N, 84]
        x = self.lifnode4(x)                        # [N, 84] -> [N, 84]
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
        self.conv1 = layer.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False)
        self.bn1 = layer.BatchNorm2d(out_channel)
        self.neuron1 = neuron.LIFNode()
        self.conv2 = layer.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)
        self.bn2 = layer.BatchNorm2d(out_channel)
        self.neuron2 = neuron.LIFNode()
        self.downsample = downsample

    def forward(self, input_tensor):
        self.input = input_tensor
        identity = self.input
        if self.downsample is not None:
            identity = self.downsample(self.input)
        x = self.conv1(self.input)
        x = self.bn1(x)
        x = self.neuron1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        self.output = self.neuron2(x)
        return self.output


class ResNet1(nn.Module):
    """
    从经典ANN的ResNet转换的SNN的ResNet网络模型
    """
    def __init__(self, block=_BasicBlock, blocks_num=[2,2,2,2], num_classes=10, is_train=False):
        super(ResNet1, self).__init__()
        self.is_train = is_train
        self.in_channel = 64

        # self.conv1 = layer.Conv2d(1, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = layer.Conv2d(1, self.in_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = layer.BatchNorm2d(self.in_channel)
        self.neuron1 = neuron.LIFNode()
        self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.avgpool = layer.AvgPool2d(kernel_size=14, stride=1)
        self.flatten = layer.Flatten()
        self.fc = layer.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, channel, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                layer.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                layer.BatchNorm2d(channel * block.expansion)
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
        x = self.neuron1(x)                         # [N, 64, 28, 28] -> [N, 64, 28, 28]
        x = self.maxpool(x)                         # [N, 64, 28, 28] -> [N, 64, 14, 14]
        x = self.layer1(x)                          # [N, 64, 14, 14] -> [N, 64, 14, 14]
        x = self.avgpool(x)                         # [N, 64, 14, 14] -> [N, 64, 1, 1]
        x = self.flatten(x)                         # [N, 64, 1, 1] -> [N, 64*1*1]
        self.output = self.fc(x)                    # [N, 64] -> [N, 10]
        if (self.is_train):
            self.output.retain_grad()
        return self.output


class ResNet2(nn.Module):
    """
    从经典ANN的ResNet转换的SNN的ResNet网络模型(用二层残差块，效果好一点)
    """
    def __init__(self, block=_BasicBlock, blocks_num=[2,2,2,2], num_classes=10, is_train=False):
        super(ResNet2, self).__init__()
        self.is_train = is_train
        self.in_channel = 64

        # self.conv1 = layer.Conv2d(1, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = layer.Conv2d(1, self.in_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = layer.BatchNorm2d(self.in_channel)
        self.neuron1 = neuron.LIFNode()
        self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        self.avgpool = layer.AvgPool2d(kernel_size=7, stride=1)
        self.flatten = layer.Flatten()
        self.fc = layer.Linear(128 * block.expansion, num_classes)

    def _make_layer(self, block, channel, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                layer.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                layer.BatchNorm2d(channel * block.expansion)
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
        x = self.neuron1(x)                         # [N, 64, 28, 28] -> [N, 64, 28, 28]
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



class BasicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, tau=2.0, v_threshold=0.1, v_reset=0.0):
        """
        ANN LSTM 模块
        :param input_size: 输入维度
        :param hidden_size: 隐藏层维度
        :param num_layers: 层数
        :param need_bias: 是否需要偏置
        :param batch_first: 输入数据格式是否 batch 在最前
        """
        super(BasicLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        # 门控机制的线性变换
        self.input_to_gate = nn.Linear(in_features=input_size, out_features=4*hidden_size, bias=bias)
        self.hidden_to_gate = nn.Linear(in_features=hidden_size, out_features=4*hidden_size, bias=bias)
        self.i_neuron = neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset)
        self.f_neuron = neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset)
        self.o_neuron = neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset)
        self.cs_neuron = neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset)
        self.ct_neuron = neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset)

    def forward(self, inputs):
        """
        :param inputs: [T, N, input_size] 输入序列
        :return: 输出和隐藏状态
        """
        if self.batch_first:
            inputs = inputs.transpose(0, 1)

        T, N, _ = inputs.size()

        # 初始化隐藏状态和细胞状态
        h_t = torch.zeros(N, self.hidden_size, device=inputs.device)
        c_t = torch.zeros(N, self.hidden_size, device=inputs.device)

        outputs = []
        for t in range(T):
            x_t = inputs[t]  # 提取当前时间步的输入

            gates = self.input_to_gate(x_t) + self.hidden_to_gate(h_t)
            i_gate, o_gate, f_gate, c_state = torch.split(gates, self.hidden_size, dim=1)
            c_t = self.f_neuron(f_gate) * c_t + self.i_neuron(i_gate) * self.cs_neuron(c_state)
            h_t = self.o_neuron(o_gate) * self.ct_neuron(c_t)

            outputs.append(h_t)

        outputs = torch.stack(outputs, dim=0)  # [T, N, hidden_size]

        if self.batch_first:
            outputs = outputs.transpose(0, 1)

        return outputs, (h_t, c_t)


class SimpleLSTM(nn.Module):
    def __init__(self, is_train=False):
        super(SimpleLSTM, self).__init__()
        self.is_train = is_train

        self.lstm = BasicLSTM(input_size=28, hidden_size=128, num_layers=1, bias=True, batch_first=True)
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



class HybridLSTM(nn.Module):
    def __init__(self, is_train=False):
        super(HybridLSTM, self).__init__()
        self.is_train = is_train

        self.lstm = nn.LSTM(input_size=28, hidden_size=128, num_layers=1, bias=True, batch_first=True)  # ANN LSTM 部分
        self.lif = neuron.LIFNode(v_threshold=0.1)  # SNN LIFNode部分
        self.fc = nn.Linear(in_features=128, out_features=10)  # Linear部分

    def forward(self, input_tensor):
        self.input = input_tensor
        if (self.is_train):
            self.input.retain_grad()
        x = self.input.squeeze(1)                   # [N, 1, 28, 28] -> [N, 28, 28]
        x, _ = self.lstm(x)                         # [N, 28, 28] -> [N, 28, 128]
        x = x[:, -1, :]                             # [N, 28, 128] -> [N, 128]
        x = self.lif(x)                             # [N, 128] -> [N, 128]
        self.output = self.fc(x)                    # [N, 128] -> [N, 10]
        if (self.is_train):
            self.output.retain_grad()
        return self.output
