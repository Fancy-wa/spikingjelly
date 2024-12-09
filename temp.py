import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import visdom

from spikingjelly.activation_based import functional, encoding, neuron, layer
import ann
import snn
# from ann import LeNet, ResNet


np.set_printoptions(threshold=float('inf'), precision=4)  	# 设置 NumPy 打印选项



def map_range(x, new_min, new_max):
    return new_min + (new_max - new_min) * x


class LSTM_Test(nn.Module):
    def __init__(self):
        super(LSTM_Test, self).__init__()
        # 定义LSTM层
        self.lstm = nn.LSTM(input_size=3, hidden_size=3, num_layers=1, bias=False, batch_first=True)
        self.lif = neuron.LIFNode(v_threshold=0.5)
        self.linear = nn.Linear(in_features=3, out_features=4, bias=False)
        self.squeeze_out = None
        self.lstm_out = None
        self.gather_out = None
        self.lifnode_out = None

    def forward(self, x):
        self.input = x
        self.input.retain_grad()
        self.squeeze_out = self.input.squeeze(1)
        self.squeeze_out.retain_grad()
        self.lstm_out, _ = self.lstm(self.squeeze_out)
        self.lstm_out.retain_grad()
        self.gather_out = self.lstm_out[:, -1, :]
        self.gather_out.retain_grad()
        self.lifnode_out = self.lif(self.gather_out)
        self.lifnode_out.retain_grad()
        self.output = self.linear(self.lifnode_out)
        self.output.retain_grad()
        return self.output


class Linear_Test(nn.Module):
    def __init__(self):
        super(Linear_Test, self).__init__()
        # 定义LSTM层
        self.flatten = layer.Flatten()
        self.linear = layer.Linear(in_features=9, out_features=4, bias=False)
        # self.reshape2 = nn.
        self.lifnode = neuron.LIFNode()
        self.input = None
        self.temp1 = None
        self.temp2 = None
        self.temp3 = None
        self.output = None

    def forward(self, x):
        self.input = x
        self.input.retain_grad()  # 因为所有时间步来自同一个x，所以是所有时间步的梯度
        # 前向传播Flatten层
        self.temp1 = self.flatten(self.input)
        self.temp1.retain_grad()  # 最后一个时间步的梯度
        # 前向传播Linear层
        self.temp2 = self.linear(self.temp1)
        self.temp2.retain_grad()  # 最后一个时间步的梯度
        self.temp3 = self.temp2.reshape(2,4)
        self.temp3.retain_grad()  # 最后一个时间步的梯度
        # self.output = self.lifnode(self.temp3)
        self.output = self.temp3
        self.output.retain_grad()  # 最后一个时间步的梯度
        return self.output


class Conv2d_Test(nn.Module):
    def __init__(self):
        super(Conv2d_Test, self).__init__()
        # 定义LSTM层
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=9, out_features=4, bias=False)
        self.input = None
        self.temp1 = None
        self.temp2 = None
        self.output = None

    def forward(self, x):
        self.input = x
        self.input.retain_grad()
        # 前向传播Conv2d层
        self.temp1 = self.conv2d(self.input)
        self.temp1.retain_grad()
        # 前向传播Flatten层
        self.temp2 = self.flatten(self.temp1)
        self.temp2.retain_grad()
        # 前向传播Linear层
        self.output = self.linear(self.temp2)
        self.output.retain_grad()
        return self.output


def lstm_train(input_data, label):
    # 实例化 LSTM 网络
    net = LSTM_Test()

    for _ in range(1):

        net.zero_grad()

        # 将Linear层的权重设置为全1
        for name, param in net.linear.named_parameters():
            param.data = torch.ones_like(param.data)
            param.data[3,2] = 0

        # 将LSTM层的权重设置为全1
        for name, param in net.lstm.named_parameters():
            param.data = torch.ones_like(param.data)


        pred = 0.
        for _ in range(10):
            pred += net(input_data)
        pred = pred / 10.0

        # pred = net(input_data)
        # pred = net(input_data)

        pred = map_range(pred, -3, 3)

        loss = F.cross_entropy(pred, label)
        loss.backward(retain_graph=True)
        with torch.no_grad(): [param.sub_(0.1 * param.grad) for param in net.parameters()]  # 权重更新
        functional.reset_net(net)  # 重置神经元




    # net.zero_grad()

    # # 将Linear层的权重设置为全1
    # for name, param in net.linear.named_parameters():
    #     param.data = torch.ones_like(param.data)
    #     param.data[3,2] = 0

    # # 将LSTM层的权重设置为全1
    # for name, param in net.lstm.named_parameters():
    #     param.data = torch.ones_like(param.data)


    # pred = 0.
    # for _ in range(10):
    #     pred += net(input_data)
    # pred = pred / 10.0

    # # pred = net(input_data)
    # # pred = net(input_data)

    # pred = map_range(pred, -3, 3)

    # loss = F.cross_entropy(pred, label)
    # loss.backward(retain_graph=True)
    # with torch.no_grad(): [param.sub_(0.1 * param.grad) for param in net.parameters()]  # 权重更新
    # functional.reset_net(net)  # 重置神经元


    # 打印最后一个时间步网络的输入输出grad
    print("----------------------- 输入数据 -----------------------")
    print('输入值:\n', net.input)
    print('输入梯度:\n', net.input.grad)
    print("----------------------- squeeze层 -----------------------")
    print('squeeze输出值:\n', net.squeeze_out)
    print('squeeze输入梯度:\n', net.squeeze_out.grad)
    print("----------------------- lstm层 -----------------------")
    print('lstm输出值:\n', net.lstm_out)
    print('lstm输入梯度:\n', net.lstm_out.grad)
    # 打印Linear层的权重和偏置
    for name, param in net.lstm.named_parameters():
        print('lstm权重偏置值:\n', name, param)
        print('lstm权重偏置梯度:\n', name, param.grad)
    print("----------------------- gather层 -----------------------")
    print('gather输出值:\n', net.gather_out)
    print('gather输入梯度:\n', net.gather_out.grad)
    print("----------------------- lifnode层 -----------------------")
    print('lifnode输出值:\n', net.lifnode_out)
    print('lifnode输入梯度:\n', net.lifnode_out.grad)
    print("----------------------- linear层 -----------------------")
    print('linear输出值:\n', net.output)
    print('linear输入梯度:\n', net.output.grad)
    # 打印Linear层的权重和偏置
    for name, param in net.linear.named_parameters():
        print('linear权重偏置值:\n', name, param)
        print('linear权重偏置梯度:\n', name, param.grad)
    # 打印预测值
    print("----------------------- 输出数据 -----------------------")
    print('预测值:\n', pred)


def linear_train(input_data, label):
    # 实例化 Linear 网络
    net = Linear_Test()
    net.zero_grad()

    # 将Linear层的权重设置为全1
    for name, param in net.linear.named_parameters():
        param.data = torch.ones_like(param.data)
        param.data[3,8] = 0

    pred = 0.
    for _ in range(10):
        pred += net(input_data)
    pred = pred / 10.0

    # pred = net(input_data)
    # pred = net(input_data)

    pred = map_range(pred, -3, 3)

    loss = F.cross_entropy(pred, label)
    loss.backward(retain_graph=True)

    # 打印最后一个时间步网络的输入输出grad
    print("----------------------- 输入数据 -----------------------")
    print('输入值:\n', net.input)
    print('输入梯度:\n', net.input.grad)
    print("----------------------- flatten层 -----------------------")
    print('flatten输出值:\n', net.temp1)
    print('flatten输入梯度:\n', net.temp1.grad)
    print("----------------------- linear层 -----------------------")
    print('linear输出值:\n', net.temp2)
    print('linear输入梯度:\n', net.temp2.grad)
    # 打印Linear层的权重和偏置
    for name, param in net.linear.named_parameters():
        print('权重偏置值:\n', name, param)
        print('权重偏置梯度:\n', name, param.grad)
    print("----------------------- reshape层 -----------------------")
    print('reshape输出值:\n', net.temp3)
    print('reshape输入梯度:\n', net.temp3.grad)
    # print("----------------------- lif_node层 -----------------------")
    # print('lifnode输出值:\n', net.output)
    # print('lifnode输出梯度:\n', net.output.grad)
    # 打印预测值
    print("----------------------- 输出数据 -----------------------")
    print('预测值:\n', pred)


def conv2d_train(input_data, label):
    # 实例化 Conv2d 网络
    net = Conv2d_Test()
    net.zero_grad()

    # 将Conv2d层的权重设置为全1
    for name, param in net.conv2d.named_parameters():
        param.data = torch.ones_like(param.data)

    # 将Linear层的权重设置为全1
    for name, param in net.linear.named_parameters():
        param.data = torch.ones_like(param.data)
        param.data[3,8] = 0

    pred = net(input_data)
    loss = F.cross_entropy(pred, label)
    loss.backward(retain_graph=True)

    # 打印网络的输入输出grad
    print('输入值:\n', net.input)
    print('输入梯度:\n', net.input.grad)
    # print('中间1值:\n', net.temp1)
    # print('中间1梯度:\n', net.temp1.grad)
    # print('中间2值:\n', net.temp2)
    # print('中间2梯度:\n', net.temp2.grad)
    print('输出值:\n', net.output)
    print('输出梯度:\n', net.output.grad)

    # 打印Conv2d层的权重和偏置
    for name, param in net.conv2d.named_parameters():
        print('卷积权重偏置值:\n', name, param)
        print('卷积权重偏置梯度:\n', name, param.grad)

    # 打印Linear层的权重和偏置
    for name, param in net.linear.named_parameters():
        print('线性权重偏置值:\n', name, param)
        print('线性权重偏置梯度:\n', name, param.grad)



if __name__ == '__main__':
    # input_data = torch.tensor( [[[1, 1, 0],
    #                              [0, 1, 0],
    #                              [0, 1, 1]],
    #                             [[1, 1, 0],
    #                              [0, 1, 0],
    #                              [0, 1, 1]]] ).float().requires_grad_(True)
    # label = torch.tensor([[0, 0, 0, 1], [1, 0, 0, 0]]).float().requires_grad_(True)
    # input_data = torch.tensor( [[[0, 0, 0],
    #                              [0, 1, 0],
    #                              [0, 0, 1]],
    #                             [[0, 0, 0],
    #                              [0, 1, 0],
    #                              [0, 0, 1]],
    #                             [[0, 0, 0],
    #                              [0, 1, 0],
    #                              [0, 0, 1]]] ).float().requires_grad_(True)
    # label = torch.tensor([[0, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 1]]).float().requires_grad_(True)
    # input_data = torch.tensor( [[[0.1, 0.0, 0.0],
    #                              [0.0, 0.5, 0.0],
    #                              [0.0, 0.0, 0.5]]] ).float().requires_grad_(True)
    # label = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).float().requires_grad_(True)
    # input_data = torch.tensor( [[[0.1, 0.0, 0.0],
    #                              [0.0, 0.5, 0.0],
    #                              [0.0, 0.0, 0.5]],
    #                             [[0.1, 0.0, 0.0],
    #                              [0.0, 0.5, 0.0],
    #                              [0.0, 0.0, 0.5]]] ).float().requires_grad_(True)
    input_data = torch.tensor([[[[0.1, 0.0, 0.0],
                                 [0.0, 0.5, 0.0],
                                 [0.0, 0.0, 0.5]]],
                               [[[0.1, 0.0, 0.0],
                                 [0.0, 0.5, 0.0],
                                 [0.0, 0.0, 0.5]]]] ).float().requires_grad_(True)
    label = torch.tensor([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]]).float().requires_grad_(True)
    # input_data = torch.tensor([   [[[1, 0, 1, 0, 1],
    #                                 [0, 1, 0, 1, 0],
    #                                 [1, 1, 1, 1, 1],
    #                                 [0, 0, 0, 0, 0],
    #                                 [1, 1, 0, 0, 1]]],
    #                               [[[1, 0, 1, 0, 1],
    #                                 [0, 1, 0, 1, 0],
    #                                 [1, 1, 1, 1, 1],
    #                                 [0, 0, 0, 0, 0],
    #                                 [1, 1, 0, 0, 1]]]  ]).float().requires_grad_(True)
    # label = torch.tensor([[0, 0, 0, 1], [1, 0, 0, 0]]).float().requires_grad_(True)
    # input_data = torch.tensor([[[0, 0, 0],
    #                             [0, 1, 0],
    #                             [0, 0, 1]],
    #                            [[0, 0, 0],
    #                             [0, 1, 0],
    #                             [0, 0, 1]]]).float().requires_grad_(True)
    # label = torch.tensor([[0, 0, 0, 1], [1, 0, 0, 0]]).float().requires_grad_(True)

    lstm_train(input_data, label)
    # linear_train(input_data, label)
    # conv2d_train(input_data, label)



    # # 手动单步
    # net_s = neuron.IFNode(step_mode='s')
    # T = 4
    # N = 1
    # C = 3
    # H = 8
    # W = 8
    # x_seq = torch.rand([T, N, C, H, W])
    # y_seq = []
    # for t in range(T):
    #     x = x_seq[t]  # x.shape = [N, C, H, W]
    #     y = net_s(x)  # y.shape = [N, C, H, W]
    #     y_seq.append(y.unsqueeze(0))

    # y_seq = torch.cat(y_seq)
    # print(y_seq.shape)
    # # y_seq.shape = [T, N, C, H, W]



    # # 自动单步（functional.multi_step_forward）
    # net_s = neuron.IFNode(step_mode='s')
    # T = 4
    # N = 1
    # C = 3
    # H = 8
    # W = 8
    # x_seq = torch.rand([T, N, C, H, W])
    # y_seq = functional.multi_step_forward(x_seq, net_s)
    # print(y_seq.shape)
    # # y_seq.shape = [T, N, C, H, W]



    # # 自动多步
    # net_m = neuron.IFNode(step_mode='m')
    # T = 4
    # N = 1
    # C = 3
    # H = 8
    # W = 8
    # x_seq = torch.rand([T, N, C, H, W])
    # y_seq = net_m(x_seq)
    # print(y_seq.shape)
    # # y_seq.shape = [T, N, C, H, W]

