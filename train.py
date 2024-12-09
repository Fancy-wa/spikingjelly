import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import visdom

from spikingjelly.activation_based import functional, encoding
import ann
import snn



def visdom_plot_acc(vis, x_plot, y_plot, title: str, legend: str):
    if not hasattr(visdom_plot_acc, 'has_run'):
        visdom_plot_acc.win_id=vis.line(X=np.array([x_plot]),Y=np.array([y_plot]), opts=dict(title=title, xlabel='Batch', ylabel='Accuarcy', legend=[legend], showlegend=True, ytickvals=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]))
        visdom_plot_acc.has_run = True
    else:
        vis.line(X=np.array([x_plot]),Y=np.array([y_plot]),win=visdom_plot_acc.win_id,update='append')
  

def visdom_plot_loss(vis, x_plot, y_plot, title: str, legend: str):
    if not hasattr(visdom_plot_loss, 'has_run'):
        visdom_plot_loss.win_id=vis.line(X=np.array([x_plot]),Y=np.array([y_plot]), opts=dict(title=title, xlabel='Batch', ylabel='Loss', legend=[legend], showlegend=True))
        visdom_plot_loss.has_run = True
    else:
        vis.line(X=np.array([x_plot]),Y=np.array([y_plot]),win=visdom_plot_loss.win_id,update='append')


def map_range(x, new_min, new_max):
    return new_min + (new_max - new_min) * x



def save_model(args, net, accuracy, nn_type:str):
    # 保存模型
    net.eval()
    if not hasattr(save_model, "best_accuracy"):
        save_model.best_accuracy = 0  # 初始化 static 变量

    # 保存模型
    if accuracy > save_model.best_accuracy:
        save_model.best_accuracy = accuracy
        # 导出为pth模型
        torch.save(net, f'model/{nn_type}_{args.net_name}_Acc_{accuracy}.pt')
        # 导出为onnx模型
        dummy_input = torch.randn(1, 1, 28, 28).to(args.device)
        torch.onnx.export(net, 
                          dummy_input, 
                          f'model/{nn_type}_{args.net_name}_Acc_{accuracy}.onnx', 
                          input_names=["input"], 
                          output_names=["output"])
        print(f"Best model saved with accuracy: {save_model.best_accuracy:.2f}%")
        


def ann_mnist(args):
    # 可视化窗口
    vis = visdom.Visdom(env=args.vis_env)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5], std=[0.5])  # 假设单通道图像
    ])
    train_datasets = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_datasets, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)	
    test_datasets = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(dataset=test_datasets, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)
    #  定义网络、优化器、编码器
    net = getattr(ann, args.net_name)(is_train=(True if args.mode == 'visdom' else False)).to(args.device)
    # 开始训练
    net.train()
    for epoch in range(0, args.epochs):
        net.train()
        batch_loss = 0 							# batch的损失
        batch_acc = 0							# batch的准确率
        for iter, (data, label) in enumerate(train_loader):
            data, label = data.to(args.device).requires_grad_(), label.to(args.device)

            net.zero_grad()  # 梯度清零
            pred = net(data)  # 前向传播
            loss = F.cross_entropy(pred, label) # 损失函数
            loss.backward(retain_graph=True)  # 反向传播
            with torch.no_grad(): [param.sub_(args.lr * param.grad) for param in net.parameters()]  # 权重更新

            if (args.mode == 'visdom'):
                batch_loss = loss.sum().item()
                batch_acc = (pred.argmax(1) == label).sum().item() / args.batch_size

                print(f"----------------------- epoch: {epoch} / iter: {iter} -----------------------")
                # torch.set_printoptions(threshold=float('inf'))  # 设置打印选项，显示所有元素
                # np.set_printoptions(precision=2)  	# 设置 NumPy 打印选项
                # print("预测值:\n{}".format(pred))
                # print("标签:\n{}".format(label))
                # print("损失:\n{}".format(loss))
                print(f'Batch损失: {batch_loss}')
                print(f'Batch准确率: {batch_acc*100}%')
                print(f'Start梯度: {net.input.grad.sum()}')
                print(f'End梯度: {net.output.grad.sum()}')

                # visdom 可视化准确率
                x_plot = iter+len(train_datasets)*epoch
                y_plot = batch_acc
                # visdom_plot_acc(vis, x_plot, y_plot, f'ANN_{args.net_name}_MNIST_Acc', 'P_Train_Acc')
                visdom_plot_acc(vis, x_plot, y_plot, f'ANN_{args.net_name}_MNIST_Acc', f'Pytorch_lr={args.lr}')
                # visdom 可视化损失
                x_plot = iter+len(train_datasets)*epoch
                y_plot = batch_loss
                # visdom_plot_loss(vis, x_plot, y_plot, f'ANN_{args.net_name}_MNIST_Loss', 'P_Train_Loss') 
                visdom_plot_loss(vis, x_plot, y_plot, f'ANN_{args.net_name}_MNIST_Loss', f'Pytorch_lr={args.lr}')
            
            elif (args.mode == 'save'):
                if iter % 100 == 0:
                    print(f'Train Epoch: {epoch} [{iter * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}')

    if (args.mode == 'save'):
        # 开始测试
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, label in test_loader:
                data, label = data.to(args.device).requires_grad_(), label.to(args.device)
                pred = net(data)
                test_loss += F.cross_entropy(pred, label).item()  # 累积损失
                correct += (pred.argmax(1) == label).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
        save_model(args, net, accuracy, 'ANN')



def snn_mnist(args):
    # 可视化窗口
    vis = visdom.Visdom(env=args.vis_env)
    # 数据集准备
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5], std=[0.5])  # 假设单通道图像
    ])
    train_datasets = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_datasets, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)	
    test_datasets = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(dataset=test_datasets, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)
    #  定义网络、优化器、编码器
    net = getattr(snn, args.net_name)(is_train=(True if args.mode == 'visdom' else False)).to(args.device)
    encoder = encoding.PoissonEncoder()
    # 开始训练
    for epoch in range(0, args.epochs):
        net.train()
        batch_loss = 0 							# batch的损失
        batch_acc = 0							# batch的准确率
        for iter, (data, label) in enumerate(train_loader):
            data, label = data.to(args.device).requires_grad_(), label.to(args.device)
            label_onehot = F.one_hot(label, 10).float()

            net.zero_grad()  # 梯度清零
            ## SNN前向传播
            pred = 0.
            for _ in range(args.T):
                encoded_data = encoder(data).requires_grad_()
                pred += net(encoded_data)
            pred = pred / args.T
            pred_map = map_range(pred, -3, 3)

            loss = F.cross_entropy(pred_map, label_onehot)  # 损失函数
            loss.backward(retain_graph=True)  # 反向传播
            with torch.no_grad(): [param.sub_(args.lr * param.grad) for param in net.parameters()]  # 权重更新
            functional.reset_net(net)  # 重置神经元

            if (args.mode == 'visdom'):
                batch_loss = loss.sum().item()
                batch_acc = (pred.argmax(1) == label).sum().item() / args.batch_size

                print(f"----------------------- epoch: {epoch} / iter: {iter} -----------------------")
                # torch.set_printoptions(threshold=float('inf'))  # 设置打印选项，显示所有元素
                # np.set_printoptions(precision=2)  	# 2位小数
                # print("预测值:\n{}".format(pred))
                # print("标签:\n{}".format(label_onehot))
                # print("损失:\n{}".format(loss))
                print(f'Batch损失: {batch_loss}')
                print(f'Batch准确率: {batch_acc*100}%')
                print(f'Start梯度: {net.input.grad.sum()}')
                print(f'End梯度: {net.output.grad.sum()}')

                # visdom 可视化准确率
                x_plot = iter+len(train_datasets)*epoch
                y_plot = batch_acc
                # visdom_plot_acc(vis, x_plot, y_plot, f'SNN_{args.net_name}_MNIST_Acc', 'P_Train_Acc')
                visdom_plot_acc(vis, x_plot, y_plot, f"SNN_{args.net_name}_MNIST_Acc", f'Pytorch_lr={args.lr}')
                # visdom 可视化损失
                x_plot = iter+len(train_datasets)*epoch
                y_plot = batch_loss
                # visdom_plot_loss(vis, x_plot, y_plot, f'SNN_{args.net_name}_MNIST_Loss', 'P_Train_Acc')  
                visdom_plot_loss(vis, x_plot, y_plot, f'SNN_{args.net_name}_MNIST_Loss', f'Pytorch_lr={args.lr}')
            
            elif (args.mode == 'save'):
                if iter % 100 == 0:
                    print(f'Train Epoch: {epoch} [{iter * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}')
    
    if (args.mode == 'save'):
        # 开始测试
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, label in test_loader:
                data, label = data.to(args.device), label.to(args.device)
                pred = 0.
                # 运行T个时间步，触发n个脉冲，最终脉冲发放概率 = n/T
                for t in range(args.T):
                    encoded_data = encoder(data)  # 这边输出的结果是010101...的数据
                    pred += net(encoded_data)  # 网络的输入是010101...的编码
                pred = pred / args.T  # 记录了仿真时长args.T内，输出层的每个神经元的脉冲发放概率
                test_loss += F.cross_entropy(pred, label).item()  # 累积损失
                functional.reset_net(net)
                correct += (pred.argmax(1) == label).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
        save_model(args, net, accuracy, 'SNN')



def lstm_save(args):
    # 创建模型
    model = getattr(ann, "SimpleLSTM")().to(args.device)

    with torch.no_grad():
        # 检查模型输出
        sample_input = torch.randn(32, 1, 28, 28).to(args.device)  # (batch_size, seq_length, input_size)
        print("sample_input.shape:", sample_input.shape)
        # print(sample_input)
        output = model(sample_input)
        print("output.shape:", output.shape)  # 应输出: (32, output_size)   
        print(output)

    dummy_input = torch.randn(1, 1, 28, 28).to(args.device)
    torch.onnx.export(model, 
                    dummy_input, 
                    f'model/lstm.onnx', 
                    input_names=["input"], 
                    output_names=["output"])



def parse_opt():
    # 参数定义
    parser = argparse.ArgumentParser(description = "Pytorch Framework Trainning")
    # parser.add_argument("--mode", type = str, default = 'visdom', help = "看训练曲线(visdom) or 保存训练模型(save)")
    parser.add_argument("--mode", type = str, default = 'save', help = "看训练曲线(visdom) or 保存训练模型(save)")
    parser.add_argument("--vis_env", type = str, default = 'Pytorch')
    parser.add_argument("--net_name", type = str, default = 'LeNet')
    # parser.add_argument("--net_name", type = str, default = 'ResNet1')
    # parser.add_argument("--net_name", type = str, default = 'ResNet2')
    # parser.add_argument("--net_name", type = str, default = 'SimpleLSTM')
    # parser.add_argument("--net_name", type = str, default = 'HybridLSTM')
    parser.add_argument("--lr", type = float, default = 0.1)  # Pytorch框架中ann的学习率为0.001和0.0001都可，snn的学习率为0.01和0.001都可
    parser.add_argument("--epochs", type = int, default = 1)
    parser.add_argument("--batch_size", type = int, default = 64)
    parser.add_argument("--tau", type = float, default = 2.0)
    parser.add_argument("--T", type = int, default = 10)
    # parser.add_argument("--device", type = str, default = 'cpu')
    parser.add_argument("--device", type = str, default = 'cuda')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_opt()
    # ann_mnist(args)
    snn_mnist(args)
    # lstm_save(args)

	
   
