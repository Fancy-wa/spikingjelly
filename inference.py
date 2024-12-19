import os
import argparse
import time
import torch
import onnx
import numpy
import onnxruntime
from onnx2torch import convert

# from ann import LeNet, ResNet
# from temp import conv2d


def torch_to_onnx(args):
    # 加载torch模型
    torch_model = torch.load(args.model).to(args.device)
    # 初始化模型
    # torch_model = conv2d().to(args.device)
    # 初始化输入
    dummy_input = torch.randn(1, 2, 28, 28)
    # dummy_input = torch.randn(1, 15, 20)
    # dummy_input = torch.randn(1, 2, 1024, 1024)
    # 转换为onnx
    torch.onnx.export(torch_model, 
                      dummy_input, 
                      "model/conv2d_mini2.onnx", 
                      input_names=["input"], 
                      output_names=["output"])


def onnx_runtime(args):
    '''
    不适合推理SNN网络，多个时间步下神经元没有记忆功能
    '''
    # 加载输入数据
    data_np = numpy.load(args.input_data)
    original_input = torch.from_numpy(data_np).to(args.device)

    # 加载 ONNX 模型
    t1 = time.time()
    onnx_model = onnx.load(args.model)  # 加载ONNX模型
    onnx.checker.check_model(onnx_model)  # 检查模型的有效性
    t2 = time.time()
    print("初始化时间: {:.2f}ms".format((t2 - t1) * 1000))

    # 使用 ONNX Runtime 进行推理
    ort_session = onnxruntime.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    with torch.no_grad():  # 不会记录计算图，速度加快，内存减少
        print(f"输入shape: {original_input.shape}")
        t3 = time.time()

        # ONNX Runtime 推理
        if "ANN" in args.model:
            ort_inputs = {ort_session.get_inputs()[0].name: original_input.cpu().numpy()}
            ort_outs = ort_session.run(None, ort_inputs)
        elif "SNN" in args.model:
            # for _ in range(args.T):
            ort_inputs = {ort_session.get_inputs()[0].name: original_input.cpu().numpy()}
            ort_outs = ort_session.run(None, ort_inputs)

        t4 = time.time()
        print("运行时间: {:.2f}ms".format((t4 - t3) * 1000))

        # 输出结果
        ort_output = torch.from_numpy(ort_outs[0]).to(args.device)
        print(f"输出shape: {ort_output.shape}")
        print(f"输出: {ort_output}")


def torch_runtime(args):
    # 加载输入
    data_np = numpy.load(args.input_data)
    original_input = torch.from_numpy(data_np).to(args.device)

    t1 = time.time()
    _, ext = os.path.splitext(args.model)  # 读取文件扩展名
    if ext == '.onnx':
        onnx_model = onnx.load(args.model)  # 加载ONNX模型
        torch_model = convert(onnx_model).to(args.device)  # 转换为PyTorch模型
    elif ext == '.pth' or ext == '.pt':
        torch_model = torch.load(args.model).to(args.device)  # 加载PyTorch模型
    # print(torch_model)  # 打印模型结构
    t2 = time.time()
    print("初始化时间: {:.2f}ms".format((t2 - t1)*1000))

    torch_model.eval()
    with torch.no_grad():  # 不会记录计算图，速度加快，内存减少
        print(f"输入shape: {original_input.shape}")
        t3= time.time()
        output = 0.
        if "ANN" in args.model:
            output = torch_model(original_input)
        elif "SNN" in args.model:
            for _ in range(args.T):
                output += torch_model(original_input)
            output = output / args.T
        # output = torch_model(original_input)
        t4 = time.time()
        print("运行时间: {:.2f}ms".format((t4 - t3)*1000))
        print(f"输出shape: {output.shape}")
        print(f"输出: {output}")


def parse_opt():
    # 参数定义
    parser = argparse.ArgumentParser(description = "Running SpikingJellyRuntime")
    parser.add_argument('--input_data', type=str, default='data/images.npy')
    # parser.add_argument('--model', type=str, default='model/ANN_LeNet_Acc_97.92.onnx')
    # parser.add_argument('--model', type=str, default='model/ANN_LeNet_Acc_97.92.pt')
    # parser.add_argument('--model', type=str, default='model/ANN_ResNet2_Acc_99.03.onnx')
    # parser.add_argument('--model', type=str, default='model/ANN_ResNet2_Acc_99.03.pt')
    # parser.add_argument('--model', type=str, default='model/ANN_SimpleLSTM_Acc_90.1.onnx')
    # parser.add_argument('--model', type=str, default='model/ANN_SimpleLSTM_Acc_90.1.pt')

    # parser.add_argument('--model', type=str, default='model/SNN_LeNet_Acc_96.1.onnx')
    # parser.add_argument('--model', type=str, default='model/SNN_LeNet_Acc_96.1.pt')
    # parser.add_argument('--model', type=str, default='model/SNN_ResNet2_Acc_98.35.onnx')
    # parser.add_argument('--model', type=str, default='model/SNN_ResNet2_Acc_98.35.pt')
    # parser.add_argument('--model', type=str, default='model/SNN_HybridLSTM_Acc_77.31.onnx')
    # parser.add_argument('--model', type=str, default='model/SNN_HybridLSTM_Acc_77.31.pt')

    # parser.add_argument('--model', type=str, default='model/SNN_SimpleLSTM_Acc_73.35.onnx')
    parser.add_argument('--model', type=str, default='model/SNN_SimpleLSTM_Acc_73.35.pt')

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--T", type = int, default = 10)
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_opt()
    # torch_to_onnx(args)
    # onnx_runtime(args)  # onnx不适合推理SNN网络，多个时间步下神经元没有记忆功能
    torch_runtime(args)
