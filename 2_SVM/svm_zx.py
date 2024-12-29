#python svm.py --C=0.0001 --T=1000 --loss-type=hinge --show
import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import argparse

import tqdm
import mindspore as ms
from mindspore import ops

from typing import Dict


def parse_args():
    parser = argparse.ArgumentParser(description="SVM")
    parser.add_argument("-C", type=float, default=0.001, help="损失函数权重")
    parser.add_argument("-T", type=int, default=10000, help="迭代次数")
    parser.add_argument(
        "--loss-type",
        default="log",
        choices=["hinge", "log", "exp"],
        help="损失函数类型",
    )
    parser.add_argument(
        "--interval", default=500, type=int, help="计算目标函数的迭代间隔"
    )
    parser.add_argument("--show", action="store_true", help="是否输出可视化结果")
    parser.add_argument("--device", default="CPU", choices=["CPU", "GPU"])

    args = parser.parse_args()
    return args


def load(path: str, data_type: str) -> Dict[str, ms.Tensor]:
    """
    根据指定路径读取训练集或测试集
    由于二者的数据格式略有不同，所以需要区分处理
    Args:
        path (str):数据集路径
        data_type (str): "train"或"test"
    Returns:
        Dict[str, Tensor]
    """
    data = scipy.io.loadmat(path)

    # 原始数据的label是0/1格式,需要转化为-1/1格式
    if data_type == "train":
        data["X"] = ms.Tensor.from_numpy(data["X"])
        data["y"] = ms.Tensor.from_numpy(data["y"]).to(ms.int64) * 2 - 1
    elif data_type == "test":
        data["Xtest"] = ms.Tensor.from_numpy(data["Xtest"])
        data["ytest"] = ms.Tensor.from_numpy(data["ytest"]).to(ms.int64) * 2 - 1

    return data


def plot(func_list, eval_interval, loss_type, C, T, acc):
    """
    绘制模型在训练过程中的目标函数曲线
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    ts = [t for t in range(0, T, eval_interval)]
    print(func_list)
    axes[0].plot(ts, func_list, "k", label="training_cost")
    axes[0].set_title("{} acc={}% C={} T={}".format(loss_type, acc[-1], C, T))
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("f(W,b)")
    axes[1].plot(ts, acc, "red", label="test_acc")
    axes[1].set_title("Test Accuracy")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("acc")

    if not os.path.exists("./output"):
        os.makedirs("./output")
    plt.savefig(f"./output/{loss_type}_C={C:.5f}_T={T:05d}.jpg")


def func(train_x, train_y, W, b, lambda_, loss_type):
    """
    根据当前W、b与loss种类,计算训练集样本的目标函数,平均值或总和均可
    """
    num_train = train_x.shape[0]
    func_ = 0
    for i in tqdm.trange(num_train, desc="Calc Loss", ncols=80):  # 计算经验损失的总和
        x_i = train_x[[i]].T  # 1899*1
        y_i = train_y[i]  # 1

        z = y_i * (W.T @ x_i + b)
        z = z[0][0]
        # TODO 完成计算函数
        if loss_type == "hinge":
            func_ += max(0, 1 - z)
        elif loss_type == "exp":
            func_ += ops.exp(-z)
        elif loss_type == "log":
            func_ += np.log(1 + np.exp(-z))

    func_ /= num_train  # 经验损失的平均
    func_ += 0.5 * lambda_ * ops.dot(W.T, W)  # 正则项的平均
    return func_[0][0]


def evaluate(test_x, test_y, W, b):
    num_test = test_x.shape[0]
    num_correct = 0
    for i in tqdm.trange(test_x.shape[0], desc="Test", ncols=80):
        res = ops.dot(W.T, test_x[[i]].T) + b
        if res[0][0] >= 0 and test_y[i] == 1:
            num_correct += 1
        elif res[0][0] < 0 and test_y[i] == -1:
            num_correct += 1

    acc = num_correct / num_test
    return acc


def pegasos(train, test, C, T, loss_type="hinge", eval_interval=100, show=False):
    """
    eval_interval: 每隔eval_interval次记录一次当前目标函数值,用于画图
    佩加索斯算法
    """
    train_x = train["X"].to(ms.float32)  # 4000*1899
    train_y = train["y"].to(ms.float32)  # 4000*1

    test_x = test["Xtest"].to(ms.float32)  # 1000*1899
    test_y = test["ytest"].to(ms.float32)  # 1000*1

    num_train = train_x.shape[0]  # 4000
    num_test = test_x.shape[0]  # 1000
    num_features = train_x.shape[1]  # 1899

    # 记录目标函数值,用于画图
    func_list = []

    # 初始化lambda_
    lambda_ = 1 / (num_train * C)

    # 高斯初始化权重W和偏置b
    W = ops.randn(num_features, 1)  # 1899*1
    b = ops.randn(1)  # 1

    # 随机生成一组长度为T,元素范围在[0, num_train-1]的下标(可重复),供算法中随机选取训练样本
    choose = np.random.randint(0, num_train, T)  # T

    accuracies = []

    for t in tqdm.trange(1, T + 1, desc="Train", ncols=80):
        # TODO: 写出下降步长eta_t的计算公式
        eta_t = 1 / (lambda_ * t)

        i = int(choose[t - 1])  # 随机选取的训练样本下标
        x_i = train_x[[i]].T  # shape=(1899, 1)
        y_i = train_y[i]  # shape=(1,)

        if loss_type == "hinge":
            # Hinge 损失的梯度更新公式
            z = ops.dot(W.T, x_i) + b
            z = ops.squeeze(z)  # 将z压缩为标量，去掉多余的维度
            if (y_i * z) < 1:  # 用标量比较
                grad_W = -y_i * x_i
                grad_b = -y_i
            else:
                grad_W = 0
                grad_b = 0

        elif loss_type == "exp":
            # Exp 损失的梯度更新公式
            z = ops.dot(W.T, x_i) + b
            exp_input = -y_i * z
            assert len(exp_input) == 1, "exp_input must be length 1"

            #如果大于等于3跳过 防止爆炸
            if exp_input[0].item() >= 3:
                continue

            exp_term = ops.exp(exp_input)  # Exp部分
            grad_W = -y_i * x_i * exp_term
            grad_b = -y_i * exp_term

        elif loss_type == "log":
            # Log 损失的梯度更新公式
            z = ops.dot(W.T, x_i) + b
            exp_term = ops.exp(y_i * z)  # 使用MindSpore的exp函数
            grad_W = -y_i * x_i / (1 + exp_term)
            grad_b = -y_i / (1 + exp_term)

            # 更新权重和偏置
        W = (1 - eta_t * lambda_) * W - eta_t * grad_W
        b = b - eta_t * grad_b

        if show and (t % eval_interval == 0 or t == T):
            func_now = func(train_x, train_y, W, b, lambda_, loss_type)
            func_list.append(func_now)
            print(len(func_list))
            # 计算准确率
            acc = 100 * evaluate(test_x, test_y, W, b)
            accuracies.append(acc)

    if len(accuracies) == 0:
        acc = 100 * evaluate(test_x, test_y, W, b)
        accuracies.append(acc)
    print("acc = {:.1f}%".format(accuracies[-1]))
    return accuracies, func_list


if __name__ == "__main__":
    args = parse_args()

    ms.set_context(mode=ms.GRAPH_MODE, device_target=args.device)

    C = args.C
    T = args.T  # 迭代次数
    print(f"device: {args.device}, loss type={args.loss_type}, C={C}, T={T}")
    eval_interval = args.interval  # 每隔多少次迭代计算一次目标函数

    loss_type = args.loss_type

    train = load("./data/spamTrain.mat", "train")  # 4000条
    test = load("./data/spamTest.mat", "test")  # 1000条

    acc, func_list = pegasos(train, test, C, T, loss_type, eval_interval, args.show)

    if args.show:
        plot(func_list, eval_interval, loss_type, C, T, acc)
