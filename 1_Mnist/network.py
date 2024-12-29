# 标准库
import os
import random
from typing import List, Tuple

# 第三方库
import numpy as np
import mindspore as ms
import matplotlib.pyplot as plt
import tqdm
from mindspore import ops


# 定义 cross-entropy 和 quadratic 损失函数
class CrossEntropyCost(object):
    """交叉熵损失"""

    @staticmethod
    def fn(a, y):
        """
        Args:
            a (ms.Tensor): 预测值
            y (ms.Tensor): 真值
        Returns:
            loss (ms.Tensor): 交叉熵损失
        """
        eps = 1e-5
        if len(y.shape) == 2:
            y = y.reshape(*y.shape, 1)
        ce_loss = -(y * ops.log(a + eps) + (1 - y) * ops.log(1 - a + eps))
        ce_loss = nan_to_num(ce_loss)
        # ce_loss = ops.nan_to_num(ce_loss) # midspore 自带的 ops.nan_to_num 目前不支持 GPU
        loss = ce_loss.mean()
        return loss

    @staticmethod
    def delta(z, a, y):
        """返回输出层的误差方程 δ^L"""
        return a - y


class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """返回 a 和标签 y 之间的损失"""
        return 0.5 * ops.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y, act_prime):
        """返回输出层的误差方程 δ^L"""
        return (a - y) * act_prime(z)


# 定义主网络
class Network(object):

    def __init__(
        self,
        sizes,
        act,
        learning_rate,
        lambda_,
        epochs,
        batch_size,
        training_data,
        test_data,
        shuffle=False,
        cost=CrossEntropyCost,
    ):
        """
        Args:
            sizes (List[int]): 神经网络中各层神经元的个数
                例如 [784, 192, 30, 10] 表示一个包含两个隐藏层的神经网络,
                输入层包含 784 个神经元, 第一隐层包含 192 个神经元,
                第二隐层包含 30 个神经元, 输出层包含 10 个神经元.
            act (str): "relu" 或 "sigmoid"，代表使用的激活函数方法
            learning_rate (float): 学习率
            weight_decay (float): 学习率衰减系数，在训练过程中学习率会逐渐变小，
                最终衰减至初始学习率*weight_decay.
            lambda_ (float): 正则化系数
            epochs (int): 总训练步数
            batch_size(int): 批次大小
            training_data (collection): 训练集
            test_data (collection): 测试集
            shuffle (bool): 是否打乱训练集
            cost (callable): 损失函数
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.lr = learning_rate
        self.lambda_ = lambda_
        self.epochs = epochs
        self.batch_size = batch_size
        self.default_weight_initializer()
        self.cost = cost

        training_data = list(training_data)
        self.n_train_data = len(training_data)
        if shuffle:
            random.shuffle(training_data)
        self.training_data = batch_(training_data, self.batch_size)

        test_data = list(test_data)
        self.n_test_data = len(test_data)
        self.test_data = batch_(test_data, self.batch_size)

        if act == "relu":
            self.act = ops.relu
            self.act_prime = relu_prime
        elif act == "sigmoid":
            self.act = ops.sigmoid
            self.act_prime = sigmoid_prime

    def default_weight_initializer(self):
        """
        初始化 weights 和 biases 均值为 0,
        标准差为 1 的高斯分布,
        输入层的神经元不设置 biases
        """
        dtype = ms.float32
        self.biases = [ops.zeros(size=(y, 1), dtype=dtype) for y in self.sizes[1:]]
        self.weights = [
            ops.randn((x, y), dtype=dtype)
            for x, y in zip(self.sizes[:-1], self.sizes[1:])
        ]

    def feedforward(self, a):
        """输入 a, 返回神经网络的输出结果"""
        bs = a.shape[0]
        weights = [
            ops.repeat_elements(w.reshape(1, *w.shape), bs, 0) for w in self.weights
        ]
        biases = [
            ops.repeat_elements(b.reshape(1, *b.shape), bs, 0) for b in self.biases
        ]
        for b, w in zip(biases[:-1], weights[:-1]):
            # TODO: 补全前向传播过程
            # ...
            z = ops.bmm(w.transpose(0, 2, 1), a) + b
            a = self.act(z)

        w, b = weights[-1], biases[-1]
        z = ops.bmm(w.transpose(0, 2, 1), a) + b
        a = softmax(z)
        return a

    def SGD(
        self,
        monitor_train_cost=False,
        monitor_train_accuracy=False,
        monitor_test_cost=False,
        monitor_test_accuracy=True,
    ):
        """
        使用小批量随机梯度下降算法来训练神经网络.
        该方法返回一个包含四个列表的元组, 列表内存有每个 epoch 的计算结果。
        """
        n_train = len(self.training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        # 将训练数据打乱顺序
        bx, by = self.training_data
        for epoch in range(self.epochs):
            N = len(bx)
            for i in tqdm.trange(N, desc=f"Epoch #{epoch + 1}/{self.epochs}", ncols=80):
                self.update_batch((bx[i], by[i]), self.lr, n_train)
            print(f"Epoch #{epoch + 1} finished.")

            # 计算代价与准确率
            if monitor_train_cost:
                cost = self.total_cost(self.training_data)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_train_accuracy:
                accuracy = self.accuracy(self.training_data, convert=True)
                training_accuracy.append(accuracy)
                print(f"Accuracy on training data: {accuracy*100:.2f}%.")
            if monitor_test_cost:
                cost = self.total_cost(self.test_data, convert=True)
                evaluation_cost.append(cost)
                print("Cost on test data: {}".format(cost))

            if monitor_test_accuracy:
                accuracy = self.accuracy(self.test_data)
                evaluation_accuracy.append(accuracy)
                print(f"Accuracy on test data: {accuracy*100:.2f}%.")
            print()

        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_batch(self, xy, lr, n):
        """
        Args:
            xy (Tuple[ms.Tensor]): x, y
                x.shape = [bs, 784, 1], y.shape = [bs, 10]
            lr (float): 学习率
            lambda_ (float): L2 正则化参数
            n (int): 训练集的总长度
        """
        x, y = xy
        nabla_b, nabla_w = self.backprop(x, y)
        nabla_b = [ops.mean(nb, axis=0) for nb in nabla_b]
        nabla_w = [ops.mean(nw, axis=0) for nw in nabla_w]
        # TODO: 补全更新 weights 与 biases
        # ...
        self.weights = [
            w - (lr * (nw + (self.lambda_ / n) * w))
            for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b - lr * nb
            for b, nb in zip(self.biases, nabla_b)
        ]

    def backprop(self, x: ms.Tensor, y: ms.Tensor):
        """
        返回表示对于指定损失函数 C_x 的梯度元组 (nabla_b, nabla_w),
        其中, nabla_b 和 nabla_w 是由 numpy arrays 组成的列表,
        与 self.biases 和 self.weights 的组成相似。
        Args:
            batched_x (ms.Tensor): 形状为 [bs, 784, 1] 的 Tensor
            batched_y (ms.Tensor): 形状为 [bs, 10] 的 Tensor
        """
        # bs * 784
        bs = x.shape[0]
        # bs * 784 * 196, bs * 196 * 64, bs * 64 * 10
        weights = [
            ops.repeat_elements(w.reshape(1, *w.shape), bs, 0) for w in self.weights
        ]
        # bs * 196 * 1, bs * 64 * 1
        biases = [
            ops.repeat_elements(b.reshape(1, *b.shape), bs, 0) for b in self.biases
        ]
        nabla_b = [ops.zeros(b.shape) for b in biases]
        nabla_w = [ops.zeros(w.shape) for w in weights]
        # 前向传播
        activation = x
        activations = [x]  # 该列表存储层与层之间所有的 a(激活) 值, a = sigmoid(w*x + b)
        zs = []  # 该列表存储层与层之间的 z 值, z = w*x + b

        for b, w in zip(biases, weights):
            # [bs, w, h] x [bs, h, l] -> [bs, w, l]
            # TODO: 补全前向传播计算过程
            # ...
            #print(w.shape)
            #print(x.shape)
            z = ops.bmm(w.transpose(0, 2, 1), x) + b
            zs.append(z)
            a = self.act(z)
            activations.append(a)
            x = a

        # 输出层为 softmax 函数, 而不是采用 sigmoid
        activations[-1] = softmax(zs[-1])
        # 后向传播
        delta = (self.cost).delta(zs[-1], activations[-1], y.reshape(*y.shape, 1))
        nabla_b[-1] = delta
        nabla_w[-1] = ops.bmm(activations[-2], delta.transpose(0, 2, 1))

        # 循环中变量 l 的含义如下: l = 1 表示最后一层神经元, l = 2 表示倒数第二层神经元...
        for l in range(2, self.num_layers):
            z = zs[-l]
            # TODO: 补全反向传播过程
            # ...
            sp = self.act_prime(z)  # 激活函数导数
            delta = ops.bmm(weights[-l + 1], delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = ops.bmm(activations[- l - 1], delta.transpose(0, 2, 1))
            
        return (nabla_b, nabla_w)

    def accuracy(self, data, convert=False):
        """返回 data 中分类正确的图像的总数
        Args:
            convert (bool): 是否进行 one-hot 转换
                在训练集上应初始化为 True, 验证集或测试集上应初始化为 False
        """
        N = len(data)
        bx, by = data
        N = len(bx)
        cnt, n = 0, 0
        desc = "Train Accuracy" if convert else "Test Accuracy"
        for i in tqdm.trange(N, desc=desc, ncols=80):
            x, y = bx[i], by[i]
            pred = ops.argmax(self.feedforward(x), 1).reshape(x.shape[0])
            if convert:
                gt = ops.argmax(y, 1).to(dtype=pred.dtype)
            else:
                gt = y.to(dtype=pred.dtype)
            cnt += int((pred == gt).sum())
            n += int((pred == gt).shape[0])
        return cnt / n

    def total_cost(self, data, convert=False):
        """
        Args:
            convert (bool): 是否进行 one-hot 格式转换
                当使用训练集时，convert=False
                当使用测试集时，convert=True
        Returns:
            损失值
        """
        cost = 0.0
        bx, by = data
        bs = self.batch_size
        N = len(bx)
        desc = "Train Cost" if not convert else "Test Cost"
        for i in tqdm.trange(N, desc=desc, ncols=80):
            x, y = bx[i], by[i]
            a = self.feedforward(x)
            if convert:
                y = ops.one_hot(y.to(ms.int32), 10).reshape(y.shape[0], 10, 1)
            cost += self.cost.fn(a, y) / N
        cost += (
            0.5
            * (self.lambda_ / (N * bs))
            * sum(ops.norm(w) ** 2 for w in self.weights)
        )
        return cost


def nan_to_num(tensor: ms.Tensor, nan=0, posinf=0, neginf=0):
    """将 Tensor 中的异常值替换为正常值，默认为 0.
    ops.nan_to_num 不支持 GPU 平台
    """
    tensor[tensor.isnan()] = nan
    tensor[tensor.isposinf()] = posinf
    tensor[tensor.isneginf()] = neginf

    return tensor


def relu(z: ms.Tensor):
    """relu 激活函数，同 ops.relu"""
    return ops.max(z, axis=0)


def relu_prime(z: ms.Tensor):
    """relu 激活函数的导数"""
    a = ops.ones(z.shape)
    rp = ops.where(z > 0, a, 0.0)
    return rp


def sigmoid(z: ms.Tensor):
    """ops.sigmoid 对正负无穷处理不完善"""
    z = 1 / (ops.exp(-z) + 1)
    z = nan_to_num(z)
    return z


def sigmoid_prime(z: ms.Tensor):
    return sigmoid(z) * (1 - sigmoid(z))


def softmax(x: ms.Tensor, axis: int = 1):
    """ops.softmax 对异常值处理不完善"""
    x = ops.softmax(x, axis=axis)
    x[x.isnan()] = 0.0
    x[x.isneginf()] = 0.0
    x[x.isposinf()] = 0.0
    return x


def batch_(data_list: List[Tuple[ms.Tensor]], bs: int) -> Tuple[List[ms.Tensor]]:
    N = len(data_list)
    n = N // bs
    x = [x for x, _ in data_list]
    y = [y for _, y in data_list]
    bx = []
    by = []
    for i in range(n):
        bx.append(ops.stack(x[i * bs : (i + 1) * bs]))
        by.append(ops.stack(y[i * bs : (i + 1) * bs]))
    if n * bs < N:
        bx.append(ops.stack(x[n * bs :]))
        by.append(ops.stack(y[n * bs :]))
    return bx, by


def plot_result(
    epochs, test_cost, test_accuracy, training_cost, training_accuracy, file_name
):
    """
    绘制训练集和测试集的损失及准确率,
    并将所得结果保存
    """
    epoch = np.arange(epochs)
    plt.subplot(1, 2, 1)
    plt.plot(epoch, test_cost, "r", label="test_cost")
    plt.plot(epoch, training_cost, "k", label="training_cost")
    plt.title("Cost Range")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epoch, test_accuracy, "r", label="test_accuracy")
    plt.plot(epoch, training_accuracy, "k", label="training_accuracy")
    plt.title("Accuracy Range")
    plt.legend()

    #if not os.path.exists("output"):
    #    os.makedirs("output")
    plt.savefig(file_name)
