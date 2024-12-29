# 标准库
import pickle
import gzip

# 第三方库
import numpy as np
import mindspore as ms
from mindspore import ops


def load_data():
    """
    以元组的形式加载 MNIST 数据集, 包括训练集、测试集、验证集.
    其中, 训练集是一个二维元组, 第一维具有 50000 组条目,
    每个条目有 784 个数值,
    代表单个 MNIST 图片的 28 * 28 = 784 像素值;
    第二维同样具有 50000 组条目, 每个条目对应该手写数字的标签,
    取值范围为 (0...9).
    验证集和测试集的数据组成方式与训练集相似,
    只是仅包含 10000 张照片。
    """
    f = gzip.open("./data/mnist.pkl.gz", "rb")
    training_data, validation_data, test_data = pickle.load(f, encoding="bytes")
    f.close()

    dtype = ms.float32
    training_data = (
        ms.Tensor.from_numpy(training_data[0]).to(dtype=dtype),
        ms.Tensor.from_numpy(training_data[1]).to(dtype=ms.int32),
    )
    validation_data = (
        ms.Tensor.from_numpy(validation_data[0]).to(dtype=dtype),
        ms.Tensor.from_numpy(validation_data[1]).to(dtype=dtype),
    )
    test_data = (
        ms.Tensor.from_numpy(test_data[0]).to(dtype=dtype),
        ms.Tensor.from_numpy(test_data[1]).to(dtype=dtype),
    )
    return (training_data, validation_data, test_data)


def load_data_wrapper():
    """
    在 load_data 函数的基础上,
    返回一个 (training_data, validation_data, test_data) 元组.
    其中, training_data 是包括 50000 个二维元组 (x, y) 的列表,
    x 是一个 784 维的 numpy.ndarray,
    表示输入图像 (28*28 = 784) 的像素信息,
    y 是对应的 one-hot 标签向量.
    validation_data 和 test_data 是包括 10000 个二维元组 (x, y) 的列表,
    x 是一个 784 维的 numpy.ndarray,
    表示输入图像 (28*28) 的像素信息, y 是对应的标签值.
    """
    tr_d, va_d, te_d = load_data()

    dtype = ms.float32

    training_inputs = tr_d[0].reshape(-1, 784, 1)
    training_results = ops.one_hot(tr_d[1], 10)
    training_data = zip(training_inputs, training_results)

    validation_inputs = va_d[0].reshape(-1, 784, 1)
    validation_data = zip(validation_inputs, va_d[1])

    test_inputs = te_d[0].reshape(-1, 784, 1)
    test_data = zip(test_inputs, te_d[1])

    return (training_data, validation_data, test_data)
