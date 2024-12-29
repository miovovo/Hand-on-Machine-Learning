#python main2.py --hidden-dims=64 --lr=1e-3 --batch-size=4 --epochs=20 --l2=0.01

import argparse
import mnist_loader
import network
import os, sys

import mindspore as ms


def parse_args():
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=4e-2, help="学习率")
    parser.add_argument(
        "--act", default="relu", choices=["relu", "sigmoid"], help="激活函数"
    )
    parser.add_argument("--device", default="CPU", choices=["CPU", "GPU"])
    parser.add_argument(
        "--hidden-dims", default="196,49", type=str, help="隐藏层大小，可以逗号分隔。"
    )
    parser.add_argument("--l2", default=0.0, type=float, help="L2 正则化参数。")
    parser.add_argument("--show", action="store_true", help="是否绘制可视化结果")

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()

    # 改变工作路径至当前文件目录
    os.chdir(os.path.abspath(os.path.dirname(__file__)))

    ms.set_context(mode=ms.GRAPH_MODE, device_target=args.device)

    epochs = args.epochs
    bs = args.batch_size
    lr = args.lr

    hidden_dims = [
        int(size.strip()) for size in args.hidden_dims.split(",") if size.strip()
    ]
    net_sizes = [784] + hidden_dims + [10]

    print("Net size:", *net_sizes)
    print(f"Bath size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Total epochs: {args.epochs}")
    print(f"L2: {args.l2}")
    print(f"Activator: {args.act}")

    output_pic = f"D:/Downloads/mnist/bs{bs}_lr{lr}_act-{args.act}_L2{args.l2}_{epochs}e.jpg"

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network(
        net_sizes,
        act=args.act,
        learning_rate=args.lr,
        lambda_=args.l2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        training_data=training_data,
        test_data=test_data,
        shuffle=True,
        cost=network.CrossEntropyCost,
    )

    if args.show:
        test_cost, test_accuracy, training_cost, training_accuracy = net.SGD(
            monitor_train_cost=True,
            monitor_train_accuracy=True,
            monitor_test_cost=True,
            monitor_test_accuracy=True,
        )
        network.plot_result(
            epochs,
            test_cost,
            test_accuracy,
            training_cost,
            training_accuracy,
            output_pic,
        )
    else:
        # 只输出测试集上的准确率
        net.SGD()

