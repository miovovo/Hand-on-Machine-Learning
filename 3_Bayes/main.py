import copy
from typing import List, Dict
import string
import nltk
import mindspore as ms
from mindspore import ops
import tqdm

#  加载停用词
stopwords = set(nltk.corpus.stopwords.words("english"))
#  文本类别
text_type = {"World": 0, "Sci/Tech": 1, "Sports": 2, "Business": 3}

def load(path, type_r):
    """
    加载数据并进行预处理
    Args:
        path (str): 数据路径
        type_r (str): 使用的单词还原方法
    """
    x, y = [], []
    file = open(path, "r")
    trans = str.maketrans("", "", string.punctuation)
    if type_r == "stemmer":
        re = nltk.stem.porter.PorterStemmer().stem
    elif type_r == "lemmatizer":
        re = nltk.stem.WordNetLemmatizer().lemmatize
    else:
        raise ValueError("type error")
    for line in file:
        temp = line.split("|")  # 将类别和文本分离开
        sent = temp[1].strip()
        # 预处理文本
        sent = temp[1].strip().lower()  # 全小写
        sent = sent.translate(trans)  # 去除标点符号
        sent = nltk.word_tokenize(sent)  # 将文本标记化
        sent = [
            s for s in sent if not ((s in stopwords) or s.isdigit())
        ]  # 去停用词和数字
        sent = [re(s) for s in sent]  # 还原: 词干提取/词形还原
        x.append(sent)
        #  预处理类别
        y.append(text_type[temp[0].strip()])
    file.close()
    return x, y


def words2dic(sent: List[List[str]]) -> Dict[str, int]:
    """生成数据对应的文本库id
    Args:
        sent (List[List[str]]): 数据集
    Returns:
        文本库 (Dict[str, int])
    """
    dicts = {}
    i = 0
    for words in sent:
        for word in words:
            if word not in dicts:
                dicts[word] = i
                i += 1
    return dicts


def train_TF(data_x, data_y):
    """素贝叶斯训练
    Args:
        data_x (List[List[str]]): 训练数据
        data_y (List[int]): 真值
    Returns:
        dicts (Dict[str, int])
        p_w_c (Tensor): with shape [18359, 4]
        pc (Tensor): with shape [1, 4]
    """
    # 构建词典，用于生成统计矩阵
    dicts = words2dic(data_x)
    vocab_size = len(dicts)
    
    # n(w_i in w_c) 创建词频矩阵
    word_freq = ops.zeros((len(dicts), 4), dtype=ms.int32)
    # n(c, text) 每类下的句总数
    sent_freq = ops.zeros((1, 4), dtype=ms.int32)
    # 更新矩阵
    N = len(data_x)
    for i in tqdm.trange(N, desc="Train", ncols=80):
        x, y = data_x[i], data_y[i]
        for word in x:
            word_freq[dicts[word], y] += 1
        sent_freq[0, y] += 1
    # TODO 计算P(c)
    total_sentences = sent_freq.sum()
    pc = sent_freq / total_sentences
    # TODO 计算P(w_i|c)，并加入拉普拉斯平滑
    alpha = 1.0
    # Calculate the sum of word frequencies for each class
    sum_word_freq_per_class = word_freq.sum(axis=0) + alpha * vocab_size
    # Apply Laplace smoothing and calculate conditional probabilities
    p_w_c = (word_freq + alpha) / sum_word_freq_per_class
    return dicts, p_w_c, pc


def test_TF(data_x, data_y, dicts, p_w_c, p_c):
    """
    测试准确率
    Args:
        data_x (List[List[str]]): 测试数据
        data_y (List[int]): 真值
        dicts (Dict[str, int]): len=18359
        p_w_c (Tensor): with shape [18359, 4]
        p_c (Tensor): with shape [1, 4]
    """
    # 计算ln P(c)
    ln_p_c = ops.log(p_c)
    # 计算ln P(w_i|c)
    ln_p_s = ops.log(p_w_c)
    # 计算准确率
    count = 0
    N = len(data_x)
    for i in tqdm.trange(N, desc="Test", ncols=80):
        # for x, y in zip(data_x, data_y):
        x, y = data_x[i], data_y[i]
        p = copy.deepcopy(ln_p_c)
        for word in x:
            if word in dicts:
                idx = dicts[word]
                p += ln_p_s[idx, :]  # Add ln P(w_i|c) to ln P(c|x)
        
        # 预测类别：选择具有最大log P(c|x)的类别
        predicted_class = p.argmax().item()
        
        # 统计正确预测的数量
        if predicted_class == y:
            count += 1
    print(
        "Accuracy: {}/{} {:.2f}%".format(count, len(data_y), 100 * count / len(data_y))
    )


def train_B(data_x, data_y):
    # 构建词典，用于生成统计矩阵
    dicts = words2dic(data_x)
    # n(w in w_c) 创建词频矩阵
    sent_count = ops.zeros((len(dicts), 4), dtype=ms.int32)
    # n(c, text) 每类下的句总数
    sent_freq = ops.zeros((1, 4), dtype=ms.int32)
    N = len(data_x)
    for i in tqdm.trange(N, desc="Train", ncols=80):
        x, y = data_x[i], data_y[i]
        # TODO (选做) 实现Bernoulli方法
        # 对于每个单词，如果它出现在文档中，则更新sent_count
        seen_words = set(x)  # 使用集合去重，确保一个词在一个文档中只计数一次
        for word in seen_words:
            if word in dicts:
                sent_count[dicts[word], y] += 1
        
        # 更新每类下的文档总数
        sent_freq[0, y] += 1
    p_c = sent_freq.to(ms.float32) / len(data_y)
    sent_freq += 2
    sent_count += 1
    # 计算P(d_?|c)
    p_w_c = sent_count.to(ms.float32) / sent_freq
    return dicts, p_w_c, p_c


def test_B(data_x, data_y, dicts, p_w_c, p_c):
    ln_p_c = ops.log(p_c)
    count = 0
    N = len(data_x)
    for i in tqdm.trange(N, desc="Test", ncols=80):
        x, y = data_x[i], data_y[i]
        p = copy.deepcopy(ln_p_c)
        p_s = 1 - p_w_c
        for word in set(x):
            # TODO (选做) 实现Bernoulli方法
            # 注意过滤未收录词
            if word in dicts:
                idx = dicts[word]
                p += ops.log(p_w_c[idx, :])  # Add ln P(w|c) to ln P(c|x)              
        p += ops.log(p_s).sum(axis=0)
        if ops.argmax(p) == y:
            count += 1
    print(
        "Accuracy: {}/{} {:.2f}%".format(count, len(data_y), 100 * count / len(data_y))
    )


import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument("--device", default="GPU", choices=["CPU", "GPU"])
    parser.add_argument(
        "--type-train", default="TF", choices=["TF", "Bernoulli"], help="训练方式"
    )
    parser.add_argument(
        "--type-re",
        default="stemmer",
        choices=["stemmer", "lemmatizer"],
        help="单词还原方法",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if args.device == "GPU":
        ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

    # 单词还原方法
    type_re = args.type_re
    # 训练方法
    type_train = args.type_train

    print("训练方法: {}".format(type_train))
    print("还原方法: {}".format(type_re))

    """读取训练数据并进行预处理"""
    print("load data...")
    train_x, train_y = load("./data/news_category_train_mini.csv", type_re)
    test_x, test_y = load("./data/news_category_test_mini.csv", type_re)
    print("load success.")

    """开始训练"""
    if type_train == "TF":
        dictionary, p_w_c, p_c = train_TF(train_x, train_y)
    elif type_train == "Bernoulli":
        dictionary, p_w_c, p_c = train_B(train_x, train_y)

    """计算准确率"""
    if type_train == "TF":
        test_TF(train_x, train_y, dictionary, p_w_c, p_c)
        test_TF(test_x, test_y, dictionary, p_w_c, p_c)
    elif type_train == "Bernoulli":
        test_B(train_x, train_y, dictionary, p_w_c, p_c)
        test_B(test_x, test_y, dictionary, p_w_c, p_c)
