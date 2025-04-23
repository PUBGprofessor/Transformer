import torch
import torchtext
from sklearn.model_selection import train_test_split

import re
from tqdm import tqdm  # 进度条
import pandas as pd
import unicodedata


# 将数据管道组织成与torch.utils.data.DataLoader相似的inputs, targets的输出形式
class DataLoader:
    def __init__(self, data_iter):
        self.data_iter = data_iter
        self.length = len(data_iter)  # 一共有多少个batch？

    def __len__(self):
        return self.length

    def __iter__(self):
        # 注意，在此处调整text的shape为batch first
        for batch in self.data_iter:
            yield (torch.transpose(batch.src, 0, 1), torch.transpose(batch.targ, 0, 1)) # [batch_size, seq_len]
tokenizer = lambda x: x.split() # 分词器

# 将unicode字符串转化为ASCII码：
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


# 规范化字符串
def normalizeString(s):
    # print(s) # list  ['Go.']
    # s = s[0]
    s = s.lower().strip()
    s = unicodeToAscii(s)
    s = re.sub(r"([.!?])", r" \1", s)  # \1表示group(1)即第一个匹配到的 即匹配到'.'或者'!'或者'?'后，一律替换成'空格.'或者'空格!'或者'空格？'
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)  # 非字母以及非.!?的其他任何字符 一律被替换成空格
    s = re.sub(r'[\s]+', " ", s)  # 将出现的多个空格，都使用一个空格代替。例如：w='abc  1   23  1' 处理后：w='abc 1 23 1'
    return s


# 删除原始文本长度大于10个标记的样本
def filterPair(p, MAX_LENGTH=60):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH # startswith first arg must be str or a tuple of str

# 文件是英译法，我们实现的是法译英，所以进行了reverse，所以pair[1]是英语
def filterPairs(pairs, MAX_LENGTH=60):
    # 过滤，并交换句子顺序，得到法英句子对（之前是英法句子对）
    return [[pair[1], pair[0]] for pair in pairs if filterPair(pair, MAX_LENGTH=MAX_LENGTH)] # 135842

def get_dataset(pairs, src, targ):
    fields = [('src', src), ('targ', targ)]  # filed信息 fields dict[str, Field])
    examples = []  # list(Example)
    for fra, eng in tqdm(pairs): # 进度条
        # 创建Example时会调用field.preprocess方法
        examples.append(torchtext.data.Example.fromlist([fra, eng], fields))
    # print(examples[0])
    # for example in examples:
    #     print(f"Source: {example.src}, Target: {example.targ}")
          # Source: ['<start>', 'tu', 'n', 'es', 'qu', 'un', 'lache', '.', '<end>'], Target: ['<start>', 'you', 're', 'just', 'a', 'coward', '.', '<end>']
    #     break
    return examples, fields


def load_data(data_dir, MAX_LENGTH=60, BATCH_SIZE=64):
    # 数据集路径
    data_df = pd.read_csv(data_dir + 'eng-fra.txt',  # 数据格式：英语\t法语，注意我们的任务源语言是法语，目标语言是英语
                      encoding='UTF-8', sep='\t', header=None,
                      names=['eng', 'fra'], index_col=False)

    pairs = [[normalizeString(s) for s in line] for line in data_df.values] # 正则化字符串
    pairs = filterPairs(pairs, MAX_LENGTH) # reverse pairs和过滤
    # 划分数据集：训练集和验证集
    train_pairs, val_pairs = train_test_split(pairs, test_size=0.2, random_state=1234)

    # SRC_TEXT和TARG_TEXT是torchtext.data.Field的实例，负责数据预处理（preprocess）和词表构建（build_vocab）
    SRC_TEXT = torchtext.data.Field(sequential=True,
                                    tokenize=tokenizer,
                                    # lower=True,
                                    fix_length=MAX_LENGTH + 2, # 句子长度限制
                                    # 句子长度限制，超过的会被截断，短的会被填充
                                    preprocessing=lambda x: ['<start>'] + x + ['<end>'],
                                    # after tokenizing but before numericalizing
                                    # postprocessing # after numericalizing but before the numbers are turned into a Tensor
                                    )
    # print(SRC_TEXT.preprocess('hello world')) # ['<start>', 'hello', 'world', '<end>']
    # print(SRC_TEXT)
    TARG_TEXT = torchtext.data.Field(sequential=True,
                                    tokenize=tokenizer,
                                    # lower=True,
                                    fix_length=MAX_LENGTH + 2,
                                    preprocessing=lambda x: ['<start>'] + x + ['<end>'],
                                    )

    # examples, fields = get_dataset(pairs, SRC_TEXT, TARG_TEXT)
    ds_train = torchtext.data.Dataset(*get_dataset(train_pairs, SRC_TEXT, TARG_TEXT))
    ds_val = torchtext.data.Dataset(*get_dataset(val_pairs, SRC_TEXT, TARG_TEXT))
    # 访问方法：ds_train[0].src, ds_train[0].targ 此时数据被处理成了[batch_size, seq_len]的形式, 即每个句子都被填充到相同的长度,  且为数字

    # 构建词典
    SRC_TEXT.build_vocab(ds_train)  # 建立词表 并建立token和ID的映射关系
    TARG_TEXT.build_vocab(ds_train)

    # 构建数据管道迭代器
    train_iter, val_iter = torchtext.data.Iterator.splits(
        (ds_train, ds_val),           # 训练集和验证集
        sort_within_batch=True,       # 在每个批次内根据长度排序
        sort_key=lambda x: len(x.src),# 按照源序列的长度进行排序
        batch_sizes=(BATCH_SIZE, BATCH_SIZE)  # 设置训练和验证集的批次大小
    )

    train_dataloader = DataLoader(train_iter)
    val_dataloader = DataLoader(val_iter)

    return train_dataloader, val_dataloader, SRC_TEXT, TARG_TEXT, len(SRC_TEXT.vocab), len(TARG_TEXT.vocab) # vocab_size