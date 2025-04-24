import torch
import os
print(os.getcwd()) # /home/xijian

from model.Transformer import Transformer
from dataset.data_loader import load_data 
from utils.schedule import CustomSchedule
from trainer import trainer
# 参数
ngpu = 1

use_cuda = torch.cuda.is_available()  ##检测是否有可用的gpu
device = torch.device("cuda:0")
print('device=', device)

project_dir = './'
# 当前目录cwd
cwd = project_dir
data_dir = cwd + './data/' # 数据所在目录
num_layers = 6 # 编码器和解码器的层数
d_model = 512
# dff = 512
num_heads = 8
EPOCHS = 100 # 50 # 30  # 20
print_trainstep_every = 50  # 每50个step做一次打印
save_dir = './save_model/'  # 模型保存目录
MAX_LENGTH = 60 # 句子最大长度
dropout_rate = 0.1
pad = 1 # padding的ID
BATCH_SIZE = 64 * ngpu

train_dataloader, val_dataloader, SRC_TEXT, TARG_TEXT, input_vocab_size, target_vocab_size = load_data(data_dir, MAX_LENGTH, BATCH_SIZE=BATCH_SIZE)

transformer = Transformer(
                            src_vocab=input_vocab_size,
                            trg_vocab=target_vocab_size,
                            d_model=d_model,
                            #  dff=2048,
                            N=num_layers,
                            heads=num_heads,
                            dropout=dropout_rate,
                            )
transformer = transformer.to(device)
if ngpu > 1: # 并行化
    transformer = torch.nn.DataParallel(transformer,  device_ids=list(range(ngpu))) # 设置并行执行  device_ids=[0,1]

checkpoint = torch.load(r"save_model\新建文件夹\050_0.94_ckpt.tar", map_location=device)  # checkpoint_path 是保存的路径
transformer.load_state_dict(checkpoint['net'])
transformer.eval()

out = torch.zeros(1, 2 * MAX_LENGTH).to(device) # [1, 10]  # 1个batch，10个词
for i in val_dataloader:
    print(i[0][0], i[1][0]) # [64, 10] [64, 10]
    while out[-1] != 2:
        out.append(transformer(i[0][0].squeeze(0).to(device), out, None, i[1][j].to(device)))
    pre = transformer(i[0].to(device), i[1].to(device), None, i[1].to(device))
    break