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
d_model = 128
# dff = 512
num_heads = 8
EPOCHS = 100 # 50 # 30  # 20
print_trainstep_every = 50  # 每50个step做一次打印
save_dir = './save_model/'  # 模型保存目录
MAX_LENGTH = 60 # 句子最大长度
dropout_rate = 0.1
pad = 1 # padding的ID
BATCH_SIZE = 128 * ngpu

train_dataloader, val_dataloader, SRC_TEXT, TARG_TEXT, input_vocab_size, target_vocab_size = load_data(data_dir, MAX_LENGTH, BATCH_SIZE=64)

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


optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
lr_scheduler = CustomSchedule(optimizer, d_model, warm_steps=4000)

trainer = trainer(transformer, optimizer, lr_scheduler, train_dataloader, val_dataloader, device, pad) # trainer类

if not os.path.exists(save_dir): # 如果目录不存在，则创建
    os.mkdir(save_dir) # 创建目录

# 开始训练
df_history = trainer.train_model(EPOCHS, save_dir, print_trainstep_every)
print(df_history)