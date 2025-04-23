import torch
import torchtext
# from torchtext.legacy.data import Field, TabularDataset, BucketIterator
from sklearn.model_selection import train_test_split

import random
import re
from tqdm import tqdm  # 进度条
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import unicodedata
import datetime
import time
import copy

import os
print(os.getcwd()) # /home/xijian

from model.Transformer import Transformer

ngpu = 1

# 加载model
checkpoint = save_dir + '040_0.76_ckpt.tar'
print('checkpoint:', checkpoint)
# ckpt = torch.load(checkpoint, map_location=device)  # dict  save 在 CPU 加载到GPU
ckpt = torch.load(checkpoint)  # dict  save 在 GPU 加载到 GPU
# print('ckpt', ckpt)
transformer_sd = ckpt['net']
# optimizer_sd = ckpt['opt'] # 不重新训练的话不需要
# lr_scheduler_sd = ckpt['lr_scheduler']

reload_model = Transformer(num_layers,
                           d_model,
                           num_heads,
                           dff,
                           input_vocab_size,
                           target_vocab_size,
                           pe_input=input_vocab_size,
                           pe_target=target_vocab_size,
                           rate=dropout_rate)

reload_model = reload_model.to(device)
if ngpu > 1:
    reload_model = torch.nn.DataParallel(reload_model,  device_ids=list(range(ngpu))) # 设置并行执行  device_ids=[0,1]


print('Loading model ...')
if device.type == 'cuda' and ngpu > 1:
   reload_model.module.load_state_dict(transformer_sd)
else:
   reload_model.load_state_dict(transformer_sd)
print('Model loaded ...')