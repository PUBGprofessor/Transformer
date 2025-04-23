import torch
import torch.nn as nn
import math
from torch.autograd import Variable
from model.layer import Norm, FeedForward
from model.attention import MultiHeadAttention

from utils.common import get_clones

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model # 嵌入维度

        # 根据 pos 和 i 创建一个常量 PE 矩阵
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        pe = pe.unsqueeze(0) # (1, max_seq_len, d_model)
        self.register_buffer('pe', pe) # 使得 pe 成为一个 buffer，避免在训练时更新
        
    def forward(self, x):
        # x : [batch_size, seq_len, d_model]
        # 使得单词嵌入表示相对大一些 why？
        x = x * math.sqrt(self.d_model)
        # 增加位置常量到单词嵌入表示中
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len], requires_grad=False).cuda()
        return x # [batch_size, seq_len, d_model]
    
class EncoderLayer(nn.Module):
    def __init__(self,d_model, heads,dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x) # ADD & NORM 为什么是先norm再add？
        x = x + self.dropout_2(self.ff(x2))
        return x

class Encoder(nn.Module):
    def __init__(self,vocab_size,d_model,N,heads,dropout):
        super().__init__()
        self.N =N
        self.embed = torch.nn.Embedding(vocab_size,d_model) # [b, inp_seq_len] => [b, inp_seq_len, d_model]
        self.pe =PositionalEncoder(d_model) # [b, inp_seq_len, d_model]
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm =Norm(d_model)

    def forward(self, src, mask):
        x= self.embed(src)
        x= self.pe(x)
        for i in range(self.N):
            x=self.layers[i](x, mask)
        return self.norm(x)
