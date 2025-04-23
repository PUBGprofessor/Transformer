import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self,heads,d_model,dropout =0.1):
        super().__init__()
        self.d_model =d_model
        self.d_k = d_model //heads
        self.h = heads # 头数

        self.q_linear=nn.Linear(d_model,d_model)
        self.v_linear=nn.Linear(d_model,d_model)
        self.k_linear=nn.Linear(d_model,d_model)

        self.dropout =nn.Dropout(dropout)
        self.out= nn.Linear(d_model,d_model)

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores =torch.matmul(q,k.transpose(-2,-1))/ math.sqrt(d_k) # [batch_size, h, seq_len, seq_len]
        #掩盖掉那些为了填补长度增加的单元，使其通过softmax计算后为0
        if mask is not None:
            # mask= mask.unsqueeze(1)
            scores = scores.masked_fill(mask==1,-1e9)

        scores =F.softmax(scores,dim=-1)

        if dropout is not None:
            scores =dropout(scores) # 在这加dropout，有一定概率忽略到一些token之间的相关性
        output =torch.matmul(scores,v)

        return output # [batch_size, h, seq_len, d_k]
    
    def forward(self,q,k,v,mask=None):
        bs= q.size(0)
        # q,k,v的维度都是[batch_size,seq_len,d_model]
        # 进行线性操作划分为成h个头
        k= self.k_linear(k).view(bs,-1, self.h, self.d_k)
        q= self.q_linear(q).view(bs,-1, self.h, self.d_k)
        v= self.v_linear(v).view(bs,-1, self.h, self.d_k) # [batch_size, seq_len, h, d_k]

        # 矩阵转置 
        # 转置成 [batch_size, h, seq_len, d_k]
        k=k.transpose(1,2)
        q=q.transpose(1,2)
        v=v.transpose(1,2)
        #计算attention
        scores = self.attention(q,k,v,self.d_k,mask, self.dropout)
        #连接多个头并输入到最后的线性层
        concat =scores.transpose(1,2).contiguous().view(bs,-1,self.d_model)
        output =self.out(concat)
        return output  # [batch_size, seq_len, d_model]