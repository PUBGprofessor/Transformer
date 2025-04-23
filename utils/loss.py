import torch

pad = 1  # padding token

# 'none'表示直接返回b个样本的loss，默认求平均
loss_object = torch.nn.CrossEntropyLoss(reduction='none')
# 【注意】，当输入是多维时交叉熵的参数维度，跟tf2不一样，tf2中pred是【b,seq_len,vocab_size】
# pytorch中pred应该调整为【b,vocab_size,seq_len】
"""
- Input: :math:`(N, C)` where `C = number of classes`, or
          :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.

- Target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of
          K-dimensional loss.
"""
# real [b, targ_seq_len]
# pred [b, targ_seq_len, target_vocab_size]
def mask_loss_func(real, pred):
    # print(real.shape, pred.shape)
    # _loss = loss_object(pred, real) # [b, targ_seq_len]
    _loss = loss_object(pred.transpose(-1, -2), real)  # [b, targ_seq_len]
    # print(_loss.shape)  # [b, targ_seq_len]
    # logical_not  取非
    # mask 每个元素为bool值，如果real中有pad，则mask相应位置就为False
    # mask = torch.logical_not(real.eq(0)).type(_loss.dtype) # [b, targ_seq_len] pad=0的情况
    mask = torch.logical_not(real.eq(pad)).type(_loss.dtype)  # [b, targ_seq_len] pad!=0的情况

    # 对应位置相乘，token上的损失被保留了下来，pad的loss被置为0或False 去掉，不计算在内
    _loss *= mask
    return _loss.sum() / mask.sum().item()

# real [b, targ_seq_len]
# pred [b, targ_seq_len, target_vocab_size]
def mask_accuracy_func(real, pred):
    _pred = pred.argmax(dim=-1)  # [b, targ_seq_len, target_vocab_size]=>[b, targ_seq_len]
    corrects = _pred.eq(real)  # [b, targ_seq_len] bool值

    # logical_not  取非
    # mask 每个元素为bool值，如果real中有pad，则mask相应位置就为False
    # mask = torch.logical_not(real.eq(0)) # [b, targ_seq_len] bool值 pad=0的情况
    mask = torch.logical_not(real.eq(pad))  # [b, targ_seq_len] bool值 pad!=0的情况

    # 对应位置相乘，token上的值被保留了下来，pad上的值被置为0或False 去掉，不计算在内
    corrects *= mask
    return corrects.sum().float() / mask.sum().item()