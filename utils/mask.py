import torch
import numpy as np

def create_padding_mask(seq, pad):  # seq [b, seq_len]
    # seq = torch.eq(seq, torch.tensor(0)).float() # pad=0的情况
    seq = torch.eq(seq, torch.tensor(pad)).float()  # pad!=0
    return seq[:, np.newaxis, np.newaxis, :]  # =>[b, 1, 1, seq_len]

def create_look_ahead_mask(size):  # seq_len
    mask = torch.triu(torch.ones((size, size)), diagonal=1)
    # mask = mask.device() #
    return mask  # [seq_len, seq_len]


# inp [b, inp_seq_len] 序列已经加入pad填充
# targ [b, targ_seq_len] 序列已经加入pad填充
def create_mask(inp, targ, pad=0):
    # encoder padding mask
    enc_padding_mask = create_padding_mask(inp, pad)  # =>[b,1,1,inp_seq_len] mask=1的位置为pad

    # decoder's first attention block(self-attention)
    # 使用的padding create_mask & look-ahead create_mask
    look_ahead_mask = create_look_ahead_mask(targ.shape[-1])  # =>[targ_seq_len,targ_seq_len] ##################
    dec_targ_padding_mask = create_padding_mask(targ, pad)  # =>[b,1,1,targ_seq_len]
    combined_mask = torch.max(look_ahead_mask, dec_targ_padding_mask)  # 结合了2种mask =>[b,1,targ_seq_len,targ_seq_len]

    # decoder's second attention block(encoder-decoder attention) 使用的padding create_mask
    # 【注意】：这里的mask是用于遮挡encoder output的填充pad，而encoder的输出与其输入shape都是[b,inp_seq_len,d_model]
    # 所以这里mask的长度是inp_seq_len而不是targ_mask_len
    dec_padding_mask = create_padding_mask(inp, pad)  # =>[b,1,1,inp_seq_len] mask=1的位置为pad

    return enc_padding_mask, combined_mask, dec_padding_mask
    # [b,1,1,inp_seq_len], [b,1,targ_seq_len,targ_seq_len], [b,1,1,inp_seq_len]