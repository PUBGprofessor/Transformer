import torch
import os
print(os.getcwd()) # /home/xijian
from utils.mask import create_mask
from model.Transformer import Transformer
from dataset.data_loader import load_data, tokenzier_decode, tokenizer_encode
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

checkpoint = torch.load(r"save_model\epoch30_d512\030_0.81_ckpt.tar", map_location=device)  # checkpoint_path 是保存的路径
transformer.load_state_dict(checkpoint['net'])
transformer.eval()

# out = torch.zeros(1, 2 * MAX_LENGTH).to(device) # [1, 10]  # 1个batch，10个词
# for inp, targ in val_dataloader:
#     print(inp[0], targ[0]) # [64, 10] [64, 10]
#     print(SRC_TEXT.vocab.itos[inp[0][0]], TARG_TEXT.vocab.itos[targ[0][1]]) # [64, 10] [64, 10]
#     # enc_padding_mask, combined_mask, dec_padding_mask = create_mask(inp, targ, 1)
#     # print(combined_mask) # [64, 1, 1, 10] [64, 1, 10, 10] [64, 1, 10, 10]
#     # while out[-1] != 2:
#     #     break
#     break

#  inp_sentence 一个法语句子，例如"je pars en vacances pour quelques jours ."
def evaluate(model, inp_sentence):
    model.eval()  # 设置eval mode

    inp_sentence_ids = tokenizer_encode(inp_sentence, SRC_TEXT.vocab)  # 转化为索引
    # print(tokenzier_decode(inp_sentence_ids, SRC_TEXT.vocab))
    encoder_input = torch.tensor(inp_sentence_ids).unsqueeze(dim=0)  # =>[b=1, inp_seq_len=10]
    # print(encoder_input.shape)

    decoder_input = [TARG_TEXT.vocab.stoi['<start>']]
    decoder_input = torch.tensor(decoder_input).unsqueeze(0)  # =>[b=1,seq_len=1]
    # print(decoder_input.shape)

    with torch.no_grad():
        for i in range(MAX_LENGTH + 2):
            enc_padding_mask, combined_mask, dec_padding_mask = create_mask(encoder_input.cpu(), decoder_input.cpu()) ################
            # [b,1,1,inp_seq_len], [b,1,targ_seq_len,inp_seq_len], [b,1,1,inp_seq_len]

            encoder_input = encoder_input.to(device)
            decoder_input = decoder_input.to(device)
            enc_padding_mask = enc_padding_mask.to(device)
            combined_mask = combined_mask.to(device)
            dec_padding_mask = dec_padding_mask.to(device)

            # forward
            predictions= model(encoder_input,
                                    decoder_input,
                                    enc_padding_mask,
                                    combined_mask,
                                    dec_padding_mask)
            # [b=1, targ_seq_len, target_vocab_size]
            # {'..block1': [b, num_heads, targ_seq_len, targ_seq_len],
            #  '..block2': [b, num_heads, targ_seq_len, inp_seq_len], ...}

            # 看最后一个词并计算它的 argmax
            prediction = predictions[:, -1:, :]  # =>[b=1, 1, target_vocab_size]
            prediction_id = torch.argmax(prediction, dim=-1)  # => [b=1, 1]
            # print('prediction_id:', prediction_id, prediction_id.dtype) # torch.int64
            if prediction_id.squeeze().item() == TARG_TEXT.vocab.stoi['<end>']:
                return decoder_input.squeeze(dim=0)

            decoder_input = torch.cat([decoder_input, prediction_id],
                                      dim=-1)  # [b=1,targ_seq_len=1]=>[b=1,targ_seq_len=2]
            # decoder_input在逐渐变长

    return decoder_input.squeeze(dim=0)
    # [targ_seq_len],
    # {'..block1': [b, num_heads, targ_seq_len, targ_seq_len],
    #  '..block2': [b, num_heads, targ_seq_len, inp_seq_len], ...}



# s = 'je pars en vacances pour quelques jours .'
# evaluate(s)

s = 'je pars en vacances pour quelques jours .'
s_targ = 'i m taking a couple of days off .'
pred_result = evaluate(Transformer, s)
pred_sentence = tokenzier_decode(pred_result, TARG_TEXT.vocab)
print('real target:', s_targ)
print('pred_sentence:', pred_sentence)