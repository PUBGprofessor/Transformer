import torch
import pandas as pd
import datetime
import time
import copy
from utils.mask import create_mask
from utils.loss import mask_loss_func, mask_accuracy_func


# 打印时间
def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m_%d %H:%M:%S')
    print('\n' + "=========="*8 + '%s'%nowtime)

class trainer:
    def __init__(self, model, optimizer, lr_scheduler, train_loader, val_loader, device='cuda', pad=0):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_loader
        self.val_dataloader = val_loader
        self.device = device
        self.metric_name = 'acc'
        self.create_mask = lambda x, y: create_mask(x, y, pad=pad)  # 创建mask函数
        # df_history = pd.DataFrame(columns=['epoch', 'loss', metric_name]) # 记录训练历史信息
        self.df_history = pd.DataFrame(columns=['epoch', 'loss', self.metric_name, 'val_loss', 'val_' + self.metric_name])

    def train_model(self, epochs, save_dir, print_every=50):
        ngpu = 1
        starttime = time.time()
        print('*' * 27, 'start training...')
        printbar()

        best_acc = 0.
        for epoch in range(1, epochs + 1):

            # lr_scheduler.step() # 更新学习率

            loss_sum = 0.
            metric_sum = 0.

            for step, (inp, targ) in enumerate(self.train_dataloader, start=1):
                # inp [64, 10] , targ [64, 10]
                loss, metric = self.train_step(inp, targ)

                loss_sum += loss
                metric_sum += metric

                # 打印batch级别日志
                if step % print_every == 0:
                    print('*' * 8, f'[step = {step}] loss: {loss_sum / step:.3f}, {self.metric_name}: {metric_sum / step:.3f}')

                self.lr_scheduler.step()  # 更新学习率

            # 一个epoch的train结束，做一次验证
            # test(model, train_dataloader)
            val_loss_sum = 0.
            val_metric_sum = 0.
            for val_step, (inp, targ) in enumerate(self.val_dataloader, start=1):
                # inp [64, 10] , targ [64, 10]
                loss, metric = self.validate_step(self.model, inp, targ)

                val_loss_sum += loss
                val_metric_sum += metric

            # 记录和收集1个epoch的训练（和验证）信息
            # record = (epoch, loss_sum/step, metric_sum/step)
            record = (epoch, loss_sum/step, metric_sum/step, val_loss_sum/val_step, val_metric_sum/val_step)
            self.df_history.loc[epoch - 1] = record

            # 打印epoch级别的日志
            print('EPOCH = {} loss: {:.3f}, {}: {:.3f}, val_loss: {:.3f}, val_{}: {:.3f}'.format(
                record[0], record[1], self.metric_name, record[2], record[3], self.metric_name, record[4]))
            printbar()

            # 保存模型
            # current_acc_avg = metric_sum / step
            current_acc_avg = val_metric_sum / val_step # 看验证集指标
            if current_acc_avg > best_acc  and epoch % 5 == 0:  # 保存更好的模型
                best_acc = current_acc_avg
                checkpoint = save_dir + '{:03d}_{:.2f}_ckpt.tar'.format(epoch, current_acc_avg)
                if self.device.type == 'cuda' and ngpu > 1:
                    # model_sd = model.module.state_dict()  ##################
                    model_sd = copy.deepcopy(self.model.module.state_dict())
                else:
                    # model_sd = model.state_dict(),  ##################
                    model_sd = copy.deepcopy(self.model.state_dict())  ##################
                torch.save({
                    'loss': loss_sum / step,
                    'epoch': epoch,
                    'net': model_sd,
                    'opt': self.optimizer.state_dict(),
                    'lr_scheduler': self.lr_scheduler.state_dict()
                }, checkpoint)


        print('finishing training...')
        endtime = time.time()
        time_elapsed = endtime - starttime
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        return self.df_history

    def train_step(self, inp, targ):
        # 目标（target）被分成了 tar_inp 和 tar_real
        # tar_inp 作为输入传递到解码器。
        # tar_real 是位移了 1 的同一个输入：在 tar_inp 中的每个位置，tar_real 包含了应该被预测到的下一个标记（token）。
        targ_inp = targ[:, :-1]
        targ_real = targ[:, 1:] # [b, targ_seq_len - 1] 这里和input的长度不一样了，是可以的（编码器和解码器的seq_len可以不同）

        enc_padding_mask, combined_mask, dec_padding_mask = self.create_mask(inp, targ_inp)

        inp = inp.to(self.device)
        targ_inp = targ_inp.to(self.device)
        targ_real = targ_real.to(self.device)
        enc_padding_mask = enc_padding_mask.to(self.device)
        combined_mask = combined_mask.to(self.device)
        dec_padding_mask = dec_padding_mask.to(self.device)
        # print('device:', inp.device, targ_inp)

        self.model.train()  # 设置train mode

        self.optimizer.zero_grad()  # 梯度清零

        # forward
        prediction = self.model(inp, targ_inp, enc_padding_mask, dec_padding_mask)
        # [b, targ_seq_len, target_vocab_size]
        # {'..block1': [b, num_heads, targ_seq_len, targ_seq_len],
        #  '..block2': [b, num_heads, targ_seq_len, inp_seq_len], ...}

        loss = mask_loss_func(targ_real, prediction)
        metric = mask_accuracy_func(targ_real, prediction)

        # backward
        loss.backward()  # 反向传播计算梯度
        self.optimizer.step()  # 更新参数

        return loss.item(), metric.item()

    def validate_step(self, inp, targ):
        targ_inp = targ[:, :-1]
        targ_real = targ[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = self.create_mask(inp, targ_inp)

        inp = inp.to(self.device)
        targ_inp = targ_inp.to(self.device)
        targ_real = targ_real.to(self.device)
        enc_padding_mask = enc_padding_mask.to(self.device)
        combined_mask = combined_mask.to(self.device)
        dec_padding_mask = dec_padding_mask.to(self.device)

        self.model.eval()  # 设置eval mode

        with torch.no_grad():
            # forward
            prediction = self.model(inp, targ_inp, enc_padding_mask, dec_padding_mask)
            # [b, targ_seq_len, target_vocab_size]
            # {'..block1': [b, num_heads, targ_seq_len, targ_seq_len],
            #  '..block2': [b, num_heads, targ_seq_len, inp_seq_len], ...}

            val_loss = mask_loss_func(targ_real, prediction)
            val_metric = mask_accuracy_func(targ_real, prediction)

        return val_loss.item(), val_metric.item()
