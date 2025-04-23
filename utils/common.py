
import torch.nn as nn

def get_clones(module, N):
    # 复制N次module
    return nn.ModuleList([module for i in range(N)])