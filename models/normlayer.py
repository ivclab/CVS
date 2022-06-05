import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

# Reference
# https://medium.com/@auro_227/writing-a-custom-layer-in-pytorch-14ab6ac94b77
# https://github.com/azgo14/classification_metric_learning
# https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch/blob/master/criteria/softmax.py
class NormLayer(nn.Module):
    """ Custom Linear layer but mimics a standard linear layer """
    def __init__(self, n_classes, embed_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(n_classes,embed_dim))
        # initialize weight without bias terms
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return F.linear(
            F.normalize(x,p=2,dim=1),
            F.normalize(self.weight,p=2,dim=1)
        )
