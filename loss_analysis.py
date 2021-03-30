import torch
from torch import nn
#多属性分类
def binary_cross_entropyloss(prob, target, weight=None):
    loss = -weight * (target * torch.log(prob) + (1 - target) * (torch.log(1 - prob)))
    loss = torch.sum(loss) / torch.numel(target)
    return loss

label = torch.tensor([
    [1., 0,1],
    [1., 0,1],

])
predict = torch.tensor([
    [0.1, 0.3,0.9],
    [0.2, 0.8,0.5]
])

weight1 = torch.tensor([
    [1., 1, 1],
    [1., 1.,1],
])

###########################################
# loss1 = nn.BCELoss(weight=weight1)
# l1 = loss1(predict, label)
# loss = binary_cross_entropyloss(predict, label, weight=weight1)
# print(l1, loss)

###########################################
import torch
import torch.nn as nn
import numpy as np
entroy = nn.CrossEntropyLoss()
input = torch.Tensor([[-0.7715,-0.6205,-0.2562]])
target = torch.tensor([0,0,1])
output = entroy(input,target)
print(output) #采用CrossEntropyLoss计算的结果。
myselfout = -(input[:,0])+np.log(np.exp(input[:,0])+np.exp(input[:,1])+np.exp(input[:,2])) #自己带公式计算的结果
print(myselfout)
lsf = nn.LogSoftmax()
loss = nn.NLLLoss()
lsfout = lsf(input)
lsfnout = loss(lsfout,target)
print(lsfnout)