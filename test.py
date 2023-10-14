from torch.nn.functional import sigmoid
import torch
a = torch.randn([2,3,4])
b = a.flatten(0,1)
print(b.size())