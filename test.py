import torch

a = torch.randint(0,5,(2, 1, 4))
print(a)
print(a[...,::2])
print(a[...,1::2])