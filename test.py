import torch
a = torch.zeros((1,3,4,5))
b = torch.tensor([1,2,3]).unsqueeze(1).unsqueeze(2)
print(a-b)