import torch

 
input = torch.tensor([1, 2, 2, 3, 3, 3, 5,5])
counts = torch.bincount(input)
 
print(counts)  # 输出: tensor([0, 1, 2, 3, 4])