from torch.nn.functional import sigmoid
import torch
image_size_tesnsor = [torch.tensor([300,400]), torch.tensor([200,500])]
q=  torch.stack(image_size_tesnsor).max(0)
print(q)