import torch
a = torch.tensor(1.).cuda()

b=a.cpu().numpy()
print(b)