import torch
from torch import cuda 
import time
while True: 
    startTime = time.time()

    x1 = torch.zeros([1,1024,1024,128],requires_grad=True,device='cuda:0')
    x2 = torch.zeros([1,1024,1024,128],requires_grad=True,device='cuda:1')
    x3 = torch.zeros([1,1024,1024,128],requires_grad=True,device='cuda:2')
    x4 = torch.zeros([1,1024,1024,128],requires_grad=True,device='cuda:3')
    x5 = torch.zeros([1,1024,1024,128],requires_grad=True,device='cuda:4')
    x6 = torch.zeros([1,1024,1024,128],requires_grad=True,device='cuda:5')
    x7 = torch.zeros([1,1024,1024,128],requires_grad=True,device='cuda:6')
    x8 = torch.zeros([1,1024,1024,128],requires_grad=True,device='cuda:7')
    y1 = 5 * x1
    y2 = 5 * x2
    y3 = 5 * x3
    y4 = 5 * x4
    y5 = 5 * x5
    y6 = 5 * x6
    y7 = 5 * x7
    y8 = 5 * x8
    torch.mean(y1).backward()
    torch.mean(y2).backward()
    torch.mean(y3).backward()
    torch.mean(y4).backward()
    torch.mean(y5).backward()
    torch.mean(y6).backward()
    torch.mean(y7).backward()
    torch.mean(y8).backward()
    endTime = time.time()
    dualTime = endTime-startTime
    print("8 gpu dualTime: ",dualTime)