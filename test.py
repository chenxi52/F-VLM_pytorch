import torch.nn as nn

# 定义一个简单的模型
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 30),
)

# 遍历模型及其所有子模块
for module in model.modules():
    print(module)