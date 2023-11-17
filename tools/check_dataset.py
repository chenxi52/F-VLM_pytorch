import json
from pprint import pprint
# 加载 JSON 文件
with open('datasets/lvis/lvis_v1_train.json', 'r') as f:
    data = json.load(f)
print(data.keys())
# 提取所有的类别 ID
annotations = data['annotations']

# 打印标注

print('*'*50)

with open('datasets/lvis/lvis_v1_train_norare.json', 'r') as f:
    data = json.load(f)
print(data.keys())
# 提取所有的类别 ID
# class_ids = [category['id'] for category in data['categories']]



