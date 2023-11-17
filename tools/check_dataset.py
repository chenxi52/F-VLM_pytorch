import json
from pprint import pprint
# 加载 JSON 文件
with open('datasets/lvis/lvis_v1_train.json', 'r') as f:
    data = json.load(f)

# 提取所有的类别 ID
annotations = data['annotations']

# 打印注释的数量
print('Number of annotations: ', len(annotations))
with open('datasets/lvis/lvis_v1_train_seen.json', 'r') as f:
    data = json.load(f)

# 提取所有的类别 ID
annotations = data['annotations']

# 打印注释的数量
print('Number of annotations: ', len(annotations))

