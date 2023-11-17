import json
from pprint import pprint
# 加载 JSON 文件
with open('datasets/lvis/lvis_v1_train.json', 'r') as f:
    data = json.load(f)
print(data.keys())
# 提取所有的类别 ID
annotations = data['annotations']

<<<<<<< HEAD
# 打印注释的数量
print('Number of annotations: ', len(annotations))
with open('datasets/lvis/lvis_v1_train_seen.json', 'r') as f:
=======
# 打印标注

print('*'*50)

with open('datasets/lvis/lvis_v1_train_norare.json', 'r') as f:
>>>>>>> bbfd164d998e76a2d1da29a49b200d0cfdc1576a
    data = json.load(f)
print(data.keys())
# 提取所有的类别 ID
<<<<<<< HEAD
annotations = data['annotations']

# 打印注释的数量
print('Number of annotations: ', len(annotations))
=======
# class_ids = [category['id'] for category in data['categories']]


>>>>>>> bbfd164d998e76a2d1da29a49b200d0cfdc1576a

