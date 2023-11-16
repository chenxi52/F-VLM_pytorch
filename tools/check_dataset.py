import json
from pprint import pprint
# 加载 JSON 文件
with open('datasets/lvis/lvis_v1_train.json', 'r') as f:
    data = json.load(f)

# 提取所有的类别 ID
# class_ids = [category['id'] for category in data['categories']]

image_id = data['images'][0]['id']
annotations = [annotation for annotation in data['annotations'] if annotation['image_id'] == image_id]

# 打印标注
for annotation in annotations:
    pprint(annotation)
print('*'*50)

with open('datasets/lvis/lvis_v1_train_norare.json', 'r') as f:
    data = json.load(f)

# 提取所有的类别 ID
# class_ids = [category['id'] for category in data['categories']]


annotations = [annotation for annotation in data['annotations'] if annotation['image_id'] == image_id]

# 打印标注
for annotation in annotations:
    pprint(annotation)


