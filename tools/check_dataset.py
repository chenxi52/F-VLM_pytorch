import json
from pprint import pprint
# 加载 JSON 文件
with open('datasets/coco/zero-shot/instances_val2017_all_2_oriorder.json', 'r') as f:
    data = json.load(f)
# 提取所有的类别 ID
annotations = data['annotations']
print(len(annotations))

# 打印标注

print('*'*50)

with open('datasets/coco/annotations/instances_val2017.json', 'r') as f:
    data = json.load(f)
annotations = data['annotations']
print(len(annotations))
# 提取所有的类别 ID
# class_ids = [category['id'] for category in data['categories']]



