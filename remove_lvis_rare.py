# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import json
import torch
from detic.modeling.clip import clip as clip_model
from detic.prompt_engineering import get_prompt_templates
import pickle

@torch.no_grad()
def get_custom_text_feat(class_names):
    clip,  _ = clip_model.load('RN50')
    clip = clip.cuda()
    def extract_mean_emb(text):
        tokens = clip_model.tokenize(text).cuda()
        if len(text) > 10000:
            text_features = torch.cat([
                clip.encode_text(text[:len(text) // 2]),
                clip.encode_text(text[len(text) // 2:])],
                dim=0)
        else:
            text_features = clip.encode_text(tokens)
        text_features = torch.mean(text_features, 0, keepdims=True)
        return text_features[0]

    templates = get_prompt_templates()
    clss_embeddings = []
    for clss in class_names:
        txts = [template.format(clss.replace('-other','').replace('-merged','').replace('-stuff','')) for template in templates]
        clss_embeddings.append(extract_mean_emb(txts))
    txts = ['background']
    clss_embeddings.append(extract_mean_emb(txts))
    text_emb = torch.stack(clss_embeddings, dim=0)
    text_emb /= text_emb.norm(dim=-1, keepdim=True) 
    return text_emb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann', default='datasets/lvis/lvis_v1_train.json')
    args = parser.parse_args()

    print('Loading', args.ann)
    data = json.load(open(args.ann, 'r'))
    catid2freq = {x['id']: x['frequency'] for x in data['categories']}# [id: freq]
    # print('ori #anns', len(data['annotations']))
    exclude = ['r']
    data['annotations'] = [x for x in data['annotations'] 
                            if catid2freq[x['category_id']] not in exclude]  
    print('filtered #anns', len(data['annotations']))
    out_path = args.ann[:-5] + '_norare.json'
    print('Saving to', out_path)
    json.dump(data, open(out_path, 'w'))
    print('done')
    
    # exclude = ['r']
    # data['base_class_name'] = [x['name'] for x in data['categories'] if x['frequency'] not in exclude]
    # data['name'] = [x['name'] for x in data['categories']]
    # base_text_feats = get_custom_text_feat(data['base_class_name'])
    # all_text_feats = get_custom_text_feat(data['name'])
    # print('class_name', len(data['name']))
    # # print(data['base_class_name'])
    # with open('datasets/lvis/lvis_base_cls.pkl', 'wb') as f:
    #     pickle.dump(base_text_feats, f)

    # with open('datasets/lvis/lvis_cls.pkl', 'wb') as f:
    #     pickle.dump(all_text_feats, f)