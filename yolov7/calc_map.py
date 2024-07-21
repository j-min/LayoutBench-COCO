

import argparse
import json
from pathlib import Path
from tqdm import tqdm, trange

from pprint import pprint
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def calculate_mAP(anno_json, pred_json, verbose=False):
    anno = COCO(anno_json)  # init annotations api
    pred = anno.loadRes(pred_json)  # init predictions api

    if verbose:
        print('anno_json', anno_json)
        print('pred_json', pred_json)


    eval = COCOeval(anno, pred, 'bbox')
    eval.evaluate()
    eval.accumulate()
    eval.summarize()

    return eval.stats[0]

# def filter_coco(coco_dict, count=2):
#     # coco_dict = {
#     #     'images': images,
#     #     'annotations': annotations,
#     #     'categories': categories,
#     # }
    
#     out_coco_dict = {}
    
#     out_coco_dict['categories'] = coco_dict['categories']

#     out_coco_dict['images'] = []
#     out_coco_dict['annotations'] = []

#     for img in coco_dict['images']:



# def calculate_mAP_count(anno_json, pred_json, verbose=True, iter_up_to=-1):
#     if verbose:
#         print('anno_json', anno_json)
#         print('pred_json', pred_json)
        
#     # total_anno = json.load(open(anno_json))  # init annotations api
#     total_pred = json.load(open(pred_json))

#     coco = COCO(anno_json)
    
#     if verbose:
#         print("preparing count-specific image ids")
    
#     count2imageids = {}
#     for img_id in coco.imgs:
#         datum = coco.imgs[img_id]
#         file_name = datum['file_name']

#         # 'count_2_person_iter0.png'.split('_')
        
#         skip = False
#         if iter_up_to != -1:
#             max_iter = iter_up_to
#             for j in range(max_iter):
#                 if f'iter{j}' in file_name:
#                     skip = True
#                     break
#         if skip:
#             continue

#         count = file_name.split('_')[1]
#         count = int(count)
#         if count not in count2imageids:
#             count2imageids[count] = []
            
#         if img_id not in count2imageids[count]:
#             count2imageids[count] += [img_id]
            
#     if verbose:
#         pprint({k: len(v) for k, v in count2imageids.items()})
        
#     out_stats = {}

#     n_total_bbox_preds = 0
#     for count in trange(2,11,1):

#         # filter pred
#         img_ids = count2imageids[count]

#         pred_filtered = [p for p in total_pred if p['image_id'] in img_ids]
#         n_total_bbox_preds += len(pred_filtered)

#         if verbose:
#             print(count)
#             print()

#         coco_filtered = COCO()
#         dataset = {'images': [], 'annotations': [], 'categories': coco.dataset['categories']}
#         for img_id in img_ids:
#             datum = coco.imgs[img_id]
#             dataset['images'].append(datum)

#             ann_ids = coco.getAnnIds(imgIds=img_id)
#             anns = coco.loadAnns(ann_ids)
#             dataset['annotations'] += anns
#         coco_filtered.dataset = dataset

#         pred_filtered = coco_filtered.loadRes(pred_filtered)
#         eval = COCOeval(coco_filtered, pred_filtered, 'bbox')
#         eval.evaluate()
#         eval.accumulate()
#         eval.summarize()
        
#         out_stats[count] = eval.stats
        
#     # assert n_total_bbox_preds == len(total_pred)
        
#     # overall
#     if verbose:
#         print("overall")

#     pred = coco.loadRes(total_pred)
#     eval = COCOeval(coco, pred, 'bbox')
#     eval.evaluate()
#     eval.accumulate()
#     eval.summarize()
    
#     out_stats['overall'] = eval.stats
    
#     return out_stats


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_json', type=str, default='/nas-ssd/jmincho/workspace/MSR/layoutbench_real/data/count_layouts_coco.json', help='*.data path')
    parser.add_argument('--pred_json', type=str, default='/nas-ssd/jmincho/workspace/MSR/yolov7/runs/test/count_reco14/yolov7_predictions.json', help='*.data path')
    args = parser.parse_args()


    anno_json = args.anno_json
    pred_json = args.pred_json

    # anno_json = '/nas-ssd/jmincho/workspace/MSR/layoutbench_real/data/count_layouts_coco.json'
    # pred_json = '/nas-ssd/jmincho/workspace/MSR/yolov7/runs/test/count_reco13/yolov7_predictions.json'

    print('anno_json', anno_json)
    print('pred_json', pred_json)

    # print global mAP
    # mAP_out = calculate_mAP_count(anno_json, pred_json, verbose=True)

    total_result = {}

    overall_result = calculate_mAP(anno_json, pred_json, verbose=True)
    total_result['overall'] = overall_result

    for count_interval in [(2, 4), (5, 7), (8, 10)]:
        print(f"count interval: {count_interval}")

        anno_json = args.anno_json.replace('.json', f'_count{count_interval[0]}to{count_interval[1]}.json')

        calculate_mAP()
        # print(f"mAP @ {count}:", mAP_out[count][0])



    # print per class mAP
    # for i in range(10):
    #     calculate_mAP(anno_json, pred_json, verbose=True, class_id=i)