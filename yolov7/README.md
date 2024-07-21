

# Evaluation on LayoutBench-COCO

Evaluation of generated images on LayoutBench-COCO, with pretrained [YOLOv7 model](https://github.com/WongKinYiu/yolov7).

## Setup Env

```bash
pip install -r requirements.txt
pip install pycocotools
```

# Evaluation on LayoutBench-COCO

We provide evaluation scripts for the following models:

- [GLIGEN](https://gligen.github.io/) - [https://huggingface.co/masterful/gligen-1-4-generation-text-box](https://huggingface.co/masterful/gligen-1-4-generation-text-box)
- [ReCo](https://arxiv.org/abs/2211.15518) (COCO checkpoint) - [https://huggingface.co/j-min/reco_sd14_coco](https://huggingface.co/j-min/reco_sd14_coco)
- [ControlNet](https://arxiv.org/abs/2302.05543) - [https://huggingface.co/lllyasviel/control_v11p_sd15_seg](https://huggingface.co/lllyasviel/control_v11p_sd15_seg)
- [IterInpaint](https://github.com/j-min/IterInpaint) (COCO checkpoint) - [https://huggingface.co/j-min/iterinpaint_sd15inpaint_coco](https://huggingface.co/j-min/iterinpaint_sd15inpaint_coco)

Please see [../image_generation/README.md](../image_generation/README.md) for image generation scripts on LayoutBench-COCO.


```bash
models=(
    'reco'
    'gligen'
    'controlnet_seg'
    'iterinpaint'
)

for model in "${models[@]}"

do
    echo $model
    python prepare_image_fnames.py \
        --model $model \
        --layout_input_path "../image_generation/data/layoutbench-coco/all_layouts.json"
        --image_dump_base_dir "../image_generation/images/${model}"
        --image_list_dump_dir "./image_fnames_dir/${model}_all_layouts.txt"
done
```

## Run evaluation

```bash
export RESULTS_OUT_PATH='results/layoutbench_real.json'

layout_names=(
    'combinations_layouts_all'
    'count_layouts_all'
    'position_layouts_all'
    'size_layouts_all'
)

models=(
    'reco'
    'gligen'
    'controlnet_seg'
    'iterinpaint'
)

for model in "${models[@]}"
do
    echo $model
    for layout_name in "${layout_names[@]}"
    do
        export image_paths=./image_fnames_dir/${model}_${layout_name}.txt
        export anno_json=./data/layoutbench-coco/${layout_name}.json
        export exp_name=${model}_${layout_name}

        echo $layout_fname
        echo $image_paths

        python test.py --data data/layoutbench_real.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights yolov7.pt \
            --name $exp_name \
            --image_paths $image_paths \
            --anno_json $anno_json \
            --save-json \
            --save_result_json_path $RESULTS_OUT_PATH
    done
done
```