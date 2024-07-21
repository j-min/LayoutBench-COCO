

# Download LayoutBench-COCO

```bash
mkdir data && cd data
git clone https://huggingface.co/datasets/j-min/layoutbench-coco
```

# Setup Env

```bash
pip install -r requirements.txt
```

# Generate Images from LayoutBench-COCO layouts

We provide inference scripts for the following models:

- [GLIGEN](https://gligen.github.io/) - [https://huggingface.co/masterful/gligen-1-4-generation-text-box](https://huggingface.co/masterful/gligen-1-4-generation-text-box)
- [ReCo](https://arxiv.org/abs/2211.15518) (COCO checkpoint) - [https://huggingface.co/j-min/reco_sd14_coco](https://huggingface.co/j-min/reco_sd14_coco)
- [ControlNet](https://arxiv.org/abs/2302.05543) - [https://huggingface.co/lllyasviel/control_v11p_sd15_seg](https://huggingface.co/lllyasviel/control_v11p_sd15_seg)
- [IterInpaint](https://github.com/j-min/IterInpaint) (COCO checkpoint) - [https://huggingface.co/j-min/iterinpaint_sd15inpaint_coco](https://huggingface.co/j-min/iterinpaint_sd15inpaint_coco)

```bash
python inference.py --model "gligen" --layout_input_path "data/layoutbench-coco/all_layouts.json"

python inference.py --model "reco" --layout_input_path "data/layoutbench-coco/all_layouts.json"

python inference.py --model "controlnet_seg" --layout_input_path "data/layoutbench-coco/all_layouts.json"

python inference.py --model "iterinpaint" --layout_input_path "data/layoutbench-coco/all_layouts.json"
```

# Evaluation of Generated Images

please check [../yolov7/README.md](../yolov7/README.md) for evaluation scripts.