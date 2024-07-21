
from viz_utils import plot_results, prepare_blank_image
from gen_utils import generate_from_layout, iterative_inpaint

import torch

from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline

from PIL import Image, ImageOps
import json
import argparse
from pathlib import Path
from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--layout_input_path', type=str,
                        default='./data/layoutbench-coco/all_layouts.json',
                        help='Path to read layout inputs'
                        )

    parser.add_argument('--layout_input_format', type=str,
                        default='normalized',
                        choices=['bbox_widget', 'normalized'],
                        )

    parser.add_argument('--image_dump_dir', type=str,
                        default='./images'
                        )
    parser.add_argument('--layout_image_dump_dir', type=str,
                        default='./images_layout_overlay'
                        )
    parser.add_argument('--layout_only_dump_dir', type=str,
                        default='./layout_only'
                        )
    parser.add_argument('--model', type=str, default='gligen',
                        choices=['gligen', 'sd', 'reco', 'iterinpaint', 'controlnet_seg'])

    parser.add_argument('--exp_name', type=str, default='')

    parser.add_argument('--n_proc', type=int, default=1)
    parser.add_argument('--proc_id', type=int, default=0)

    parser.add_argument('--layout_only', action='store_true')

    parser.add_argument('--layout_overlay_only', action='store_true')

    parser.add_argument('--skip_if_exists', action='store_true')

    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if args.model == 'gligen':
        from diffusers import StableDiffusionGLIGENPipeline

    if args.model == 'controlnet_seg':
        from diffusers import (
            ControlNetModel,
            StableDiffusionControlNetPipeline,
            UniPCMultistepScheduler,
        )
    # set seed
    from accelerate.utils import set_seed
    set_seed(args.seed)

    with open(args.layout_input_path, 'r') as f:
        _layout_input_data = json.load(f)

    print('Loaded layout data from', args.layout_input_path)

    layout_input_data = _layout_input_data
    
    if args.n_proc > 1:
        # multi-processing
        # n_procs
        # proc_id
        print(f"Number of processes: {args.n_proc}")
        print(f"Process ID: {args.proc_id}")

        print(f"Number of total data points: {len(layout_input_data)}")
        layout_input_data = layout_input_data[args.proc_id::args.n_proc]
        print(f"Number of local data points: {len(layout_input_data)}")

    model_gen_image_dir = Path(args.image_dump_dir) / args.model
    model_gen_image_layout_overlay_dir = Path(args.layout_image_dump_dir) / args.model

    if args.exp_name != '':
        exp_name = args.exp_name
    else:
        exp_name = args.layout_input_path.split('/')[-1].split('.')[0]
    
    model_gen_image_dir = model_gen_image_dir / exp_name
    model_gen_image_layout_overlay_dir = model_gen_image_layout_overlay_dir / exp_name

    print('model_gen_image_dir:', model_gen_image_dir)
    print('model_gen_image_layout_overlay_dir:', model_gen_image_layout_overlay_dir)

    if not model_gen_image_dir.exists():
        model_gen_image_dir.mkdir(parents=True, exist_ok=True)
        print('Created model_gen_image_dir:', model_gen_image_dir)

    if not model_gen_image_layout_overlay_dir.exists():
        model_gen_image_layout_overlay_dir.mkdir(parents=True, exist_ok=True)
        print('Created model_gen_image_layout_overlay_dir:', model_gen_image_layout_overlay_dir)

    pipe = None

    if args.model == 'gligen':
        print('Loading GLIGEN model...')
        # gligen_pipe = StableDiffusionGLIGENPipeline.from_pretrained(
        #     "gligen/diffusers-generation-text-box", revision="fp16", torch_dtype=torch.float16)
        gligen_pipe = StableDiffusionGLIGENPipeline.from_pretrained(
            "masterful/gligen-1-4-generation-text-box",
            variant="fp16",
            torch_dtype=torch.float16
        )
        pipe = gligen_pipe
    elif args.model == 'sd':
        print('Loading SD model...')
        original_SD_pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16)
        pipe = original_SD_pipe
    elif args.model == 'reco':
        print('Loading ReCo model...')
        reco_pipe = StableDiffusionPipeline.from_pretrained(
            "j-min/reco_sd14_coco",
            torch_dtype=torch.float16
            )
        pipe = reco_pipe

    elif args.model in ['iterinpaint']:
        it_coco_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "j-min/iterinpaint_sd15inpaint_coco",
            torch_dtype=torch.float16
        )
        pipe = it_coco_pipe

    elif args.model == 'controlnet_seg':
        checkpoint = "lllyasviel/control_v11p_sd15_seg"

        controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16
        )

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        # pipe.enable_model_cpu_offload()

    if pipe is not None:
        pipes = [pipe]

    for pipe in pipes:   
        if pipe is None:
            continue
        pipe.to('cuda')

        # def dummy(images, **kwargs):
        #     return images, False
        # pipe.safety_checker = dummy
        
        pipe.safety_checker = None
        pipe.requires_safety_checker = False
        print("Disabled safety checker")

        print("Suppressing the progress bar")
        # if one wants to set `leave=False`
        pipe.set_progress_bar_config(leave=False)
        # if one wants to disable `tqdm`
        pipe.set_progress_bar_config(disable=True)

    desc = f"Proc ID: {args.proc_id} / {args.n_proc} - {args.model}"
    for i, datum in tqdm(enumerate(layout_input_data),
                         desc=desc,
                         total=len(layout_input_data),
                         ncols=50,
                         dynamic_ncols=True,
                         disable=(args.proc_id > 0),
                         ):

        prompt = datum['caption']

        if 'file_name' in datum:
            image_fname = datum['file_name']

        elif 'id' in datum:
            image_fname = f"{datum['id']}.png"
        else:
            image_fname = f"{prompt}.png"

        model_gen_image_dump_path = model_gen_image_dir / image_fname
        model_gen_layout_dump_path = model_gen_image_layout_overlay_dir / image_fname


        # pass if there is already a generated image
        if args.skip_if_exists:
            if model_gen_image_dump_path.exists():
                # check if the image is valid
                try:
                    Image.open(model_gen_image_dump_path)
                    print('Skipping', model_gen_image_dump_path)
                    continue
                except Exception as e:
                    print('Error in loading image', model_gen_image_dump_path)
                    print(e)
                    print('Regenerating image')
                    pass

        if args.model == 'sd':
            generated_images = original_SD_pipe(prompt).images
            generated_image = generated_images[0]

        elif args.model in ['reco', 'gligen', 'controlnet_seg']:
            generated_image = generate_from_layout(
                pipe,
                model=args.model,
                prompt=prompt,
                box_captions=datum['box_captions'],
                boxes_normalized=datum['boxes_normalized'],
            )

        elif args.model in ['iterinpaint']:
            out = iterative_inpaint(
                pipe=it_coco_pipe,
                model='iterinpaint',
                prompt=datum['caption'],
                boxes_normalized=datum['boxes_normalized'],
                box_captions=datum['box_captions'],
                # verbose=True,
                last_background_inpaint=True,

                use_global_caption=True,
                parallel_foreground='parallel_fg' in args.model
            )
            generated_image = out['generated_images'][-1]

        # Saving generated images
        if not (args.layout_only or args.layout_overlay_only):
            generated_image.save(model_gen_image_dump_path)
        # print('Saved gen_image_dump_path:', model_gen_image_dump_path)

        # Saving generated images + layouts
        # if args.model in ['reco', 'gligen', 'iterinpaint']:
        if args.model != 'sd':

            # Add boounding box overlay to generated image
            try:
                if args.layout_only:
                    blank_img = prepare_blank_image()
                    src_img = blank_img
                elif args.layout_overlay_only:
                    # load from model_gen_image_dump_path
                    src_img = Image.open(model_gen_image_dump_path)
                else:
                    src_img = generated_image

                box_img = plot_results(
                    src_img,
                    boxes=datum['boxes_normalized'],
                    box_captions=datum['box_captions'],
                    to_pil=True,
                )

                resize_H = 512
                resize_W = int(512 * box_img.size[0] / box_img.size[1])
                resize_size = (resize_W, resize_H)
                box_img = box_img.resize(resize_size)

                gen_image_box_overlay = ImageOps.expand(
                    box_img, border=3, fill='black')

                # if args.layout_only:
                #     gen_image_box_overlay.save(layout_dump_path)
                # else:
                gen_image_box_overlay.save(model_gen_layout_dump_path)
                # print('Saved gen_image_box_overlay:', model_gen_layout_dump_path)
            except Exception as e:
                print('Error in plotting boxes')
                print('datum:', datum)
                print(e)