import torch
# from torch import autocast
# from torch.utils.data import Dataset
# import json
# import pandas as pd
from PIL import Image
from PIL import ImageDraw
# from pathlib import Path
import numpy as np
# import random
import copy

from utils import create_reco_prompt

def generate_from_layout(pipe,
                         model='gligen',
                         prompt: str ='',
                         box_captions=[],
                         boxes_normalized=[],
                         height=512,
                         width=512,

                         box_size_ratio=1.0,

                         num_inference_steps=50,
                         verbose=False,
                         ):
    assert model in ['gligen', 'reco', 'stable', 'controlnet_seg', 'boxdiff_sd']

    assert len(box_captions) == len(boxes_normalized), \
        f"Number of box captions ({len(box_captions)}) and boxes ({len(boxes_normalized)}) do not match"
    
    if box_size_ratio != 1.0:
        # resize the box sizes
        boxes_normalized = resize_boxes(boxes_normalized=boxes_normalized, size_ratio=box_size_ratio)
        if verbose:
            print(f'Resized boxes with ratio {box_size_ratio}')

    if model == 'gligen':
        images = pipe(
            prompt=prompt,
            gligen_phrases=box_captions,
            gligen_boxes=boxes_normalized,
            # gligen_scheduled_sampling_beta=1,

            output_type="pil",
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
        ).images
        return images[0]
    
    elif model == 'reco':
        reco_prompt = create_reco_prompt(
            caption=prompt,
            phrases=box_captions,
            boxes=boxes_normalized,
            normalize_boxes=False,
        )

        images = pipe(
            prompt=reco_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
        ).images
        return images[0]
    
    elif model == 'controlnet_seg':
        #     seg_mask = create_seg_mask_from_layout(
        #         boxes_normalized=datum['boxes_normalized'],
        #         box_captions=datum['box_captions']
        #     )
        seg_mask = create_seg_mask_from_layout(
            boxes_normalized=boxes_normalized,
            box_captions=box_captions
        )

        # image = pipe(prompt, num_inference_steps=30, generator=generator, image=seg_mask).images[0]
        images = pipe(
            prompt=prompt,
            image=seg_mask,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
        ).images
        return images[0]
    

        return img



    
def create_binary_mask_from_layouts(
    boxes_normalized,
    size=512,
    ):
    """    
    boxes_normalized: list of normalized boxes (xyxy)

    Returns:
        PIL image with a single channel (luminance) and black background. 
        White pixels in the mask are repainted while black pixels are preserved.
    """

    mask_img = Image.new('L', (size,size))
    mask_draw = ImageDraw.Draw(mask_img)
    mask_draw.rectangle([(0, 0), mask_img.size], fill=0) # fill with black

    if len(boxes_normalized) == 0:
        return mask_img

    # unnormalized box coordinates
    unnormalized_boxes = np.array(boxes_normalized) * size
    unnormalized_boxes = unnormalized_boxes.astype(int)
    # clip to image size
    unnormalized_boxes[:, 0] = np.clip(unnormalized_boxes[:, 0], 0, size)
    unnormalized_boxes[:, 1] = np.clip(unnormalized_boxes[:, 1], 0, size)
    unnormalized_boxes[:, 2] = np.clip(unnormalized_boxes[:, 2], 0, size)
    unnormalized_boxes[:, 3] = np.clip(unnormalized_boxes[:, 3], 0, size)

    for box in unnormalized_boxes.tolist():
        mask_draw.rectangle(box, fill=255)

    return mask_img


from seg_utils import coco_label2color, ade_label2color

def create_seg_mask_from_layout(
    boxes_normalized,
    box_captions,
    size=512,
    # palette=ada_palette,
    background_class='wall',
):
    """create segementation mask with bounding box layouts
    
    using ada palette from https://huggingface.co/lllyasviel/control_v11p_sd15_seg
    """
    seg_mask_image = Image.new('RGB', (size,size))
    seg_mask_draw = ImageDraw.Draw(seg_mask_image)

    # background_color = coco_label2color[background_class]
    background_color = ade_label2color[background_class]
    seg_mask_draw.rectangle([(0, 0), seg_mask_image.size],
                            fill=background_color)

    if len(boxes_normalized) == 0:
        return seg_mask_image

    # unnormalized box coordinates
    unnormalized_boxes = np.array(boxes_normalized) * size
    unnormalized_boxes = unnormalized_boxes.astype(int)
    # clip to image size
    unnormalized_boxes[:, 0] = np.clip(unnormalized_boxes[:, 0], 0, size)
    unnormalized_boxes[:, 1] = np.clip(unnormalized_boxes[:, 1], 0, size)
    unnormalized_boxes[:, 2] = np.clip(unnormalized_boxes[:, 2], 0, size)
    unnormalized_boxes[:, 3] = np.clip(unnormalized_boxes[:, 3], 0, size)

    for box_caption, box in zip(box_captions, unnormalized_boxes.tolist()):
        color = coco_label2color[box_caption]
        seg_mask_draw.rectangle(box, fill=color)

    return seg_mask_image


def resize_boxes(
    boxes_normalized,
    size_ratio=1.0,
    ):
    """
    boxes_normalized: list of normalized boxes (xyxy)
    size_ratio: ratio to resize the boxes

    Returns:
        list of resized boxes (xyxy)        
    """
    new_boxes_normalized = copy.deepcopy(boxes_normalized)
    for i in range(len(boxes_normalized)):
        box = boxes_normalized[i]

        x1, y1, x2, y2 = box

        w = x2 - x1
        h = y2 - y1

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        new_w = w * size_ratio
        new_h = h * size_ratio

        new_x1 = center_x - new_w / 2
        new_y1 = center_y - new_h / 2
        new_x2 = center_x + new_w / 2
        new_y2 = center_y + new_h / 2

        # clip to image size
        new_x1 = np.clip(new_x1, 0, 1)
        new_y1 = np.clip(new_y1, 0, 1)
        new_x2 = np.clip(new_x2, 0, 1)
        new_y2 = np.clip(new_y2, 0, 1)

        new_boxes_normalized[i] = [new_x1, new_y1, new_x2, new_y2]
    return new_boxes_normalized

def inpaint_from_layout(pipe,
                        model='gligen',
                        image: Image = None,
                        prompt: str ='',
                        box_captions=[],
                        boxes_normalized=[],

                        box_size_ratio=1.0,

                        height=512,
                        width=512,
                        mask_image=None,
                        num_inference_steps=50,

                        strength=0.99,
                        ):
    
    assert model in ['gligen', 'reco', 'stable', 'kandinsky', 'sdxl', 'iterinpaint']

    assert len(box_captions) == len(boxes_normalized), \
        f"Number of box captions ({len(box_captions)}) and boxes ({len(boxes_normalized)}) do not match"
    

    if box_size_ratio != 1.0:
        # resize the box sizes
        boxes_normalized = resize_boxes(boxes_normalized=boxes_normalized, size_ratio=box_size_ratio)

    if model == 'gligen':
        images = pipe(
            prompt=prompt,
            gligen_inpaint_image=image,
            gligen_phrases=box_captions,
            gligen_boxes=boxes_normalized,
            # gligen_scheduled_sampling_beta=1,
            height=height,
            width=width,
            output_type="pil",

            num_inference_steps=num_inference_steps,
        ).images
        return images[0]
    
    elif model == 'reco':
        image = generate_from_layout(
            pipe=pipe,
            model=model,
            prompt=prompt,
            box_captions=box_captions,
            boxes_normalized=boxes_normalized,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
        )
        return image
    
    elif model == 'iterinpaint':

        if mask_image is None:
            mask_image = create_binary_mask_from_layouts(
                boxes_normalized=boxes_normalized,
                size=image.size[0],
            )

        images = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            ).images
        return images[0]

def iterative_inpaint(
    pipe,
    model='gligen',
    prompt: str ='',
    initial_image: Image = None,

    box_captions=[],
    boxes_normalized=[],
    size=512,
    paste=True,
    verbose=False,

    strength=0.99,
    # guidance_scale=4.0

    last_background_inpaint=False,

    box_size_ratio=1.0,

    use_global_caption=False,

    parallel_foreground=False,

    num_inference_steps=50,
    ):
    """
    inpaint_pipe: callable to inpaint model
        - prompt: str
        - context_img: PIL Image
        - 

    box_captions: list of box captions
    boxes_normalized: list of normalized boxes (xyxy)
    """

    global_caption = None
    if use_global_caption:
        global_caption = prompt

    if box_size_ratio != 1.0:
        # resize the box sizes
        boxes_normalized = resize_boxes(boxes_normalized=boxes_normalized, size_ratio=box_size_ratio)
        if verbose:
            print(f'Resized boxes with ratio {box_size_ratio}')

    # unnormalized box coordinates
    unnormalized_boxes = np.array(boxes_normalized) * size
    unnormalized_boxes = unnormalized_boxes.astype(int)
    # clip to image size
    unnormalized_boxes[:, 0] = np.clip(unnormalized_boxes[:, 0], 0, size)
    unnormalized_boxes[:, 1] = np.clip(unnormalized_boxes[:, 1], 0, size)
    unnormalized_boxes[:, 2] = np.clip(unnormalized_boxes[:, 2], 0, size)
    unnormalized_boxes[:, 3] = np.clip(unnormalized_boxes[:, 3], 0, size)

    # print('boxes_normalized', boxes_normalized)
    # print('unnormalized_boxes', unnormalized_boxes)
    
    n_total_boxes = len(box_captions)

    context_imgs = []
    mask_imgs = []
    generated_images = []
    prompts = []

    if initial_image is None:
        context_img = Image.new('RGB', (size, size))
    else:
        context_img = initial_image.copy()

    if verbose:
        print('Initiailzed context image')


    if last_background_inpaint:
        background_mask_img = create_binary_mask_from_layouts(
            boxes_normalized=[],
            size=size,
        )
        background_mask_draw = ImageDraw.Draw(background_mask_img)
        background_mask_draw.rectangle([(0, 0), background_mask_img.size], fill=255)

    if parallel_foreground:
        # generate all objects at once

        target_caption = " and ".join(box_captions)

        mask_img = Image.new('L', (size,size))

        for i in range(n_total_boxes):
            target_box_normalized = boxes_normalized[i]
            target_box_unnormalized = unnormalized_boxes[i]

            _mask_img = create_binary_mask_from_layouts(
                boxes_normalized=[target_box_normalized],
                size=size,
            )
            mask_img = Image.composite(_mask_img, mask_img, _mask_img)

            if last_background_inpaint:
                # background_mask_draw.rectangle(box.long().tolist(), fill=0)
                background_mask_draw.rectangle(target_box_unnormalized.tolist(), fill=0)

        prompt = f"{target_caption}"
        if use_global_caption:
            prompt = f"{global_caption}; Add {prompt}"
        if verbose:
            print('prompt:', prompt)
        prompts += [prompt]

        generated_image = inpaint_from_layout(
            pipe=pipe,
            model=model,
            image=context_img,
            mask_image=mask_img,
            prompt=prompt,
            box_captions=[target_caption],
            boxes_normalized=[target_box_normalized],
            height=size,
            width=size,
            strength=strength,
            num_inference_steps=num_inference_steps,
        )
        
        if paste:
            # box = target_box_unnormalized
            # src_box = box.tolist()

            # # x1 -> x1 + 1
            # # y1 -> y1 + 1
            # paste_box = box.tolist()
            # paste_box[0] -= 1
            # paste_box[1] -= 1
            # paste_box[2] += 1
            # paste_box[3] += 1

            # box_w = paste_box[2] - paste_box[0]
            # box_h = paste_box[3] - paste_box[1]

            # context_img.paste(generated_image.crop(src_box).resize((box_w, box_h)), paste_box)
            # generated_images.append(context_img.copy())

            # mask composition
            generated_image = Image.composite(generated_image, context_img, mask_img)
            context_img = generated_image.copy()
            # print('new composition')
            generated_images.append(generated_image.copy())
        else:
            context_img = generated_image
            generated_images.append(context_img.copy())

    else:

        for i in range(n_total_boxes):
            if verbose:
                print('Iter: ', i+1, 'total: ', n_total_boxes)

            target_caption = box_captions[i]
            target_box_normalized = boxes_normalized[i]
            target_box_unnormalized = unnormalized_boxes[i]

            mask_img = create_binary_mask_from_layouts(
                boxes_normalized=[target_box_normalized],
                size=size,
            )

            if last_background_inpaint:
                # background_mask_draw.rectangle(box.long().tolist(), fill=0)
                background_mask_draw.rectangle(target_box_unnormalized.tolist(), fill=0)

            if verbose:
                print('Drawing', target_caption)

            mask_imgs.append(mask_img.copy())

            # prompt = f"Add {d['box_captions'][i]}"
            # prompt = f"Add {target_caption}"
            prompt = f"{target_caption}"

            if use_global_caption:
                prompt = f"{global_caption}; Add {prompt}"
                # print('prompt:', prompt)

            if verbose:
                print('prompt:', prompt)
            prompts += [prompt]

            context_imgs.append(context_img.copy())

            generated_image = inpaint_from_layout(
                pipe=pipe,
                model=model,
                image=context_img,
                mask_image=mask_img,
                prompt=prompt,
                box_captions=[target_caption],
                boxes_normalized=[target_box_normalized],
                height=size,
                width=size,
                strength=strength,
                num_inference_steps=num_inference_steps,
            )
            
            if paste:
                # box = target_box_unnormalized
                # src_box = box.tolist()

                # # x1 -> x1 + 1
                # # y1 -> y1 + 1
                # paste_box = box.tolist()
                # paste_box[0] -= 1
                # paste_box[1] -= 1
                # paste_box[2] += 1
                # paste_box[3] += 1

                # box_w = paste_box[2] - paste_box[0]
                # box_h = paste_box[3] - paste_box[1]

                # context_img.paste(generated_image.crop(src_box).resize((box_w, box_h)), paste_box)
                # generated_images.append(context_img.copy())

                # mask composition
                generated_image = Image.composite(generated_image, context_img, mask_img)
                context_img = generated_image.copy()
                # print('new composition')
                generated_images.append(generated_image.copy())
            else:
                context_img = generated_image
                generated_images.append(context_img.copy())

    if last_background_inpaint:
        if verbose:
            print('Fill background')

        mask_img = background_mask_img

        mask_imgs.append(mask_img)

        prompt = 'Add background'

        # if use_global_caption:
        #     prompt = f"{global_caption}; {prompt}"

        if verbose:
            print('prompt:', prompt)
        prompts += [prompt]

        generated_image = pipe(
            prompt,
            context_img,
            mask_img,
            # guidance_scale=guidance_scale
            num_inference_steps=num_inference_steps
            ).images[0]

        generated_images.append(generated_image)
        
    return {
        'context_imgs': context_imgs,
        'mask_imgs': mask_imgs,
        'prompts': prompts,
        'generated_images': generated_images,
    }