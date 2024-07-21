

def encode_scene(obj_list, H=320, W=320, src_bbox_format='xywh', tgt_bbox_format='xyxy'):
    """Encode scene into text and bounding boxes
    Args:
        obj_list: list of dicts
            Each dict has keys:
                
                'color': str
                'material': str
                'shape': str
                or 
                'caption': str

                and

                'bbox': list of 4 floats (unnormalized)
                    [x0, y0, x1, y1] or [x0, y0, w, h]
    """
    box_captions = []
    for obj in obj_list:
        if 'caption' in obj:
            box_caption = obj['caption']
        else:
            box_caption = f"{obj['color']} {obj['material']} {obj['shape']}"
        box_captions += [box_caption]
    
    assert src_bbox_format in ['xywh', 'xyxy'], f"src_bbox_format must be 'xywh' or 'xyxy', not {src_bbox_format}"
    assert tgt_bbox_format in ['xywh', 'xyxy'], f"tgt_bbox_format must be 'xywh' or 'xyxy', not {tgt_bbox_format}"

    boxes_unnormalized = []
    boxes_normalized = []
    for obj in obj_list:
        if src_bbox_format == 'xywh':
            x0, y0, w, h = obj['bbox']
            x1 = x0 + w
            y1 = y0 + h
        elif src_bbox_format == 'xyxy':
            x0, y0, x1, y1 = obj['bbox']
            w = x1 - x0
            h = y1 - y0
        assert x1 > x0, f"x1={x1} <= x0={x0}"
        assert y1 > y0, f"y1={y1} <= y0={y0}"
        assert x1 <= W, f"x1={x1} > W={W}"
        assert y1 <= H, f"y1={y1} > H={H}"

        if tgt_bbox_format == 'xywh':
            bbox_unnormalized = [x0, y0, w, h]
            bbox_normalized = [x0 / W, y0 / H, w / W, h / H]

        elif tgt_bbox_format == 'xyxy':
            bbox_unnormalized = [x0, y0, x1, y1]
            bbox_normalized = [x0 / W, y0 / H, x1 / W, y1 / H]
            
        boxes_unnormalized += [bbox_unnormalized]
        boxes_normalized += [bbox_normalized]

    assert len(box_captions) == len(boxes_normalized), f"len(box_captions)={len(box_captions)} != len(boxes_normalized)={len(boxes_normalized)}"
        
        
    # text = prepare_text(box_captions, boxes_normalized)
    
    out = {}
    # out['text'] = text
    out['box_captions'] = box_captions
    out['boxes_normalized'] = boxes_normalized
    out['boxes_unnormalized'] = boxes_unnormalized
        
    return out

def encode_from_custom_annotation(custom_annotations, size=512):
    #     custom_annotations = [
    #     {'x': 83, 'y': 335, 'width': 70, 'height': 69, 'label': 'blue metal cube'},
    #     {'x': 162, 'y': 302, 'width': 110, 'height': 138, 'label': 'blue metal cube'},
    #     {'x': 274, 'y': 250, 'width': 191, 'height': 234, 'label': 'blue metal cube'},
    #     {'x': 14, 'y': 18, 'width': 155, 'height': 205, 'label': 'blue metal cube'},
    #     {'x': 175, 'y': 79, 'width': 106, 'height': 119, 'label': 'blue metal cube'},
    #     {'x': 288, 'y': 111, 'width': 69, 'height': 63, 'label': 'blue metal cube'}
    # ]
    H, W = size, size

    objects = []
    for j in range(len(custom_annotations)):
        xyxy = [
            custom_annotations[j]['x'],
            custom_annotations[j]['y'],
            custom_annotations[j]['x'] + custom_annotations[j]['width'],
            custom_annotations[j]['y'] + custom_annotations[j]['height']]
        objects.append({
            'caption': custom_annotations[j]['label'],
            'bbox': xyxy,
        })

    out = encode_scene(objects, H=H, W=W,
                       src_bbox_format='xyxy', tgt_bbox_format='xyxy')

    return out



def create_reco_prompt(
    caption: str = '',
    phrases=[],
    boxes=[],
    normalize_boxes=True,
    image_resolution=512,
    num_bins=1000,
    ):
    """
    method to create ReCo prompt

    caption: global caption
    phrases: list of regional captions
    boxes: list of regional coordinates (unnormalized xyxy)
    """

    SOS_token = '<|startoftext|>'
    EOS_token = '<|endoftext|>'
    
    box_captions_with_coords = []
    
    box_captions_with_coords += [caption]
    box_captions_with_coords += [EOS_token]

    for phrase, box in zip(phrases, boxes):
                    
        if normalize_boxes:
            box = [float(x) / image_resolution for x in box]

        # quantize into bins
        quant_x0 = int(round((box[0] * (num_bins - 1))))
        quant_y0 = int(round((box[1] * (num_bins - 1))))
        quant_x1 = int(round((box[2] * (num_bins - 1))))
        quant_y1 = int(round((box[3] * (num_bins - 1))))
        
        # ReCo format
        # Add SOS/EOS before/after regional captions
        box_captions_with_coords += [
            f"<bin{str(quant_x0).zfill(3)}>",
            f"<bin{str(quant_y0).zfill(3)}>",
            f"<bin{str(quant_x1).zfill(3)}>",
            f"<bin{str(quant_y1).zfill(3)}>",
            SOS_token,
            phrase,
            EOS_token
        ]

    text = " ".join(box_captions_with_coords)
    return text


# caption = "a photo of bus and boat; boat is left to bus."
# phrases = ["a photo of a bus.", "a photo of a boat."]
# boxes =  [[0.702, 0.404, 0.927, 0.601], [0.154, 0.383, 0.311, 0.487]]
# prompt = create_reco_prompt(caption, phrases, boxes, normalize_boxes=False)
# prompt
# >>> 'a photo of bus and boat; boat is left to bus. <|endoftext|> <bin701> <bin404> <bin926> <bin600> <|startoftext|> a photo of a bus. <|endoftext|> <bin154> <bin383> <bin311> <bin487> <|startoftext|> a photo of a boat. <|endoftext|>'


# caption = "A box contains six donuts with varying types of glazes and toppings."
# phrases = ["chocolate donut.", "dark vanilla donut.", "donut with sprinkles.", "donut with powdered sugar.", "pink donut.", "brown donut."]
# boxes = [[263.68, 294.912, 380.544, 392.832], [121.344, 265.216, 267.392, 401.92], [391.168, 294.912, 506.368, 381.952], [120.064, 143.872, 268.8, 270.336], [264.192, 132.928, 393.216, 263.68], [386.048, 148.48, 490.688, 259.584]]
# prompt = create_reco_prompt(caption, phrases, boxes)
# prompt
# >>> 'A box contains six donuts with varying types of glazes and toppings. <|endoftext|> <bin514> <bin575> <bin743> <bin766> <|startoftext|> chocolate donut. <|endoftext|> <bin237> <bin517> <bin522> <bin784> <|startoftext|> dark vanilla donut. <|endoftext|> <bin763> <bin575> <bin988> <bin745> <|startoftext|> donut with sprinkles. <|endoftext|> <bin234> <bin281> <bin524> <bin527> <|startoftext|> donut with powdered sugar. <|endoftext|> <bin515> <bin259> <bin767> <bin514> <|startoftext|> pink donut. <|endoftext|> <bin753> <bin290> <bin957> <bin506> <|startoftext|> brown donut. <|endoftext|>'



