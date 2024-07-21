from pathlib import Path
import json
import argparse

layout_fnames = [
 'combinations_layouts_all.json',
#  'combinations_layouts_common.json',
#  'combinations_layouts_uncommon.json',

 'count_layouts_all.json',    
#  'count_layouts_2_4.json',
#  'count_layouts_5_7.json',
#  'count_layouts_8_10.json',

    
 'position_layouts_all.json',
#  'position_layouts_boundary.json',
#  'position_layouts_center.json',
    
 'size_layouts_all.json',
#  'size_layouts_large.json',
#  'size_layouts_small.json'
]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='reco')
    parser.add_argument('--layout_data_dir', type=str, default='../image_generation/data/layoutbench-coco',
                        help='Directory containing LayoutBench-COCO layouts')
    parser.add_argument('--image_dump_base_dir', type=str, default='../image_generation/images/',
                        help='Directory containing images generated rom LayoutBench-COCO layouts'
                        )
    parser.add_argument('--image_list_dump_dir', type=str, default='./image_fnames_dir/',
                        help='Directory to save image filepaths for YOLO v7'
                        )

    args = parser.parse_args()
    print(args)

    model_name = args.model_name
    layout_data_dir = Path(args.layout_data_dir)
    image_dump_base_dir = Path(args.image_dump_base_dir)
    image_list_dump_dir = Path(args.image_list_dump_dir)
    
    print("model:", model_name)
    for layout_fname in layout_fnames:
        print("layout_fname", layout_fname)
        image_filepaths = []
        
        # 1) Read layout data
        layout_data = json.load(open(layout_data_dir / layout_fname))
        for d in layout_data:
            
            # Check if image exists
            image_fpath = image_dump_base_dir / model_name / "all_layouts" / d['file_name']
            assert Path(image_fpath).exists(), f"Image not found: {image_fpath}"
            image_filepaths += [str(image_fpath)]
        print(len(image_filepaths))
        
        # 2) Save image filepaths for YOLO v7
        save_path = image_list_dump_dir / f"{model_name}_{layout_fname}"
        save_path = save_path.with_suffix('.txt')
        print('Image filepaths saved at', save_path)
        with open(save_path, 'w') as f:
            for img_fname in image_filepaths:
                f.write(img_fname+'\n')
