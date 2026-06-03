import json
import os
import glob
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a JSON file with image paths for the dataset.")
    parser.add_argument('--h5_folder', type=str, required=True, help='Path to the folder containing the generated .h5 files')
    parser.add_argument('--img_folder', type=str, required=True, help='Path to the folder containing the dataset images')
    parser.add_argument('--output_json', type=str, default='img.json', help='Path to the final output json file')
    parser.add_argument('--ext', type=str, default='png', help='Image extension (e.g. jpg, png)')
    args = parser.parse_args()

    img_list = []

    for h5_path in glob.glob(os.path.join(args.h5_folder, '*.h5')):
        base_name = os.path.basename(h5_path)
        img_name = base_name.replace('.h5', f'.{args.ext}')
        img_path = os.path.join(args.img_folder, img_name)
        
        if os.path.exists(img_path):
            img_list.append(img_path)

    with open(args.output_json, 'w') as f:
        json.dump(img_list,f)
        
    print(f"Successfully saved {len(img_list)} image paths to {args.output_json}")
