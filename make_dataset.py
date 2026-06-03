import h5py
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
import argparse

def generate_density_maps(csv_file, img_folder, output_folder):
    print(f"Processing annotations from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for index, row in df.iterrows():
        img_name = row['image_name']
        boxes_string = row['BoxesString']
        
        img_path = os.path.join(img_folder, img_name)
        if not os.path.exists(img_path):
            print(f"Warning: Image not found at {img_path}")
            continue
            
        try:
            img = plt.imread(img_path)
        except Exception as e:
            print(f"Warning: Failed to read image {img_name}. Error: {e}. Skipping.")
            continue
        
        # Create a zero matrix of the same spatial dimensions as the image
        k = np.zeros((img.shape[0], img.shape[1]))
        
        # Check if there are boxes (handle 'no_box' or NaN values)
        if pd.notna(boxes_string) and str(boxes_string).strip() != "no_box":
            boxes = str(boxes_string).split(';')
            for box in boxes:
                if not box.strip():
                    continue
                    
                # Format is typically "xmin ymin xmax ymax"
                xmin, ymin, xmax, ymax = map(float, box.strip().split(' '))
                
                # Calculate center point of the wheat head
                cx = int((xmin + xmax) / 2)
                cy = int((ymin + ymax) / 2)
                
                # Set the center point to 1 (making sure it's inside image bounds)
                if cy < img.shape[0] and cx < img.shape[1]:
                    k[cy, cx] = 1
                    
        # Apply Gaussian filter to blur the points into a density map
        k = gaussian_filter(k, 15)
        
        # Save to hdf5 file
        h5_name = img_name.rsplit('.', 1)[0] + '.h5'
        h5_path = os.path.join(output_folder, h5_name)
        
        with h5py.File(h5_path, 'w') as hf:
            hf['density'] = k
            
    print(f"Finished generating density maps in {output_folder}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate h5 density maps from GWHD csv files.")
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the train/val/test CSV file')
    parser.add_argument('--img_folder', type=str, required=True, help='Folder containing the images')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save the .h5 files')
    
    args = parser.parse_args()
    generate_density_maps(args.csv_file, args.img_folder, args.output_folder)
