import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from model import CANNet, CANNet_GRL_Frontend, CANNet_CORAL_Frontend
import dataset

def load_model(model_path, model_class, device):
    if not os.path.exists(model_path):
        print(f"Warning: Model checkpoint not found at {model_path}")
        return None
    model = model_class()
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None
    model = model.to(device)
    model.eval()
    return model

def main():
    print("==================================================")
    print("Generating Thesis Visualizations (Baseline Branch Style)...")
    print("==================================================")

    # Detect hardware
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load test set paths
    with open('path/to/test.json', 'r') as f:
        test_list = json.load(f)

    # Load test csv for bounding box dots
    csv_path = 'path/to/test.csv'
    if not os.path.exists(csv_path):
        print(f"Error: test.csv not found at {csv_path}")
        return
    df = pd.read_csv(csv_path)
    
    # Map image name to bounding boxes
    bbox_dict = {}
    for _, row in df.iterrows():
        bbox_dict[row['image_name']] = row['BoxesString']

    # Load the three core models
    model_baseline = load_model("path/to/model_best.pth.tar", CANNet, device)
    model_grl = load_model("path/to/model_best_grl.pth.tar", CANNet_GRL_Frontend, device)
    model_coral = load_model("path/to/model_best_coral.pth.tar", CANNet_CORAL_Frontend, device)

    # Setup the data loader exactly like test.py on baseline branch
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(test_list,
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                            ]),
                            train=False),
        batch_size=1)

    num_images_to_visualize = min(50, len(test_loader))
    
    with torch.no_grad():
        for i, (img, target, _) in enumerate(test_loader):
            if i >= num_images_to_visualize:
                break
                
            img_path = test_list[i]
            filename = os.path.basename(img_path)
            print(f"Processing image {i+1}/{num_images_to_visualize}: {filename}...")

            img = img.to(device)
            target = target.type(torch.FloatTensor).to(device)

            # --- Forward Pass: Baseline CACC (Full Image Single Pass) ---
            if model_baseline:
                d_baseline = model_baseline(img)
                pred_baseline = d_baseline[0, 0].cpu().numpy()
                count_baseline = pred_baseline.sum()
            else:
                pred_baseline = None
                count_baseline = 0.0

            # --- Forward Pass: GRL Frontend (Full Image Single Pass) ---
            if model_grl:
                res_grl = model_grl(img, alpha=1.0)
                d_grl = res_grl[0] if isinstance(res_grl, tuple) else res_grl
                pred_grl = d_grl[0, 0].cpu().numpy()
                count_grl = pred_grl.sum()
            else:
                pred_grl = None
                count_grl = 0.0

            # --- Forward Pass: CORAL Frontend (Full Image Single Pass) ---
            if model_coral:
                res_coral = model_coral(img)
                d_coral = res_coral[0] if isinstance(res_coral, tuple) else res_coral
                pred_coral = d_coral[0, 0].cpu().numpy()
                count_coral = pred_coral.sum()
            else:
                pred_coral = None
                count_coral = 0.0

            # Extract ground-truth density map from data loader
            target_map = target[0].cpu().numpy()

            # Retrieve bounding box centers for dots
            centers = []
            boxes_str = bbox_dict.get(filename, '')
            if pd.notna(boxes_str) and str(boxes_str).strip() != "no_box" and str(boxes_str).strip() != "":
                boxes = str(boxes_str).split(';')
                for box in boxes:
                    if box.strip():
                        # Format: "xmin ymin xmax ymax"
                        xmin, ymin, xmax, ymax = map(float, box.strip().split(' '))
                        cx = (xmin + xmax) / 2
                        cy = (ymin + ymax) / 2
                        centers.append((cx, cy))
            
            # Ground truth count
            count_gt = len(centers) if len(centers) > 0 else float(target_map.sum())

            # Un-normalize the image tensor for plotting, exactly like test.py on baseline branch
            img_np = img[0].cpu().numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)

            # Generate the beautiful 5-panel figure
            fig, axes = plt.subplots(1, 5, figsize=(25, 5))
            
            # Subplot 1: Original Image with Red Dots
            axes[0].imshow(img_np)
            for cx, cy in centers:
                circle = patches.Circle((cx, cy), radius=10, edgecolor='red', facecolor='none', linewidth=1.5)
                axes[0].add_patch(circle)
                axes[0].plot(cx, cy, 'ro', markersize=3)
            axes[0].set_title(f'Original (Annotated Count: {count_gt})', fontsize=12, fontweight='bold')
            axes[0].axis('off')

            # We set a consistent, robust color limit matching model prediction peaks
            vmax = 0.03

            # Subplot 2: Ground Truth Density Map
            axes[1].imshow(target_map, cmap='jet', vmin=0, vmax=vmax)
            axes[1].set_title(f'Ground Truth Density Map', fontsize=12, fontweight='bold')
            axes[1].axis('off')

            # Subplot 3: Baseline Prediction
            if pred_baseline is not None:
                axes[2].imshow(pred_baseline, cmap='jet', vmin=0, vmax=vmax)
                axes[2].set_title(f'Baseline CACC (Count: {count_baseline:.1f})', fontsize=12, fontweight='bold')
            else:
                axes[2].text(0.5, 0.5, 'Model Missing', ha='center', va='center')
            axes[2].axis('off')

            # Subplot 4: GRL Frontend Prediction
            if pred_grl is not None:
                axes[3].imshow(pred_grl, cmap='jet', vmin=0, vmax=vmax)
                axes[3].set_title(f'GRL Frontend (Count: {count_grl:.1f})', fontsize=12, fontweight='bold')
            else:
                axes[3].text(0.5, 0.5, 'Model Missing', ha='center', va='center')
            axes[3].axis('off')

            # Subplot 5: DeepCORAL Frontend Prediction
            if pred_coral is not None:
                axes[4].imshow(pred_coral, cmap='jet', vmin=0, vmax=vmax)
                axes[4].set_title(f'DeepCORAL Frontend (Count: {count_coral:.1f})', fontsize=12, fontweight='bold')
            else:
                axes[4].text(0.5, 0.5, 'Model Missing', ha='center', va='center')
            axes[4].axis('off')

            plt.tight_layout(pad=2.0)
            output_name = f'thesis_comparison_image_{i}.png'
            plt.savefig(output_name, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Successfully saved multi-panel visualization to '{output_name}'")

    print("\n==================================================")
    print("All comparison visualizations generated successfully!")
    print("==================================================")

if __name__ == '__main__':
    main()
