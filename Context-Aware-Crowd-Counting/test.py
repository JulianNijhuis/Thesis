import os
import torch
import json
import argparse
from model import CANNet, CANNet_GRL_Frontend, CANNet_GRL_Context, CANNet_GRL_Concat
import dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Test the best CANNet model on a dataset')
parser.add_argument('test_json', metavar='TEST', help='path to test json file')
parser.add_argument('--model_path', type=str, default='model_best.pthGRlFrontEnd.tar', help='path to saved model checkpoint')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for testing')
parser.add_argument('--grl_location', type=str, default='none', choices=['none', 'frontend', 'context', 'concat'], help='Location for Gradient Reversal Layer')


def main():
    args = parser.parse_args()

    # Detect hardware
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Testing on device: {device}")

    # Load the test list
    with open(args.test_json, 'r') as outfile:
        test_list = json.load(outfile)

    # Initialize the model
    if args.grl_location == 'frontend':
        model = CANNet_GRL_Frontend()
    elif args.grl_location == 'context':
        model = CANNet_GRL_Context()
    elif args.grl_location == 'concat':
        model = CANNet_GRL_Concat()
    else:
        model = CANNet()
    checkpoint = torch.load(args.model_path, map_location=device)

    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint, strict=False)

    model = model.to(device)
    model.eval()

    # Setup the data loader
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(test_list,
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                            ]),
                            train=False),
        batch_size=args.batch_size)

    mae = 0.0

    print(f"Starting evaluation on {len(test_loader)} images...")

    if len(test_loader) == 0:
        print("Error: The test dataset is empty! Please check your JSON file.")
        return

    with torch.no_grad():
        for i, (img, target, _) in enumerate(test_loader):
            img = img.to(device)
            target = target.type(torch.FloatTensor).to(device)

            # The Contextual Module is scale-dependent, so we MUST split into quadrants
            # to match the training crop size, otherwise the spatial pyramid produces noise.
            h, w = img.shape[2:4]
            h_d = h // 2
            w_d = w // 2
            
            img_1 = img[:, :, :h_d, :w_d]
            img_2 = img[:, :, :h_d, w_d:]
            img_3 = img[:, :, h_d:, :w_d]
            img_4 = img[:, :, h_d:, w_d:]
            
            if args.grl_location != 'none':
                density_1, _ = model(img_1, alpha=1.0)
                density_2, _ = model(img_2, alpha=1.0)
                density_3, _ = model(img_3, alpha=1.0)
                density_4, _ = model(img_4, alpha=1.0)
            else:
                density_1 = model(img_1)
                density_2 = model(img_2)
                density_3 = model(img_3)
                density_4 = model(img_4)
                
            density_1 = density_1.data.cpu().numpy()
            density_2 = density_2.data.cpu().numpy()
            density_3 = density_3.data.cpu().numpy()
            density_4 = density_4.data.cpu().numpy()
            
            pred_sum = density_1.sum() + density_2.sum() + density_3.sum() + density_4.sum()
            mae += abs(pred_sum - target.sum().item())
            
            if i < 5:
                h_out_d, w_out_d = density_1.shape[2:]
                pred_map = np.zeros((h_out_d * 2, w_out_d * 2))
                pred_map[:h_out_d, :w_out_d] = density_1[0, 0]
                pred_map[:h_out_d, w_out_d:] = density_2[0, 0]
                pred_map[h_out_d:, :w_out_d] = density_3[0, 0]
                pred_map[h_out_d:, w_out_d:] = density_4[0, 0]

                img_np = img[0].cpu().numpy().transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = std * img_np + mean
                img_np = np.clip(img_np, 0, 1)

                target_map = target[0].cpu().numpy()

                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.imshow(img_np)
                plt.title('Original Image')
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.imshow(target_map, cmap='jet')
                plt.title(f'Ground Truth (Count: {target.sum().item():.1f})')
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.imshow(pred_map, cmap='jet')
                plt.title(f'Prediction (Count: {pred_sum:.1f})')
                plt.axis('off')

                plt.tight_layout(pad=1.5)
                plt.savefig(f'test_result_{i}.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved visualization for image {i}")

    final_mae = mae / len(test_loader)
    print(f"\n======================================")
    print(f"Final Test MAE: {final_mae:.3f}")
    print(f"======================================\n")


if __name__ == '__main__':
    main()