import os
import torch
import json
import argparse
from model import CANNet
import dataset
from torchvision import transforms

parser = argparse.ArgumentParser(description='Test the best CANNet model on a dataset')
parser.add_argument('test_json', metavar='TEST', help='path to test json file')
parser.add_argument('--model_path', type=str, default='model_best.pth.tar', help='path to saved model checkpoint')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for testing')

def main():
    args = parser.parse_args()
    
    # Detect hardware
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Testing on device: {device}")

    # Load the test list
    with open(args.test_json, 'r') as outfile:
        test_list = json.load(outfile)

    # Initialize the model
    model = CANNet()
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        
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
        for i, (img, target) in enumerate(test_loader):
            img = img.to(device)
            target = target.type(torch.FloatTensor).to(device)
            
            # Replicate the quadrant splitting used in training for memory safety
            h, w = img.shape[2:4]
            h_d = h // 2
            w_d = w // 2
            
            img_1 = img[:, :, :h_d, :w_d]
            img_2 = img[:, :, :h_d, w_d:]
            img_3 = img[:, :, h_d:, :w_d]
            img_4 = img[:, :, h_d:, w_d:]
            
            density_1 = model(img_1).data.cpu().numpy()
            density_2 = model(img_2).data.cpu().numpy()
            density_3 = model(img_3).data.cpu().numpy()
            density_4 = model(img_4).data.cpu().numpy()
            
            pred_sum = density_1.sum() + density_2.sum() + density_3.sum() + density_4.sum()
            mae += abs(pred_sum - target.sum().item())
            
    final_mae = mae / len(test_loader)
    print(f"\n======================================")
    print(f"Final Test MAE: {final_mae:.3f}")
    print(f"======================================\n")

if __name__ == '__main__':
    main()