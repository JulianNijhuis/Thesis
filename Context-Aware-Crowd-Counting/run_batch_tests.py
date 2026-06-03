import os
import sys
import json
import torch
import numpy as np
from torchvision import transforms
from model import CANNet, CANNet_GRL_Frontend, CANNet_CORAL_Frontend
import dataset

def evaluate_model(model_path, model_class, grl_location="none", test_json="gwhd_2021/test.json"):
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    if not os.path.exists(model_path):
        print(f"Warning: Model checkpoint not found at {model_path}")
        return None, None

    # Initialize model
    model = model_class()
    
    # Load weights
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    except Exception as e:
        print(f"Error loading weights from {model_path}: {e}")
        return None, None

    model = model.to(device)
    model.eval()

    # Load test dataset
    with open(test_json, 'r') as f:
        test_list = json.load(f)

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

    mae = 0.0
    rmse_sum = 0.0

    with torch.no_grad():
        for i, (img, target, _) in enumerate(test_loader):
            img = img.to(device)
            target = target.type(torch.FloatTensor).to(device)

            is_baseline = (model_class == CANNet)
            
            if is_baseline:
                # Full image pass for the baseline CACC model to avoid edge/padding artifacts
                res = model(img)
                pred_sum = res.sum().item()
            else:
                # Quadrant splitting for GRL and CORAL models as they are scale-aware
                h, w = img.shape[2:4]
                h_d = h // 2
                w_d = w // 2
                
                img_1 = img[:, :, :h_d, :w_d]
                img_2 = img[:, :, :h_d, w_d:]
                img_3 = img[:, :, h_d:, :w_d]
                img_4 = img[:, :, h_d:, w_d:]
                
                res_1 = model(img_1, alpha=1.0) if grl_location != 'none' else model(img_1)
                res_2 = model(img_2, alpha=1.0) if grl_location != 'none' else model(img_2)
                res_3 = model(img_3, alpha=1.0) if grl_location != 'none' else model(img_3)
                res_4 = model(img_4, alpha=1.0) if grl_location != 'none' else model(img_4)
    
                # Check if output is a tuple (e.g. GRL returning (out, domain_logits) or CORAL returning (out, features))
                density_1 = res_1[0] if isinstance(res_1, tuple) else res_1
                density_2 = res_2[0] if isinstance(res_2, tuple) else res_2
                density_3 = res_3[0] if isinstance(res_3, tuple) else res_3
                density_4 = res_4[0] if isinstance(res_4, tuple) else res_4
                
                pred_sum = (density_1.sum() + density_2.sum() + density_3.sum() + density_4.sum()).item()
                
            target_sum = target.sum().item()
            err = pred_sum - target_sum
            mae += abs(err)
            rmse_sum += err ** 2

            # Print real-time progress update
            total_imgs = len(test_loader)
            if (i + 1) % 5 == 0 or i == total_imgs - 1:
                progress = (i + 1) / total_imgs * 100
                sys.stdout.write(f"\r  -> Progress: [{i + 1}/{total_imgs}] ({progress:.1f}%) | Current MAE: {mae / (i + 1):.4f}")
                sys.stdout.flush()

    print("") # Print newline after loop completes
    final_mae = mae / len(test_loader)
    final_rmse = np.sqrt(rmse_sum / len(test_loader))
    return final_mae, final_rmse

def main():
    print("==================================================")
    print("Starting Batch Test Evaluation on GWHD 2021...")
    print("==================================================")

    # 3 core models for the thesis experiments
    models_to_test = [
        {
            "key": "baseline",
            "name": "Baseline CACC",
            "path": "model_best.pth.tar",
            "class": CANNet,
            "grl": "none"
        },
        {
            "key": "grl_frontend",
            "name": "GRL Frontend",
            "path": "model_best_grl_frontend_fixed.pth.tar",
            "class": CANNet_GRL_Frontend,
            "grl": "frontend"
        },
        {
            "key": "coral_frontend",
            "name": "DeepCORAL Frontend",
            "path": "model_best_coralfrontend.pth.tar",
            "class": CANNet_CORAL_Frontend,
            "grl": "none"
        }
    ]

    results = {}
    
    for m in models_to_test:
        print(f"Evaluating model: {m['name']} ({m['path']})...")
        mae, rmse = evaluate_model(m["path"], m["class"], grl_location=m["grl"])
        if mae is not None:
            results[m["key"]] = {
                "name": m["name"],
                "mae": float(mae),
                "rmse": float(rmse)
            }
            print(f"-> Result for {m['name']}: Average MAE = {mae:.4f} | Average RMSE = {rmse:.4f}\n")
            
    # Save results as JSON
    output_file = "test_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print("==================================================")
    print(f"Batch test evaluation complete!")
    print(f"Results saved to '{output_file}':")
    print(json.dumps(results, indent=4))
    print("==================================================")

if __name__ == '__main__':
    main()
