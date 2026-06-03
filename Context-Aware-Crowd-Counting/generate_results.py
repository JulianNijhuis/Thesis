import os
import json
import numpy as np

def parse_histories():
    results = []
    
    # 1. Parse root JSON files
    root_files = {
        'training_history.json': 'GRL Concat (100 Epochs, Lambda 0.0001)',
        'training_history_coralfrontend.json': 'DeepCORAL Frontend (100 Epochs, Lambda 0.0001)',
        'training_historyL0.1GRLFrontEnd.json': 'GRL Frontend (1 Epoch, Lambda 0.1)'
    }
    
    for filename, name in root_files.items():
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                try:
                    data = json.load(f)
                    val_mae = data.get('val_mae', [])
                    if len(val_mae) > 0:
                        best_mae = min(val_mae)
                        best_epoch = val_mae.index(best_mae) + 1
                        final_mae = val_mae[-1]
                        results.append({
                            'source': 'Root',
                            'name': name,
                            'epochs': len(val_mae),
                            'best_mae': best_mae,
                            'best_epoch': best_epoch,
                            'final_mae': final_mae
                        })
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
                    
    # 2. Parse experiment_results directory
    exp_dir = 'experiment_results'
    if os.path.exists(exp_dir):
        for filename in os.listdir(exp_dir):
            if filename.endswith('.json') and filename.startswith('training_history_'):
                filepath = os.path.join(exp_dir, filename)
                with open(filepath, 'r') as f:
                    try:
                        data = json.load(f)
                        val_mae = data.get('val_mae', [])
                        if len(val_mae) > 0:
                            # Extract experiment details from filename
                            # e.g., training_history_concat_lam_0.0001.json
                            parts = filename.replace('training_history_', '').replace('.json', '').split('_lam_')
                            loc = parts[0]
                            lam = parts[1] if len(parts) > 1 else 'N/A'
                            
                            best_mae = min(val_mae)
                            best_epoch = val_mae.index(best_mae) + 1
                            final_mae = val_mae[-1]
                            
                            # Standardize name
                            name_str = f"GRL {loc.capitalize()} (Lambda: {lam})"
                            if loc == 'none':
                                name_str = "Baseline CACC (No GRL)"
                                
                            results.append({
                                'source': 'Experiment Grid',
                                'name': name_str,
                                'location': loc,
                                'lambda': lam,
                                'epochs': len(val_mae),
                                'best_mae': best_mae,
                                'best_epoch': best_epoch,
                                'final_mae': final_mae
                            })
                    except Exception as e:
                        print(f"Error reading {filepath}: {e}")
                        
    # Sort results
    results_root = [r for r in results if r['source'] == 'Root']
    results_grid = [r for r in results if r['source'] == 'Experiment Grid']
    
    results_root.sort(key=lambda x: x['best_mae'])
    results_grid.sort(key=lambda x: x['best_mae'])
    
    print("\n=== LONG 100-EPOCH RUNS ===")
    print(f"{'Experiment Model':<55} | {'Epochs':<6} | {'Best MAE':<10} | {'At Epoch':<8} | {'Final MAE':<10}")
    print("-" * 98)
    for r in results_root:
        print(f"{r['name']:<55} | {r['epochs']:<6} | {r['best_mae']:<10.4f} | {r['best_epoch']:<8} | {r['final_mae']:<10.4f}")
        
    print("\n=== 5-EPOCH EXPERIMENT SWEEP GRID ===")
    print(f"{'Experiment Model':<35} | {'Epochs':<6} | {'Best MAE':<10} | {'At Epoch':<8} | {'Final MAE':<10}")
    print("-" * 78)
    for r in results_grid:
        print(f"{r['name']:<35} | {r['epochs']:<6} | {r['best_mae']:<10.4f} | {r['best_epoch']:<8} | {r['final_mae']:<10.4f}")

if __name__ == '__main__':
    parse_histories()
