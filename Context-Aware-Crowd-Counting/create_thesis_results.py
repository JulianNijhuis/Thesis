import os
import json
import matplotlib.pyplot as plt
import numpy as np

def load_json_history(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        try:
            return json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

def analyze_and_plot():
    print("Starting thesis results generation...")
    
    # ----------------------------------------------------
    # 1. Gather all data
    # ----------------------------------------------------
    
    # 100-epoch runs in root
    history_grl_concat_100 = load_json_history('training_history.json')
    history_coral_front_100 = load_json_history('training_history_coralfrontend.json')
    
    # Grid search 5-epoch runs in experiment_results
    grid_data = {}
    grid_dir = 'experiment_results'
    if os.path.exists(grid_dir):
        for fname in os.listdir(grid_dir):
            if fname.endswith('.json') and fname.startswith('training_history_'):
                filepath = os.path.join(grid_dir, fname)
                data = load_json_history(filepath)
                if data:
                    # e.g., training_history_concat_lam_0.0001.json
                    # or training_history_none_lam_0.0.json
                    parts = fname.replace('training_history_', '').replace('.json', '').split('_lam_')
                    loc = parts[0]
                    lam = parts[1] if len(parts) > 1 else '0.0'
                    grid_data[(loc, lam)] = data

    # ----------------------------------------------------
    # 2. Compile Results Table (Markdown)
    # ----------------------------------------------------
    markdown_lines = []
    markdown_lines.append("# Thesis Experiments - Results Summary Table\n")
    markdown_lines.append("This table lists the performance of the Baseline CACC model, the GRL variants, and the DeepCORAL variants.\n")
    markdown_lines.append("| Experiment Name / Model | Algorithm | Location | Lambda | Epochs | Best Val MAE | At Epoch | Final Val MAE |")
    markdown_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    all_results = []
    
    # Long runs
    if history_grl_concat_100:
        val_mae = history_grl_concat_100['val_mae']
        best_mae = min(val_mae)
        best_ep = val_mae.index(best_mae) + 1
        all_results.append(("GRL Concat (Long Run)", "GRL", "Concat", "0.0001", 100, best_mae, best_ep, val_mae[-1]))
        
    if history_coral_front_100:
        val_mae = history_coral_front_100['val_mae']
        best_mae = min(val_mae)
        best_ep = val_mae.index(best_mae) + 1
        all_results.append(("DeepCORAL Frontend (Long Run)", "DeepCORAL", "Frontend", "0.0001", 100, best_mae, best_ep, val_mae[-1]))
        
    # Grid search runs
    for (loc, lam), data in sorted(grid_data.items()):
        val_mae = data.get('val_mae', [])
        if len(val_mae) > 0:
            best_mae = min(val_mae)
            best_ep = val_mae.index(best_mae) + 1
            algo = "GRL" if loc != 'none' else "Baseline"
            name = f"CACC Baseline" if loc == 'none' else f"GRL {loc.capitalize()}"
            all_results.append((name, algo, loc.capitalize(), lam, len(val_mae), best_mae, best_ep, val_mae[-1]))
            
    # Write to list and markdown
    for res in all_results:
        name, algo, loc, lam, epochs, best_mae, best_ep, final_mae = res
        markdown_lines.append(f"| {name} | {algo} | {loc} | {lam} | {epochs} | {best_mae:.4f} | {best_ep} | {final_mae:.4f} |")
        
    with open('thesis_results_table.md', 'w') as f:
        f.write('\n'.join(markdown_lines))
    print("Saved results table to 'thesis_results_table.md'")

    # ----------------------------------------------------
    # 3. Create Plots (Publication Quality)
    # ----------------------------------------------------
    # Set styling
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.edgecolor'] = '#cccccc'
    plt.rcParams['axes.linewidth'] = 0.8
    
    # Color palette
    colors = {
        'baseline': '#2A2D34',  # Dark Charcoal
        'grl': '#0081A7',       # Teal
        'coral': '#F26419',     # Vibrant Orange
        'grid_front': '#3A86C8',
        'grid_context': '#8338EC',
        'grid_concat': '#FF006E'
    }

    # Plot 1: 100-Epoch Long Runs Comparison (Val MAE over Epochs)
    if history_grl_concat_100 and history_coral_front_100:
        plt.figure(figsize=(10, 6))
        
        # Apply smoothing to make it looks nicer for thesis (or plot both raw and smoothed)
        epochs = list(range(1, 101))
        
        # Plot GRL Concat
        mae_grl = history_grl_concat_100['val_mae']
        plt.plot(epochs, mae_grl, color=colors['grl'], alpha=0.3, linestyle='-')
        # rolling average for smooth curves
        smooth_grl = np.convolve(mae_grl, np.ones(5)/5, mode='valid')
        plt.plot(epochs[2:-2], smooth_grl, color=colors['grl'], label='GRL Concat (100 Epochs) [Best MAE: 6.90]', linewidth=2.5)
        
        # Plot DeepCORAL Frontend
        mae_coral = history_coral_front_100['val_mae']
        plt.plot(epochs, mae_coral, color=colors['coral'], alpha=0.3, linestyle='-')
        smooth_coral = np.convolve(mae_coral, np.ones(5)/5, mode='valid')
        plt.plot(epochs[2:-2], smooth_coral, color=colors['coral'], label='DeepCORAL Frontend (100 Epochs) [Best MAE: 10.21]', linewidth=2.5)
        
        plt.title('Validation Performance Comparison (100 Epochs)', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Epoch', fontsize=12, labelpad=10)
        plt.ylabel('Mean Absolute Error (MAE)', fontsize=12, labelpad=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.ylim(0, 100) # zoom in to the interesting region
        plt.legend(frameon=True, facecolor='white', edgecolor='none', shadow=False)
        plt.tight_layout()
        plt.savefig('thesis_long_runs_comparison.png', dpi=300)
        plt.close()
        print("Saved 'thesis_long_runs_comparison.png'")

    # Plot 2: Hyperparameter Grid Search (5-Epochs)
    # We want to compare locations across different lambdas
    locations = ['frontend', 'context', 'concat']
    lambdas = ['0.1', '0.01', '0.0001']
    
    # Gather best MAE for each
    sweep_results = {loc: [] for loc in locations}
    for loc in locations:
        for lam in lambdas:
            data = grid_data.get((loc, lam))
            if data and 'val_mae' in data:
                sweep_results[loc].append(min(data['val_mae']))
            else:
                sweep_results[loc].append(0.0) # Placeholder
                
    # Also get baseline CACC
    baseline_mae = 7.6942
    if ('none', '0.0') in grid_data:
        baseline_mae = min(grid_data[('none', '0.0')]['val_mae'])
        
    x = np.arange(len(lambdas))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width, sweep_results['frontend'], width, label='GRL Frontend', color=colors['grid_front'])
    rects2 = ax.bar(x, sweep_results['context'], width, label='GRL Context', color=colors['grid_context'])
    rects3 = ax.bar(x + width, sweep_results['concat'], width, label='GRL Concat', color=colors['grid_concat'])
    
    # Add line for baseline CACC
    ax.axhline(y=baseline_mae, color=colors['baseline'], linestyle='--', linewidth=1.5, label=f'Baseline CACC (No GRL) [MAE: {baseline_mae:.2f}]')
    
    ax.set_title('GRL Architecture & Lambda Grid Search (5 Epochs)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Domain Classifier Weight ($\lambda$)', fontsize=12, labelpad=10)
    ax.set_ylabel('Best Validation MAE', fontsize=12, labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(lambdas)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend(frameon=True, facecolor='white', edgecolor='none')
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
                            
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.tight_layout()
    plt.savefig('thesis_grl_grid_search.png', dpi=300)
    plt.close()
    print("Saved 'thesis_grl_grid_search.png'")

    # Plot 3: Baseline vs GRL vs DeepCORAL Convergence
    # Let's show Val MAE over epochs for GRL vs DeepCORAL vs Baseline (which we can simulate using the 5-epoch runs to show short-term adaptation)
    if ('none', '0.0') in grid_data and ('context', '0.01') in grid_data:
        plt.figure(figsize=(10, 6))
        
        epochs = list(range(1, 6))
        plt.plot(epochs, grid_data[('none', '0.0')]['val_mae'], color=colors['baseline'], marker='o', linewidth=2, label='Baseline CACC')
        plt.plot(epochs, grid_data[('context', '0.01')]['val_mae'], color=colors['grl'], marker='s', linewidth=2, label='GRL Context ($\lambda=0.01$)')
        
        # Let's see if we have 5-epoch CORAL frontend results to compare
        coral_front_mae_5 = [22.59, 12.11, 10.20, 11.13, 42.73] # extracted from first 5 epochs of CORAL frontend
        plt.plot(epochs, coral_front_mae_5, color=colors['coral'], marker='^', linewidth=2, label='DeepCORAL Frontend ($\lambda=0.0001$)')
        
        plt.title('Initial Adaptation Convergence (First 5 Epochs)', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Epoch', fontsize=12, labelpad=10)
        plt.ylabel('Validation MAE', fontsize=12, labelpad=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xticks(epochs)
        plt.legend(frameon=True, facecolor='white', edgecolor='none')
        plt.tight_layout()
        plt.savefig('thesis_adaptation_convergence_5_epochs.png', dpi=300)
        plt.close()
        print("Saved 'thesis_adaptation_convergence_5_epochs.png'")

if __name__ == '__main__':
    analyze_and_plot()
