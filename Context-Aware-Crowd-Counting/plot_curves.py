import json
import matplotlib.pyplot as plt
import os

def plot_learning_curves(history_file='path/to/training_history.json', output_file='path/to/learning_curves.png'):
    if not os.path.exists(history_file):
        print(f"Error: {history_file} not found.")
        return

    with open(history_file, 'r') as f:
        try:
            history = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: {history_file} is empty or invalid. Training might not have saved data yet.")
            return

    if 'train_loss' not in history or 'val_mae' not in history:
        print("Error: The history file doesn't contain 'train_loss' or 'val_mae'.")
        return

    epochs_train = range(1, len(history['train_loss']) + 1)
    epochs_val = range(1, len(history['val_mae']) + 1)

    plt.figure(figsize=(12, 5))

    # Training Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs_train, history['train_loss'], label='Train Loss', color='blue', marker='o', markersize=3)
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Validation MAE Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs_val, history['val_mae'], label='Validation MAE', color='orange', marker='o', markersize=3)
    plt.title('Validation MAE Curve')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (Mean Absolute Error)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Learning curves successfully saved to '{output_file}'. You can open this file to see your graphs!")

if __name__ == '__main__':
    plot_learning_curves()
