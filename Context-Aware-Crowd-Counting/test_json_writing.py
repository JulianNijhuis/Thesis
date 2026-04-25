import json
import torch
import numpy as np
import os

def test_json_writing():
    # Mocking the train logic saving to history
    history = {'train_loss': [], 'val_mae': []}

    # Simulate 3 epochs
    for epoch in range(3):
        # Mock train_loss (which comes from loss.item() in the real script)
        train_loss = 1.5 - epoch * 0.1
        history['train_loss'].append(train_loss)

        # Mock validate MAE behavior
        mae = 0
        num_batches = 5
        
        for i in range(num_batches):
            # Simulated outputs from evaluate loop:
            # batch_density is a numpy array in the real script
            batch_density = np.random.rand(4, 1, 64, 64).astype(np.float32)
            # target is a PyTorch tensor
            target = torch.rand((1, 128, 128), dtype=torch.float32)

            # This is exactly what caused issues previously, now fixed with float() and .item()
            pred_sum = batch_density.sum() # numpy scalar
            
            # Simulated fix computation
            mae += float(abs(pred_sum - target.sum().item()))

        mae = mae / num_batches
        
        # Append simulated MAE 
        history['val_mae'].append(mae)

        # Attempt JSON dump (this would crash before the fix)
        test_filename = 'test_training_history.json'
        
        try:
            with open(test_filename, 'w') as f:
                json.dump(history, f)
            print(f"Epoch {epoch}: Successfully wrote JSON with train_loss: {train_loss:.3f}, val_mae: {mae:.3f}")
        except TypeError as e:
            print(f"Epoch {epoch}: FAILED to write JSON. Error: {e}")
            return False

    # Cleanup
    if os.path.exists(test_filename):
        os.remove(test_filename)
        
    print("All JSON serializations passed successfully!")
    return True

if __name__ == '__main__':
    test_json_writing()
