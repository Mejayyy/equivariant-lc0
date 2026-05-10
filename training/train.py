import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Adjust path so we can import from the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.baseline import BaselineLCZero
# Assuming dataset.py is in the data/ folder based on the screenshot
from data.dataset import LCZeroV6Dataset

def policy_loss(logits, targets):
    """
    Computes Cross Entropy for the policy head.
    Handles V6 datasets -1 sentinels by masking them out.
    """
    mask = targets >= 0
    # Replace -1 with 0 so it doesn't mess up the calculation, 
    # the mask will zero out the gradients for these positions anyway.
    valid_targets = torch.where(mask, targets, torch.zeros_like(targets))
    
    log_preds = F.log_softmax(logits, dim=1)
    # -sum(target * log(pred))
    return -torch.sum(valid_targets * log_preds * mask, dim=1).mean()

def value_loss(preds, targets):
    """
    Computes Cross Entropy for the WDL value head.
    Note: baseline.py states the value head outputs a softmax [W, D, L],
    so we must use a manual log-likelihood calculation instead of nn.CrossEntropyLoss 
    (which expects raw logits).
    """
    # Add a small epsilon to prevent log(0)
    return -torch.sum(targets * torch.log(preds + 1e-8), dim=1).mean()

def mlh_loss(preds, targets):
    """
    Computes Mean Squared Error for the Moves Left Head (MLH).
    """
    return F.mse_loss(preds.squeeze(1), targets)

def train():
    # 1. Configuration
    data_folders = ["raw_data"] # Update this to wherever your .gz files are
    batch_size = 64
    learning_rate = 1e-3
    epochs = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")

    # 2. Setup Dataset and DataLoader
    # Note: For IterableDatasets, num_workers > 0 requires a worker_init_fn 
    # to avoid data duplication. Setting to 0 for a safe baseline.
    dataset = LCZeroV6Dataset(folders=data_folders, shuffle_buffer=8192)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

    # 3. Initialize Model and Optimizer
    model = BaselineLCZero(blocks=3, filters=124).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        # Using tqdm for a nice progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for step, (planes, targets) in enumerate(pbar):
            # Move data to GPU/CPU
            planes = planes.to(device)
            policy_target = targets["policy"].to(device)
            value_target = targets["value"].to(device)
            mlh_target = targets["mlh"].to(device)

            # Forward pass
            optimizer.zero_grad()
            policy_logits, value_wdl, moves_left = model(planes)

            # Calculate individual losses
            l_pol = policy_loss(policy_logits, policy_target)
            l_val = value_loss(value_wdl, value_target)
            l_mlh = mlh_loss(moves_left, mlh_target)

            # Combined loss
            # You can tweak these weights! MLH is usually an auxiliary loss.
            loss = l_pol + (1.0 * l_val) + (0.01 * l_mlh)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Update progress bar
            if step % 10 == 0:
                pbar.set_postfix({
                    "Loss": f"{loss.item():.4f}", 
                    "Pol": f"{l_pol.item():.4f}", 
                    "Val": f"{l_val.item():.4f}",
                    "MLH": f"{l_mlh.item():.4f}"
                })

        # Save checkpoint after each epoch
        checkpoint_dir = "runs"
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
        }, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt"))
        
        print(f"Epoch {epoch+1} complete. Checkpoint saved.")

if __name__ == "__main__":
    train()
