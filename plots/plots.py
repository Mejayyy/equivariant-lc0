import json
import matplotlib.pyplot as plt
import os

def parse_log(file_path):
    """Reads a JSONL log file and extracts steps and losses."""
    train_steps, train_losses = [], []
    val_steps, val_losses = [], []

    if not os.path.exists(file_path):
        print(f"Warning: Could not find '{file_path}'. Check the path!")
        return train_steps, train_losses, val_steps, val_losses

    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                
                # Skip summary or init lines that don't have a 'step'
                if "step" not in data:
                    continue
                    
                # Extract Training Loss
                if data.get("phase") == "train" and "loss_avg" in data:
                    train_steps.append(data["step"])
                    train_losses.append(data["loss_avg"])
                    
                # Extract Validation Loss
                elif data.get("phase") == "val" and "total" in data:
                    val_steps.append(data["step"])
                    val_losses.append(data["total"])
                    
            except json.JSONDecodeError:
                continue

    return train_steps, train_losses, val_steps, val_losses

def main():
    # 1. Define where the log files are located
    # 1. Define where the log files are located

    baseline_log = r"cuda_baseline_resonable_1/train_log.jsonl"
    gcnn_log = r"cuda_gcnn_resonable_1/train_log.jsonl"

    # 2. Parse the data from the files
    print("Parsing logs...")
    b_t_steps, b_t_losses, b_v_steps, b_v_losses = parse_log(baseline_log)
    g_t_steps, g_t_losses, g_v_steps, g_v_losses = parse_log(gcnn_log)

    # 3. Set up the plot
    plt.figure(figsize=(10, 6))

    # Plot Baseline (Green: Solid for Train, Dotted for Val)
    if b_t_steps:
        plt.plot(b_t_steps, b_t_losses, color='green', linestyle='-', label='Baseline Train Loss')
    if b_v_steps:
        plt.plot(b_v_steps, b_v_losses, color='green', linestyle=':', linewidth=2, label='Baseline Val Loss')

    # Plot GCNN (Blue: Solid for Train, Dotted for Val)
    if g_t_steps:
        plt.plot(g_t_steps, g_t_losses, color='blue', linestyle='-', label='GCNN Train Loss')
    if g_v_steps:
        plt.plot(g_v_steps, g_v_losses, color='blue', linestyle=':', linewidth=2, label='GCNN Val Loss')

    # 4. Format the chart nicely
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Total Combined Loss', fontsize=12)
    plt.title('Baseline vs. Equivariant GCNN Training Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 5. Save and display
    plt.tight_layout()
    output_filename = 'loss_comparison_plot.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Success! Plot saved as '{output_filename}' in your project folder.")
    
    # Opens the interactive window
    plt.show()

if __name__ == "__main__":
    main()