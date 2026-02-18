import yaml
import torch
from torch.utils.data import DataLoader, random_split
from utils.data_utils import HSICubeMaskDataset
from utils.models import get_model
from utils.model_utils import train_model_with_logging
from utils.model_utils import evaluate_model
from pathlib import Path
import torch.optim as optim
import os
from utils.visualization_utils import visualize_predictions


# Load configuration file
with open("config.yaml") as f:
    config = yaml.safe_load(f)


# Specify device
device = torch.device(f"cuda:{config['gpu']}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load train and test dataset and create train/validation split
dataset_train = HSICubeMaskDataset(config['data']['csv_train'])
val_size = int(0.1 * len(dataset_train))
train_size = len(dataset_train) - val_size
train_dataset, val_dataset = random_split(dataset_train, [train_size, val_size])

dataset_test = HSICubeMaskDataset(config['data']['csv_test'])

train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
test_loader = DataLoader(dataset_test, batch_size=config['testing']['batch_size'], shuffle=False)


# Initialize model
model = get_model(config['model']['arch'], config['model']['in_channels'], config['model']['num_classes'])

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])

# Create logs directory
os.makedirs(config['training']['logs_dir'], exist_ok=True)

# Training
model_trained, training_log = train_model_with_logging(
    model, train_loader, val_loader, optimizer,
    config['training']['num_epochs'], device, config['training']['logs_dir'], config['model']['num_classes']
)


# Save model weights
model_path = Path(config['training']['logs_dir']) / "model_weights.pth"
torch.save(model_trained, model_path)

weight_path = Path(config['training']['logs_dir']) / "weights.pth"
torch.save(model_trained.state_dict(), weight_path)



# Evaluate the model
test_loss, test_metrics = evaluate_model(model_trained, test_loader, device, config['model']['num_classes'])
formatted_dice_scores = [f"{score:.4f}" for score in test_metrics['dice_scores']]
formatted_iou_scores = [f"{score:.4f}" for score in test_metrics['iou_scores']]
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
print(f"Dice Scores: {formatted_dice_scores}")
print(f"IoU Scores: {formatted_iou_scores}")

# Save results to a text file
results_path = Path(config['testing']['results_dir']) / config['testing']['results_file']
results_path.parent.mkdir(parents=True, exist_ok=True)

with open(results_path, "w") as f:
    f.write("Test Results\n")
    f.write("====================\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_metrics['accuracy']:.4f}\n")
    f.write(f"Dice Scores: {formatted_dice_scores}\n")
    f.write(f"IoU Scores: {formatted_iou_scores}\n")

print(f"Results saved to {results_path}")


visualize_predictions(model_trained, 3, config, device)

print("All tasks completed successfully!")