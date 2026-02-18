import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import datetime


def train_model_with_logging(model, train_loader, val_loader, optimizer, num_epochs, device, log_dir, num_classes):
    """
    Train the model with logging of losses, Dice and IoU scores per epoch.
    """
    model.to(device)
    metrics_log = [] 
    
    # Create a file with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"{log_dir}/training_metrics_{timestamp}.xlsx"
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        print(f"\n=== Start of epoch {epoch + 1}/{num_epochs} ===")
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images) 
            target = torch.argmax(masks, dim=1)

            loss = combined_loss(outputs, target, num_classes=num_classes)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Evaluate on the validation set
        val_loss, val_metrics = evaluate_model(model, val_loader, device, num_classes=num_classes)

        # Sauvegarder les métriques pour cette époque
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss / len(train_loader),
            "val_loss": val_loss,
            "accuracy": val_metrics["accuracy"],
            "dice_scores": val_metrics["dice_scores"],
            "iou_scores": val_metrics["iou_scores"]
        }
        metrics_log.append(epoch_metrics)

        # Display results
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {epoch_metrics['train_loss']:.4f}, Val Loss: {epoch_metrics['val_loss']:.4f}")
        print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Dice Scores: {val_metrics['dice_scores']}")
        print(f"IoU Scores: {val_metrics['iou_scores']}")

        # Save metrics for this epoch
        df = pd.DataFrame(metrics_log)
        df.to_excel(log_path, index=False)
        print(f"Metrics saved to: {log_path}")
    
    # Return the model and the logged metrics
    return model, metrics_log


def evaluate_model(model, val_loader, device, num_classes):
    model.eval()
    val_loss = 0
    total = 0
    accuracy = 0
    dice_scores_all = []
    iou_scores_all = []

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device) 
            
            outputs = model(images) 
            target = torch.argmax(masks, dim=1)  

            loss = combined_loss(outputs, target, num_classes=num_classes)
            val_loss += loss.item()

            probs = F.softmax(outputs, dim=1) 
            preds = torch.argmax(probs, dim=1)  

            # Compute metrics
            dice_scores_all.append(dice_score_per_class(preds, target, num_classes))
            iou_scores_all.append(iou_score_per_class(preds, target, num_classes))
            accuracy += (preds == target).sum().item()
            total += target.numel()

    # Average metrics across all classes
    dice_scores_avg = torch.tensor(dice_scores_all).mean(dim=0).tolist()
    iou_scores_avg = torch.tensor(iou_scores_all).mean(dim=0).tolist()

    return val_loss / len(val_loader), {
        "accuracy": accuracy / total,
        "dice_scores": dice_scores_avg,
        "iou_scores": iou_scores_avg
    }


# Combined loss function (Dice Loss + Cross Entropy Loss)
def combined_loss(pred, target, num_classes):
    
    # Cross Entropy Loss
    ce_loss = nn.CrossEntropyLoss()(pred, target) 

    # Dice Loss
    probs = torch.softmax(pred, dim=1) 
    preds = torch.argmax(probs, dim=1)
    
    dice_scores = []
    dice_scores.append(dice_score_per_class(preds, target, num_classes))

    # Average metrics over all classes
    dice_scores_avg = torch.tensor(dice_scores).mean(dim=0).tolist()
    dice_loss = 1 - torch.tensor(dice_scores_avg).mean(dim=0)

    # Combined loss
    return ce_loss + dice_loss

def dice_score(pred, target, smooth=1e-6):
    intersection = torch.sum(pred * target)
    return (2. * intersection + smooth) / (torch.sum(pred) + torch.sum(target) + smooth)


def dice_score_per_class(pred, target, num_classes):
    """
    Computes the Dice Score for each class.
    """
    dice_scores = []
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()

        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        dice = (2 * intersection) / (union + 1e-6) 
        dice_scores.append(dice.item())
    return dice_scores


def iou_score_per_class(pred, target, num_classes):
    """
    Computes the IoU (Intersection over Union) for each class.
    """
    iou_scores = []
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()

        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum() - intersection
        iou = intersection / (union + 1e-6) 
        iou_scores.append(iou.item())
    return iou_scores



