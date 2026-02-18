import torch
import numpy as np


def make_predictions_with_cubes(model, image, device):
    model.eval()

    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image, dtype=torch.float32)

    if image.ndim == 3:
        image = image.unsqueeze(0) 

    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        segmented_image = torch.argmax(outputs, dim=1)  
        
    segmented_image = segmented_image.cpu().numpy()

    if segmented_image.shape[0] == 1:
        segmented_image = segmented_image.squeeze(0)  

    return segmented_image


def calculate_dice_iou(predicted_mask, true_mask, num_classes):
    """
    Compute the Dice Score and IoU for each class.
    """
    dice_scores = []
    iou_scores = []

    for cls in range(num_classes):
        pred_cls = (predicted_mask == cls).astype(np.float32)  
        true_cls = (true_mask == cls).astype(np.float32)   

        # Intersection and Union
        intersection = np.sum(pred_cls * true_cls)
        union = np.sum(pred_cls) + np.sum(true_cls)

        # Dice Score
        dice = (2 * intersection) / (union + 1e-6) 
        dice_scores.append(dice)

        # IoU
        union_for_iou = np.sum(pred_cls) + np.sum(true_cls) - intersection
        iou = intersection / (union_for_iou + 1e-6) 
        iou_scores.append(iou)

    return dice_scores, iou_scores

