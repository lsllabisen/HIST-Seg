import pandas as pd
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

from utils.data_utils import readHSI, rgb_to_class, display_HSI_rgb_image
from utils.prediction_utils import make_predictions_with_cubes, calculate_dice_iou


# -----------------------------------------
# VISUALIZATION OF A SINGLE TEST IMAGE
# -----------------------------------------

def visualize_predictions(model_trained, img_idx, config, device):
    """
    Generate visualizations comparing RGB HSI, ground-truth masks, and predicted segmentation.
    """

    # Prepare output directories
    output_dir = Path(config["output"]["save_predictions_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log file
    log_file = output_dir / "visualization_log.txt"
    log_buffer = []  

    def log_print(*args):
        """Print to console + save to buffer for writing to file."""
        text = " ".join(str(a) for a in args)
        print(text)
        log_buffer.append(text)

    csv_path_test = config["data"]["csv_test"]
    df_test = pd.read_csv(csv_path_test)
    num_classes = config["model"]["num_classes"]
    output_dir = Path(config["output"]["save_predictions_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    color_mapping = {
        "Bone": [142, 223, 124],
        "Cartilage": [111, 225, 243],
        "Ligament": [245, 245, 63],
        "Flesh": [246, 58, 90],
        "Fat": [243, 172, 99],
        "Instruments": [168, 109, 109],
        "Background": [169, 169, 172]
    }

    colors = list(color_mapping.values())
    cmap = ListedColormap(np.array(colors)/255)
    norm = BoundaryNorm(np.arange(num_classes+1)-0.5, cmap.N)

    log_print("\nGenerating visualizations...")
    log_print(f"Selected image index: {img_idx}")

    image_path = Path(df_test['HSI_path'][img_idx])
    mask_path  = Path(df_test['Mask_path'][img_idx])

    image_name = image_path.name
    hdr_file_path = image_path / f"{image_name}.hdr"

    # Select spectral bands to load
    bands_img_idx = np.arange(19, 211, 1).tolist()

    # --- Load HSI cube ---
    hsi_3d, wvl = readHSI(Path_Header=hdr_file_path, bands_idx=bands_img_idx)
    image = torch.tensor(hsi_3d, dtype=torch.float32).permute(2, 0, 1)
    image_np = image.permute(1, 2, 0).numpy()

    # --- Load ground truth mask ---
    mask = np.array(Image.open(mask_path))
    mask_class = rgb_to_class(mask)
    mask_tensor = torch.tensor(np.eye(num_classes)[mask_class], dtype=torch.float32).permute(2,0,1)
    mask_indices = torch.argmax(mask_tensor, dim=0).numpy()
    mask_rgb = np.zeros((mask_indices.shape[0], mask_indices.shape[1], 3), dtype=np.uint8)
    for cls_img_idx, color in enumerate(color_mapping.values()):
        mask_rgb[mask_indices == cls_img_idx] = color

    # --- Predictions  ---
    segmented_image = make_predictions_with_cubes(model_trained, image, device)
    dice_scores, iou_scores = calculate_dice_iou(segmented_image, mask_indices, num_classes)
    log_print("Dice scores:", dice_scores)
    log_print("IoU scores :", iou_scores)

    # --- Generate RGB representation of HSI image ---
    rgb_image = display_HSI_rgb_image(image_np, wvl, verbose=False)

    output_file = output_dir / f"{image_name}_Visualization.png"

    # --- Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(30, 15))

    axes[0].imshow(rgb_image)
    axes[0].set_title("RGB Image")
    axes[0].axis('off')

    axes[1].imshow(mask_rgb)
    axes[1].set_title("Ground truth")
    axes[1].axis('off')

    axes[2].imshow(segmented_image, cmap=cmap, norm=norm)
    axes[2].set_title("Prediction")
    axes[2].axis('off')

    # --- Dynamic legends ---
    present_gt = np.unique(mask_indices)
    patches_gt = [
        mpatches.Patch(color=np.array(list(color_mapping.values())[cls])/255,
                    label=list(color_mapping.keys())[cls])
        for cls in present_gt
    ]

    present_pred = np.unique(segmented_image)
    patches_pred = [
        mpatches.Patch(color=np.array(list(color_mapping.values())[cls])/255,
                    label=list(color_mapping.keys())[cls])
        for cls in present_pred
    ]

    axes[1].legend(handles=patches_gt, title="GT Classes", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    axes[2].legend(handles=patches_pred, title="Pred Classes", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    fig.tight_layout()
    plt.savefig(output_file, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close()

    log_print(f"Visualization saved at: {output_file}")

    # Save Log File
    with open(log_file, "a") as f:
        f.write("\n".join(log_buffer))
        f.write("\n\n" + "="*50 + "\n")

    log_print(f"Log saved in: {log_file}")
