import json
import pandas as pd
from pathlib import Path
import argparse

from utils.data_utils import get_image_path, SpecsGeneration

def generate_json_csv(root_dir, subjects, output_json, output_csv):
    """
    Generate JSON file for train/test and a CSV file with HSI and mask paths
    """
    generator = SpecsGeneration(root_dir, subjects)
    generator.save_json(output_json)

    # Load generated JSON
    with Path(output_json).open("r") as f:
        folds_data = json.load(f)

    # Collect paths
    subject_data = {}
    for fold in folds_data:
        for subject, subject_info in fold.items():
            if subject == "fold_name":
                continue
            image_paths = [get_image_path(img_name, Path(root_dir)) for img_name in subject_info["image_names"]]
            mask_paths = [
                [
                    path for path in (Path(root_dir) / "Segmentation" / img_name.replace("#", "/") / "Annotations").iterdir()
                    if path.suffix == ".png"
                ]
                for img_name in subject_info["image_names"]
            ]
            subject_data[subject] = {"image_paths": image_paths, "mask_paths": mask_paths}

    # Convert to CSV
    rows = []
    for subject, data in subject_data.items():
        for img_path, masks in zip(data["image_paths"], data["mask_paths"]):
            img_path_str = str(img_path)
            masks_str = ";".join([str(mask) for mask in masks])
            rows.append([img_path_str, masks_str])

    df = pd.DataFrame(rows, columns=["HSI_path", "Mask_path"])
    df.to_csv(output_csv, index=False)
    print(f"Saved CSV to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate JSON and CSV for HypoChir dataset")
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True, help="Mode: train or test")
    args = parser.parse_args()

    root_dir = "/media/nas/LSL_Team/Data/2025_HYPOCHIR_Dataset"
    if args.mode == "train":
        subjects = ["B0001", "B0002", "B0003", "B0004", "B0005", "B0006", "B0007", "B0008"]
        output_json = "Dataset/images_list_train.json"
        output_csv = "Dataset/Data_paths_train.csv"
    else:  # test
        subjects = ["B0009", "B0010"]
        output_json = "Dataset/images_list_test.json"
        output_csv = "Dataset/Data_paths_test.csv"

    Path("Dataset").mkdir(parents=True, exist_ok=True)

    generate_json_csv(root_dir, subjects, output_json, output_csv)
