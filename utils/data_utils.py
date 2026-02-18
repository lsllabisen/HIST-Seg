import os
import json
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from spectral import open_image
import torch
from torch.utils.data import Dataset


class SpecsGeneration:
    """
    Generate a JSON file with HSI image names for selected subjects.
    """
    def __init__(self, root_dir, subjects):
        self.root_dir = root_dir
        self.subjects = subjects  

    def collect_images(self):
        """
        Collect image names for each subject.
        """
        data_specs = []  
        fold_specs = {"fold_name": "single_fold"}  

        for subject in self.subjects:
            subject_segmentation_path = os.path.join(self.root_dir, "Segmentation", subject)
            if os.path.isdir(subject_segmentation_path):
                fold_specs[subject] = {"image_names": []} 

                for recording in os.listdir(subject_segmentation_path):
                    recording_path = os.path.join(subject_segmentation_path, recording)
                    if os.path.isdir(recording_path):
                        image_name = f"{subject}#{recording}"
                        fold_specs[subject]["image_names"].append(image_name)

        data_specs.append(fold_specs)

        return data_specs

    def generate_folds(self):
        return self.collect_images()  

    def save_json(self, output_file):
        """
        Save the generated data into a JSON file.
        """
        data_specs = self.generate_folds()  
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data_specs, f, ensure_ascii=False, indent=4)  


def get_image_path(image_name, root_dir):
    """
    Return the full path to the HSI image given its name.
    """
    subject, recording = image_name.split('#')

    image_dir = root_dir / "Segmentation" /  subject / recording
    return image_dir



def readHSI(Path_Header,  bands_idx = None):
    """
    Read a hyperspectral image (HSI) from a header file using spectral library.
    Return the cube (H x W x Bands) and corresponding wavelengths.
    """
 
    # Open HSI file
    HSI = open_image(Path_Header)
    wvl = HSI.bands.centers 
    wvl = np.array(wvl,dtype='int')
   
    # Load the image in memory (all bands)
    hs_data = HSI.load()
    
    # Crop wvl
    if bands_idx is not None:
        hsi_3d= hs_data.read_bands(bands_idx)
        wvl = wvl[bands_idx[0]:(bands_idx[len(bands_idx) - 1]+1)] # crop wvl when band selection
    else:
        bands_idx = np.arange(0, len(wvl), 1).tolist()  
        hsi_3d= hs_data.read_bands(bands_idx)

    return hsi_3d, wvl


def display_HSI_rgb_image(HSI_3D, wvl_bandCrop, row_range=None, col_range=None, verbose = True):
    """
    Displays the RGB image of the hyperspectral image.
    """
    # Define RGB band indices
    red_idx = np.argmin(np.abs(wvl_bandCrop - 650))  
    green_idx = np.argmin(np.abs(wvl_bandCrop - 550)) 
    blue_idx = np.argmin(np.abs(wvl_bandCrop - 450)) 
 
    # Get RGB bands
    if row_range and col_range:
        HSI_3D = HSI_3D[row_range[0]:row_range[1], col_range[0]:col_range[1], :]

    rgb_image = np.stack([
        HSI_3D[:, :, red_idx],
        HSI_3D[:, :, green_idx],
        HSI_3D[:, :, blue_idx]
    ], axis=-1)
 
    # Ensure the RGB image is not empty before normalizing
    if rgb_image.size == 0:
        print("Error: The RGB image is empty!")
        return
 
    # Normalize the RGB image to range [0, 1]
    rgb_image = (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image))
    if verbose:
        # Display the image
        plt.imshow(rgb_image)
        plt.title("RGB Image") 
        plt.axis('off')
        plt.show()
    return rgb_image


color_mapping = { 
    "Bone": [142, 223, 124],
    "Cartilage": [111, 225, 243],
    "Ligament": [245, 245, 63],
    "Flesh": [246, 58, 90],
    "Fat": [243, 172, 99],
    "Instruments": [168, 109, 109],
    "Background": [169, 169, 172]
}
num_classes = len(color_mapping)
color_to_class = {tuple(v): i for i, v in enumerate(color_mapping.values())}

def rgb_to_class(mask_rgb):
    """
    Convert an RGB mask to class indices based on a color mapping.
    """
    h, w, _ = mask_rgb.shape
    mask_class = np.zeros((h, w), dtype=np.int64)
    for rgb, cls in color_to_class.items():
        mask_class[np.all(mask_rgb == rgb, axis=-1)] = cls
    return mask_class

class HSICubeMaskDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def rgb_to_class(self, mask_rgb):
        h, w, _ = mask_rgb.shape
        mask_class = np.zeros((h, w), dtype=np.int64)
        for rgb, cls in color_to_class.items():
            mask_class[np.all(mask_rgb == rgb, axis=-1)] = cls
        return mask_class
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load the HSI image from the CSV
        image_path = Path(row['HSI_path'])
        image_name = image_path.name
        hdr_file_path = image_path / f"{image_name}.hdr"
        bands_idx = np.arange(19, 211, 1).tolist()  # Indices of the spectral bands to load
        hsi_3d, wvl = readHSI(Path_Header=hdr_file_path, bands_idx=bands_idx)
        image = torch.tensor(hsi_3d, dtype=torch.float32).permute(2, 0, 1)

        # Load the mask
        mask_path = Path(row['Mask_path'])
        mask = np.array(Image.open(mask_path))
        mask_class = self.rgb_to_class(mask)
        mask = torch.tensor(np.eye(num_classes)[mask_class], dtype=torch.float32).permute(2,0,1)
        
        return image, mask
    


