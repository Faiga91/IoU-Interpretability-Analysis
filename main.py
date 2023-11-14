"""
Main script to get the IOU score between the LIME and GradCAM masks.
"""
import os
import pandas as pd
from PIL import Image, ImageOps
import numpy as np


IMAGES_DIR = "/Volumes/EmbryoScope/XAI/VGG16-New"
THRESHOLD = 0.5
iou_scores = []


def images_to_masks(img_orig, img_lime, img_gradcam):
    """Subtract the original image from the LIME and GradCAM images.
    Then, apply a threshold to the resulting images to obtain the binary masks.
    """
    lime_gray = ImageOps.grayscale(img_lime)
    gradcam_gray = ImageOps.grayscale(img_gradcam)
    original_gray = ImageOps.grayscale(img_orig)

    lime_array_gray = np.array(lime_gray)
    gradcam_array_gray = np.array(gradcam_gray)
    original_array_gray = np.array(original_gray)

    gradcam_subtracted = gradcam_array_gray - original_array_gray
    lime_subtracted = lime_array_gray - original_array_gray
    gradcam_subtracted = np.clip(gradcam_subtracted, 0, 1)
    lime_subtracted = np.clip(lime_subtracted, 0, 1)

    lime_bin_mask = np.where(lime_subtracted > THRESHOLD, 1, 0)
    gradcam_bin_mask = np.where(gradcam_subtracted > THRESHOLD, 1, 0)

    return lime_bin_mask, gradcam_bin_mask


def calculate_iou(mask1, mask2):
    """Calculates the IOU score between two binary masks."""
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)

    intersection_sum = np.sum(intersection)
    union_sum = np.sum(union)

    iou_s = intersection_sum / union_sum if union_sum != 0 else 0

    return iou_s


for subdir in os.listdir(IMAGES_DIR):
    subdir_path = os.path.join(IMAGES_DIR, subdir)

    if os.path.isdir(subdir_path):
        lime_path, gradcam_path, original_path = None, None, None

        for file in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, file)

            if file == "positiveRegions.png":
                lime_path = file_path
            elif file == "Gcam_Image.png":
                gradcam_path = file_path
            elif file == f"{subdir}.png":
                original_path = file_path

        if lime_path and gradcam_path and original_path:
            original_image = Image.open(original_path)
            gradcam_image = Image.open(gradcam_path)
            lime_image = Image.open(lime_path)

            lime_resized_image = lime_image.resize((224, 224))

            lime_binary_mask, gradcam_binary_mask = images_to_masks(
                original_image, lime_resized_image, gradcam_image
            )

            iou_score = calculate_iou(lime_binary_mask, gradcam_binary_mask)
            iou_scores.append([subdir, iou_score])


df = pd.DataFrame(iou_scores, columns=["Folder", "IoU-Score"])
df.to_csv("VGG16-New_iou_scores.csv", index=False)
print(f"A total of {len(iou_scores)} images folders were processed.")
print(f"The IoU scores are saved at {os.getcwd()}/VGG16-New_iou_scores.csv.")
