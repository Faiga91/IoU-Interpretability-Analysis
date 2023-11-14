"""
Main script to get the IOU score between the LIME and GradCAM masks.
"""
import os
import pandas as pd
from PIL import Image, ImageOps
import numpy as np


IMAGES_DIR = "/Volumes/EmbryoScope/XAI/VGG16-New"
THRESHOLD = 128
iou_scores = []


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
            lime_image = Image.open(lime_path)
            gradcam_image = Image.open(gradcam_path)
            original_image = Image.open(original_path)

            lime_array = np.array(lime_image).astype(float)
            lime_resized_image = lime_image.resize((224, 224))
            gradcam_array = np.array(gradcam_image).astype(float)
            original_array = np.array(original_image).astype(float)

            lime_gray = ImageOps.grayscale(lime_resized_image)
            gradcam_gray = ImageOps.grayscale(gradcam_image)
            original_gray = ImageOps.grayscale(original_image)

            lime_array_gray = np.array(lime_gray)
            gradcam_array_gray = np.array(gradcam_gray)
            original_array_gray = np.array(original_gray)

            gradcam_subtracted = gradcam_array_gray - original_array_gray
            lime_subtracted = lime_array_gray - original_array_gray
            gradcam_subtracted = np.clip(gradcam_subtracted, 0, 1)
            lime_subtracted = np.clip(lime_subtracted, 0, 1)

            lime_binary_mask = np.where(lime_array_gray > THRESHOLD, 1, 0)
            gradcam_binary_mask = np.where(gradcam_array_gray > THRESHOLD, 1, 0)

            iou_score = calculate_iou(lime_binary_mask, gradcam_binary_mask)
            iou_scores.append([subdir, iou_score])

df = pd.DataFrame(iou_scores, columns=["Folder", "IoU-Score"])
df.to_csv("VGG16-New_iou_scores.csv", index=False)
print(f"A total of {len(iou_scores)} images folders were processed.")
print(f"The IoU scores are saved at {os.getcwd()}/VGG16-New_iou_scores.csv.")
