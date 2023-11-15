"""
Main script to get the IOU score between the LIME and GradCAM masks.
"""
import os
import argparse
import pandas as pd
from PIL import Image
import numpy as np
from create_binary_masks import get_single_mask

IMAGES_DIR = "/Volumes/EmbryoScope/XAI/VGG16-New"
OUTPUT_DIR = "/Volumes/EmbryoScope/XAI/Results"
iou_scores = []


def calculate_iou(mask1, mask2):
    """Calculates the IOU score between two binary masks."""
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)

    intersection_sum = np.sum(intersection)
    union_sum = np.sum(union)

    iou_s = intersection_sum / union_sum if union_sum != 0 else 0

    return intersection_sum, union_sum, iou_s


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_dir",
        type=str,
        default=IMAGES_DIR,
        help="Directory where the images are stored.",
    )

    args = parser.parse_args()

    for subdir in os.listdir(args.images_dir):
        subdir_path = os.path.join(args.images_dir, subdir)
        output_dir = os.path.join(OUTPUT_DIR, os.path.basename(subdir_path))

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

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

                lime_image = lime_image.convert("RGB")
                lime_resized_image = lime_image.resize((224, 224))

                gradcam_binary_mask = get_single_mask(
                    gradcam_path,
                    os.path.join(output_dir, "gradcam_binary_mask.png"),
                    (0, 100, 100),
                    (90, 204, 218),
                    "GradCAM",
                )

                lime_binary_mask = get_single_mask(
                    lime_path,
                    os.path.join(output_dir, "lime_binary_mask.png"),
                    (20, 100, 100),
                    (30, 255, 255),
                    "LIME",
                )

                intersection, union, iou_score = calculate_iou(
                    lime_binary_mask, gradcam_binary_mask
                )
                iou_scores.append([subdir, intersection, union, iou_score])

    df = pd.DataFrame(
        iou_scores, columns=["Folder", "Intersection", "Union", "IoU-score"]
    )
    file_name = os.path.basename(args.images_dir) + "-" + "_iou_scores.csv"
    df.to_csv(file_name, index=False)
    print(f"A total of {len(iou_scores)} images folders were processed.")
    print(f"The IoU scores are saved at {file_name}/VGG16-New_iou_scores.csv.")
