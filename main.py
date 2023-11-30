"""
Main script to get the IOU score between the LIME and GradCAM masks.
"""
import os
import argparse
import pandas as pd
from PIL import Image
import numpy as np
from create_binary_masks import get_single_mask
from dataclasses import dataclass

IMAGES_DIR = "/Volumes/EmbryoScope/XAI/VGG16-New"
OUTPUT_DIR = "/Volumes/EmbryoScope/XAI/Results"


@dataclass
class ColorSelection:
    lime_lower: tuple
    lime_upper: tuple
    gradcam_lower: tuple
    gradcam_upper: tuple


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def calculate_iou(mask1, mask2):
    """Calculates the IOU score between two binary masks."""
    mask1 = (mask1 > 0).astype(int)
    mask2 = (mask2 > 0).astype(int)

    mask1_pixels_count = np.sum(mask1)
    mask2_pixels_count = np.sum(mask2)

    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)

    intersection_sum = np.sum(intersection)
    union_sum = np.sum(union)

    iou_s = intersection_sum / union_sum if union_sum != 0 else 0

    return intersection_sum, union_sum, iou_s, mask1_pixels_count, mask2_pixels_count


def process_images(images_dir, output_dir, color_selection: ColorSelection):
    iou_scores = []

    for subdir in os.listdir(images_dir):
        subdir_path = os.path.join(images_dir, subdir)
        output_dir = os.path.join(output_dir, os.path.basename(subdir_path))

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
                Image.open(original_path)
                Image.open(gradcam_path)
                lime_image = Image.open(lime_path)

                lime_image = lime_image.convert("RGB")
                lime_image.resize((224, 224))

                gradcam_binary_mask = get_single_mask(
                    gradcam_path,
                    os.path.join(output_dir, "gradcam_binary_mask.png"),
                    color_selection.gradcam_lower,
                    color_selection.gradcam_upper,
                    "GradCAM",
                )

                lime_binary_mask = get_single_mask(
                    lime_path,
                    os.path.join(output_dir, "lime_binary_mask.png"),
                    color_selection.lime_lower,
                    color_selection.lime_upper,
                    "LIME",
                )

                (
                    intersection,
                    union,
                    iou_score,
                    lime_pixels,
                    gradcam_pixels,
                ) = calculate_iou(lime_binary_mask, gradcam_binary_mask)
                iou_scores.append(
                    [
                        subdir,
                        intersection,
                        union,
                        iou_score,
                        lime_pixels,
                        gradcam_pixels,
                    ]
                )
    return iou_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_dir",
        type=str,
        default=IMAGES_DIR,
        help="Directory where the images are stored.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="Directory where the results will be stored.",
    )

    parser.add_argument(
        "--lime_lower",
        type=str,
        default="#146464",
        help="Lower bound for the LIME mask.",
    )
    parser.add_argument(
        "--lime_upper",
        type=str,
        default="#1EFFFF",
        help="Upper bound for the LIME mask.",
    )
    parser.add_argument(
        "--gradcam_lower",
        type=str,
        default="#006464",
        help="Lower bound for the GradCAM mask.",
    )
    parser.add_argument(
        "--gradcam_upper",
        type=str,
        default="#1EFFFF",
        help="Upper bound for the GradCAM mask.",
    )

    args = parser.parse_args()

    colors = ColorSelection(
        lime_lower=hex_to_rgb(args.lime_lower),
        lime_upper=hex_to_rgb(args.lime_upper),
        gradcam_lower=hex_to_rgb(args.gradcam_lower),
        gradcam_upper=hex_to_rgb(args.gradcam_upper),
    )

    iou_scores = process_images(args.images_dir, args.output_dir, colors)

    df = pd.DataFrame(
        iou_scores,
        columns=[
            "Folder",
            "Intersection",
            "Union",
            "IoU-score",
            "LIME pixels",
            "GradCAM pixels",
        ],
    )
    file_name = os.path.basename(args.images_dir) + "-" + "_iou_scores.csv"
    output_file_path = os.path.join(args.output_dir, file_name)
    df.to_csv(output_file_path, index=False)
    print(f"A total of {len(iou_scores)} images folders were processed.")
    print(f"The IoU scores are saved at {output_file_path}")
