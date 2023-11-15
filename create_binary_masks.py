"""
This file contains the function to create binary masks for GradCAM and LIME.
"""
import cv2
import numpy as np

def get_countour_mask(mask):
    """Returns a binary mask with the contours of the original mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area_mask = np.zeros_like(mask)
    for contour in contours:
        cv2.drawContours(area_mask, [contour], -1, (255), thickness=cv2.FILLED)
    final_mask = cv2.bitwise_or(mask, area_mask)
    return final_mask


def get_single_mask(img_path, mask_path, lower_bound, upper_bound, XAI_method):
    """Gets a binary mask for a single image."""
    heatmap_img = cv2.imread(img_path)
    hsv_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2HSV)
    binary_mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
    if XAI_method == "GradCAM":
        cv2.imwrite(mask_path, binary_mask)
    else:
        binary_mask = get_countour_mask(binary_mask)
        binary_mask = cv2.resize(binary_mask, (224, 224))
        cv2.imwrite(mask_path, binary_mask)
    return binary_mask
