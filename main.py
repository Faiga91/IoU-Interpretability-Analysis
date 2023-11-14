"""
Main script to get the IOU score between the LIME and GradCAM masks.
"""
from PIL import Image, ImageOps
import numpy as np


IMAGES_DIR = "/Volumes/EmbryoScope/XAI/VGG16-New"
THRESHOLD = 128

gradcam_image = Image.open("/Volumes/EmbryoScope/XAI/VGG16-New/t2_01/Gcam_Image.png")
gradcam_array = np.array(gradcam_image).astype(float)

lime_image = Image.open("/Volumes/EmbryoScope/XAI/VGG16-New/t2_01/positiveRegions.png")
lime_array = np.array(lime_image).astype(float)
lime_resized_image = lime_image.resize((224, 224))

original_image = Image.open("/Volumes/EmbryoScope/XAI/VGG16-New/t2_01/t2_01.png")
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

gradcam_result_image = Image.fromarray((gradcam_subtracted * 255).astype(np.uint8))
lime_result_image = Image.fromarray((lime_subtracted * 255).astype(np.uint8))


lime_binary_mask = np.where(lime_array_gray > THRESHOLD, 1, 0)
gradcam_binary_mask = np.where(gradcam_array_gray > THRESHOLD, 1, 0)


lime_mask_image = Image.fromarray((lime_binary_mask * 255).astype(np.uint8))
gradcam_mask_image = Image.fromarray((gradcam_binary_mask * 255).astype(np.uint8))

lime_mask_image.save("lime_binary_mask.png")
gradcam_mask_image.save("gradcam_binary_mask.png")


intersection = np.logical_and(lime_binary_mask, gradcam_binary_mask)
union = np.logical_or(lime_binary_mask, gradcam_binary_mask)

intersection_sum = np.sum(intersection)
union_sum = np.sum(union)

iou_score = intersection_sum / union_sum if union_sum != 0 else 0

print(iou_score)
