# LIME and Grad-CAM Comparison Using IoU

This repository contains a Python script for comparing interpretability images from LIME (Local Interpretable Model-agnostic Explanations) and Grad-CAM (Gradient-weighted Class Activation Mapping) using Intersection over Union (IoU).

## Overview

The script processes images output by LIME and Grad-CAM to create binary masks that highlight significant areas for model interpretation. It then calculates the IoU score to quantify the overlap between these areas across both interpretability methods.

## Features

- Individual processing of LIME and Grad-CAM images.
- Color segmentation to differentiate high-value areas.
- Contour detection and filling for precise area marking.
- Intersection, union, and IoU calculations for comparative analysis.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.6 or higher
- OpenCV library
- NumPy library
- Pillow library

You can install the necessary libraries using `pip`:

```bash
pip install numpy opencv-python pillow
```

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.
