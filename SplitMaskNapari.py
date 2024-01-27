# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 11:40:14 2024
Split cell mask with individual files. Cells can't touch each other
@author: azhang
"""

import cv2
import numpy as np
import os

# Load your TIFF image with the mask using an alternative method
image_path = r"Z:\skala\Andy\SingleCell_Mito_Morphology\20231006_Day8_A2\CellMask\1_CellMask_test.tif"
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# Check image properties
print("Original image shape:", image.shape)
print("Original image channels:", image.shape[-1])

# If the image still has an unexpected number of channels, try to extract the first channel
if image.ndim == 3 and image.shape[-1] != 1:
    gray_image = image[:, :, 0]
else:
    gray_image = image

# Convert the grayscale image to a format compatible with thresholding
gray_image = gray_image.astype(np.uint8)

# Check grayscale image properties
print("Grayscale image shape:", gray_image.shape)

# Threshold the grayscale image to obtain a binary mask
_, binary_mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)

# Check binary mask properties
print("Binary mask shape:", binary_mask.shape)

# Find contours in the binary mask
contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Specify the output directory for saving the new images
output_directory = r"Z:\skala\Andy\SingleCell_Mito_Morphology\20231006_Day8_A2\CellMask\1"
os.makedirs(output_directory, exist_ok=True)

# Set a minimum contour area threshold
min_contour_area = 100  # Adjust this value based on your image characteristics

# Iterate through contours and create separate images for each object
for i, contour in enumerate(contours):
    # Filter out small contours
    if cv2.contourArea(contour) < min_contour_area:
        continue

    # Create a blank mask with the same size as the original image
    mask = np.zeros_like(gray_image, dtype=np.uint8)

    # Draw the contour on the mask
    cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

    # Mask the original image with the object's contour
    object_result = cv2.bitwise_and(image, image, mask=mask)

    # Save the resulting image for the current object in the specified directory
    cv2.imwrite(os.path.join(output_directory, f'object_{i + 1}.tif'), object_result)

print("Images saved successfully.")
