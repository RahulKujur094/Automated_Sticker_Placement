"""
Script to create a default sticker image.
Run this to generate a sample sticker.png file.
"""

import cv2
import numpy as np
import os

def create_default_sticker(output_path="stickers/sticker.png", text="Rahul", width=200, height=80):
    """
    Create a default sticker image with text.
    
    Args:
        output_path: Path to save the sticker image
        text: Text to display on sticker
        width: Width of sticker in pixels
        height: Height of sticker in pixels
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create image with transparent background
    sticker = np.zeros((height, width, 4), dtype=np.uint8)
    
    # Fill with black background (you can change this color)
    sticker[:, :, 0] = 0  # B
    sticker[:, :, 1] = 0  # G
    sticker[:, :, 2] = 0  # R
    sticker[:, :, 3] = 255  # Alpha (fully opaque)
    
    # Add white text
    font = cv2.FONT_HERSHEY_BOLD
    font_scale = 1.2
    thickness = 2
    color = (255, 255, 255, 255)  # White text
    
    # Get text size for centering
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = (width - text_width) // 2
    text_y = (height + text_height) // 2
    
    # Draw text
    cv2.putText(sticker, text, (text_x, text_y), font, font_scale, color, thickness)
    
    # Save as PNG with transparency
    cv2.imwrite(output_path, sticker)
    print(f"Created default sticker at: {output_path}")
    print(f"Sticker size: {width}x{height} pixels")
    print(f"Text: {text}")

if __name__ == "__main__":
    create_default_sticker()


