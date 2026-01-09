"""
Box Detection Module
Detects rectangular boxes in images using contour detection.
"""

import cv2
import numpy as np


def preprocess_image(image):
    """
    Preprocess image for box detection.
    
    Args:
        image: Input image (BGR format)
        
    Returns:
        edges: Canny edge detected image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    return edges


def find_box_contour(edges):
    """
    Find the largest rectangular contour in the image.
    
    Args:
        edges: Edge detected image
        
    Returns:
        contour: Largest rectangular contour, or None if not found
    """
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Find the largest rectangular contour
    for contour in contours:
        # Approximate contour to polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # Check if it's roughly rectangular (4 corners)
        if len(approx) == 4:
            # Check if area is significant
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum area threshold
                return contour
    
    # If no rectangular contour found, return largest contour
    return contours[0] if contours else None


def get_min_area_rect(contour):
    """
    Get minimum area rectangle from contour.
    
    Args:
        contour: Input contour
        
    Returns:
        rect: Tuple of ((center_x, center_y), (width, height), angle)
    """
    if contour is None:
        return None
    
    rect = cv2.minAreaRect(contour)
    return rect


def detect_box(image):
    """
    Main function to detect box in image.
    
    Args:
        image: Input image (BGR format)
        
    Returns:
        contour: Detected box contour
        rect: Minimum area rectangle
        edges: Edge detected image (for visualization)
    """
    # Preprocess
    edges = preprocess_image(image)
    
    # Find box contour
    contour = find_box_contour(edges)
    
    # Get minimum area rectangle
    rect = get_min_area_rect(contour) if contour is not None else None
    
    return contour, rect, edges


