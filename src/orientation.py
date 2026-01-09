"""
Orientation Estimation Module
Calculates the rotation angle of a detected box.
"""

import cv2
import numpy as np


def calculate_orientation(rect):
    """
    Calculate orientation angle from minimum area rectangle.
    Returns the angle between the x-axis (horizontal) and the longer side of the box.
    
    Args:
        rect: Tuple from cv2.minAreaRect() containing:
              ((center_x, center_y), (width, height), angle)
              
    Returns:
        angle: Rotation angle in degrees (0-90) between x-axis and longer side
    """
    if rect is None:
        return None
    
    (cx, cy), (width, height), min_area_angle = rect
    
    # Get box corners to calculate actual orientation of longest side
    box_corners = get_box_corners(rect)
    if box_corners is not None:
        corners = np.array(box_corners, dtype=np.float32)
        
        # Find the longest side
        # Check all 4 edges
        edges = []
        for i in range(4):
            edge_vec = corners[(i + 1) % 4] - corners[i]
            edge_length = np.linalg.norm(edge_vec)
            edges.append((edge_vec, edge_length, i))
        
        # Get the longest edge (longer side of the box)
        longest_edge_info = max(edges, key=lambda x: x[1])
        longest_edge = longest_edge_info[0]
        
        # Calculate angle of longer side relative to x-axis
        # atan2(y, x) gives angle in radians
        # Positive x-axis is at 0°, positive y-axis is at 90°
        angle_rad = np.arctan2(longest_edge[1], longest_edge[0])
        angle_deg = np.degrees(angle_rad)
        
        # Normalize to [0, 90] degrees range
        # This represents the acute angle between the longer side and x-axis
        # For a rectangle, we want the angle between 0° (horizontal) and 90° (vertical)
        if angle_deg < 0:
            angle_deg = abs(angle_deg)
        
        # If angle > 90, take the complement (since it's a rectangle, 120° = 60°)
        if angle_deg > 90:
            angle_deg = 180 - angle_deg
        
        # Ensure it's in [0, 90] range
        angle_deg = max(0, min(90, angle_deg))
        
        return angle_deg
    
    # Fallback: use the angle from minAreaRect directly
    # Convert from [-90, 0] to [0, 90]
    normalized_angle = abs(min_area_angle)
    if normalized_angle > 90:
        normalized_angle = 90 - (normalized_angle - 90)
    return normalized_angle


def get_box_corners(rect):
    """
    Get the four corner points of the rotated rectangle.
    
    Args:
        rect: Tuple from cv2.minAreaRect()
        
    Returns:
        box: Array of 4 corner points
    """
    if rect is None:
        return None
    
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    return box


def get_box_center(rect):
    """
    Get the center point of the rectangle.
    
    Args:
        rect: Tuple from cv2.minAreaRect()
        
    Returns:
        center: Tuple (x, y) of center point
    """
    if rect is None:
        return None
    
    (cx, cy), _, _ = rect
    return (int(cx), int(cy))

