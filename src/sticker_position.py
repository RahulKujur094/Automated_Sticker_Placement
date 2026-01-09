"""
Sticker Position Calculation Module
Determines optimal (x, y) coordinates for sticker placement.
"""

import numpy as np
import cv2

# Handle both relative and absolute imports
try:
    from .orientation import get_box_center, get_box_corners
except ImportError:
    from orientation import get_box_center, get_box_corners


def calculate_sticker_position(rect, offset_percent=0.25):
    """
    Calculate sticker placement position on the box.
    Ensures the position is within the box boundaries using geometric constraints.
    
    Args:
        rect: Minimum area rectangle from cv2.minAreaRect()
        offset_percent: Offset from center as percentage (default: 0.25 = 25%)
        
    Returns:
        position: Tuple (x, y) for sticker placement (guaranteed to be on box)
    """
    if rect is None:
        return None
    
    # Get box center and corners
    center = get_box_center(rect)
    if center is None:
        return None
    
    cx, cy = center
    box_corners = get_box_corners(rect)
    
    if box_corners is None:
        return None
    
    # Convert box_corners to numpy array
    corners = np.array(box_corners, dtype=np.float32)
    center_pt = np.array([cx, cy], dtype=np.float32)
    
    # Find the corner closest to the "top" (minimum y in image coordinates)
    # This gives us a direction to offset towards
    top_corner_idx = np.argmin(corners[:, 1])
    top_corner = corners[top_corner_idx]
    
    # Calculate direction vector from center to top corner
    direction = top_corner - center_pt
    direction_length = np.linalg.norm(direction)
    
    if direction_length < 1:
        # Degenerate case, use center
        return (int(cx), int(cy))
    
    # Normalize direction
    direction_unit = direction / direction_length
    
    # Try different offsets from center, starting with the desired offset
    # and reducing if needed to ensure we stay inside the box
    for offset_ratio in np.linspace(offset_percent, 0.05, 30):
        # Calculate candidate position
        offset_vec = direction * offset_ratio
        candidate = center_pt + offset_vec
        candidate_pt = (int(candidate[0]), int(candidate[1]))
        
        # Check if this point is inside the box
        # pointPolygonTest returns: >0 if inside, =0 if on edge, <0 if outside
        test_result = cv2.pointPolygonTest(box_corners, candidate_pt, True)
        
        if test_result >= 0:
            # Point is inside or on the edge - use it
            return candidate_pt
    
    # If we couldn't find a point (shouldn't happen), use center
    # Center is always inside a convex polygon
    return (int(cx), int(cy))


def calculate_sticker_position_on_face(rect, face='top', offset_percent=0.1):
    """
    Calculate sticker position on a specific face of the box.
    
    Args:
        rect: Minimum area rectangle from cv2.minAreaRect()
        face: Which face to place sticker on ('top', 'bottom', 'left', 'right')
        offset_percent: Offset from center as percentage
        
    Returns:
        position: Tuple (x, y) for sticker placement
    """
    if rect is None:
        return None
    
    center = get_box_center(rect)
    if center is None:
        return None
    
    cx, cy = center
    (_, _), (width, height), angle = rect
    
    # Calculate offset
    offset = int(max(width, height) * offset_percent)
    
    if face == 'top':
        return (int(cx), int(cy - offset))
    elif face == 'bottom':
        return (int(cx), int(cy + offset))
    elif face == 'left':
        return (int(cx - offset), int(cy))
    elif face == 'right':
        return (int(cx + offset), int(cy))
    else:
        # Default to top
        return (int(cx), int(cy - offset))

