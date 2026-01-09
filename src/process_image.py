"""
Main Pipeline Script
Combines all modules to process images and detect sticker placement.
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from detect_box import detect_box
from orientation import calculate_orientation, get_box_center, get_box_corners
from sticker_position import calculate_sticker_position


def draw_axes(image, center):
    """
    Draw x-y axes (crosshair) centered at a point, extending across the image.
    
    Args:
        image: Image to draw on
        center: Center point (x, y)
        
    Returns:
        image: Image with axes drawn
    """
    if center is None:
        return image
    
    cx, cy = center
    h, w = image.shape[:2]
    
    # Draw horizontal line (x-axis) - green, extending full width
    cv2.line(image, (0, cy), (w, cy), (0, 255, 0), 2)
    
    # Draw vertical line (y-axis) - green, extending full height
    cv2.line(image, (cx, 0), (cx, h), (0, 255, 0), 2)
    
    # Draw axis labels at edges
    cv2.putText(image, 'X', (w - 30, cy - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, 'Y', (cx + 10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return image


def overlay_sticker_image(image, sticker_path, position, angle=None, scale=1.0):
    """
    Overlay a sticker image at the specified position with optional rotation.
    
    Args:
        image: Base image to overlay on
        sticker_path: Path to sticker image file
        position: (x, y) position to place sticker (center of sticker)
        angle: Rotation angle in degrees (optional, matches box orientation)
        scale: Scale factor for sticker size (default: 1.0)
        
    Returns:
        image: Image with sticker overlaid
    """
    if position is None:
        return image
    
    # Check if sticker image exists
    if not os.path.exists(sticker_path):
        print(f"Warning: Sticker image not found at {sticker_path}, using default marker")
        # Fallback to red circle
        sx, sy = position
        cv2.circle(image, (sx, sy), 8, (0, 0, 255), -1)
        cv2.circle(image, (sx, sy), 12, (0, 0, 255), 2)
        return image
    
    # Read sticker image
    sticker = cv2.imread(sticker_path, cv2.IMREAD_UNCHANGED)
    if sticker is None:
        print(f"Warning: Could not read sticker image from {sticker_path}, using default marker")
        sx, sy = position
        cv2.circle(image, (sx, sy), 8, (0, 0, 255), -1)
        cv2.circle(image, (sx, sy), 12, (0, 0, 255), 2)
        return image
    
    # Resize sticker if scale is not 1.0
    if scale != 1.0:
        h, w = sticker.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        sticker = cv2.resize(sticker, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Rotate sticker if angle is provided
    if angle is not None and angle != 0:
        h, w = sticker.shape[:2]
        center_rot = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center_rot, angle, 1.0)
        
        # Calculate new dimensions after rotation
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust rotation matrix for new center
        rotation_matrix[0, 2] += (new_w / 2) - center_rot[0]
        rotation_matrix[1, 2] += (new_h / 2) - center_rot[1]
        
        # Rotate sticker
        # Handle alpha channel separately if present
        if sticker.shape[2] == 4:
            # Separate alpha channel for proper rotation
            sticker_rgb = sticker[:, :, :3]
            sticker_alpha = sticker[:, :, 3:4]
            rotated_rgb = cv2.warpAffine(sticker_rgb, rotation_matrix, (new_w, new_h), 
                                        flags=cv2.INTER_LINEAR, 
                                        borderMode=cv2.BORDER_CONSTANT, 
                                        borderValue=(0, 0, 0))
            rotated_alpha = cv2.warpAffine(sticker_alpha, rotation_matrix, (new_w, new_h), 
                                          flags=cv2.INTER_LINEAR, 
                                          borderMode=cv2.BORDER_CONSTANT, 
                                          borderValue=0)
            sticker = np.dstack([rotated_rgb, rotated_alpha])
        else:
            sticker = cv2.warpAffine(sticker, rotation_matrix, (new_w, new_h), 
                                    flags=cv2.INTER_LINEAR, 
                                    borderMode=cv2.BORDER_CONSTANT, 
                                    borderValue=(0, 0, 0))
    
    # Get sticker dimensions
    sh, sw = sticker.shape[:2]
    sx, sy = position
    
    # Calculate top-left corner for placement (centering sticker at position)
    x1 = sx - sw // 2
    y1 = sy - sh // 2
    x2 = x1 + sw
    y2 = y1 + sh
    
    # Get image dimensions
    img_h, img_w = image.shape[:2]
    
    # Calculate clipping bounds
    clip_x1 = max(0, -x1)
    clip_y1 = max(0, -y1)
    clip_x2 = min(sw, img_w - x1)
    clip_y2 = min(sh, img_h - y1)
    
    # Check if sticker is within image bounds
    if x2 < 0 or y2 < 0 or x1 >= img_w or y1 >= img_h:
        return image
    
    # Extract region of interest
    roi_x1 = max(0, x1)
    roi_y1 = max(0, y1)
    roi_x2 = min(img_w, x2)
    roi_y2 = min(img_h, y2)
    
    roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
    
    # Extract corresponding region from sticker
    sticker_x1 = clip_x1
    sticker_y1 = clip_y1
    sticker_x2 = sticker_x1 + (roi_x2 - roi_x1)
    sticker_y2 = sticker_y1 + (roi_y2 - roi_y1)
    
    sticker_roi = sticker[sticker_y1:sticker_y2, sticker_x1:sticker_x2]
    
    # Handle alpha channel if present
    if sticker_roi.shape[2] == 4:
        # Sticker has alpha channel - blend with background
        alpha = sticker_roi[:, :, 3:4].astype(np.float32) / 255.0
        sticker_rgb = sticker_roi[:, :, :3].astype(np.float32)
        roi_float = roi.astype(np.float32)
        # Alpha blending: result = background * (1 - alpha) + foreground * alpha
        blended = roi_float * (1 - alpha) + sticker_rgb * alpha
        roi = blended.astype(np.uint8)
    else:
        # No alpha channel, simple overlay (replace pixels)
        roi = sticker_roi
    
    # Place modified ROI back
    image[roi_y1:roi_y2, roi_x1:roi_x2] = roi
    
    return image


def annotate_image(image, contour, rect, sticker_pos, angle, sticker_path=None):
    """
    Draw annotations on the image.
    
    Args:
        image: Original image
        contour: Detected box contour
        rect: Minimum area rectangle
        sticker_pos: Sticker position (x, y)
        angle: Orientation angle in degrees (relative to x-axis)
        sticker_path: Path to sticker image file (optional)
        
    Returns:
        annotated: Annotated image
    """
    annotated = image.copy()
    
    if rect is None:
        return annotated
    
    # Get box center for axes
    center = get_box_center(rect)
    
    # Draw x-y axes (crosshair) centered on box, extending across image
    if center is not None:
        annotated = draw_axes(annotated, center)
    
    # Draw box outline in red
    box = get_box_corners(rect)
    if box is not None:
        cv2.drawContours(annotated, [box], 0, (0, 0, 255), 2)  # Red color
    
    # Overlay sticker image at position
    if sticker_pos is not None:
        if sticker_path:
            # Use sticker image with rotation matching box orientation
            annotated = overlay_sticker_image(annotated, sticker_path, sticker_pos, angle, scale=0.3)
        else:
            # Fallback to red circle if no sticker image provided
            sx, sy = sticker_pos
            cv2.circle(annotated, (sx, sy), 8, (0, 0, 255), -1)  # Red filled circle
            cv2.circle(annotated, (sx, sy), 12, (0, 0, 255), 2)  # Red outline
    
    # Calculate and display orientation relative to axes
    if angle is not None:
        orientation_text = f'Orientation (x-axis to longer side): {angle:.1f}°'
        cv2.putText(annotated, orientation_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Draw coordinates (relative to image origin, top-left is (0,0))
    if sticker_pos is not None:
        coord_text = f'Position: ({sticker_pos[0]}, {sticker_pos[1]})'
        y_offset = 60 if angle is not None else 30
        cv2.putText(annotated, coord_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return annotated


def process_image(image_path, output_dir=None, sticker_path=None):
    """
    Process a single image and detect sticker placement.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save annotated image (optional)
        sticker_path: Path to sticker image file (optional)
        
    Returns:
        results: Dictionary with angle, position, and annotated image
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return None
    
    # Detect box
    contour, rect, edges = detect_box(image)
    
    if rect is None:
        print(f"Warning: No box detected in {image_path}")
        return None
    
    # Calculate orientation
    angle = calculate_orientation(rect)
    
    # Calculate sticker position
    sticker_pos = calculate_sticker_position(rect, offset_percent=0.1)
    
    # If sticker_path not provided, try to find default sticker
    if sticker_path is None:
        # Look for sticker in stickers directory relative to script
        script_dir = Path(__file__).parent.parent
        default_sticker = script_dir / "stickers" / "sticker.png"
        if default_sticker.exists():
            sticker_path = str(default_sticker)
        else:
            # Try other common formats
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                test_path = script_dir / "stickers" / f"sticker{ext}"
                if test_path.exists():
                    sticker_path = str(test_path)
                    break
    
    # Annotate image
    annotated = annotate_image(image, contour, rect, sticker_pos, angle, sticker_path)
    
    # Save annotated image if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"annotated_{filename}")
        cv2.imwrite(output_path, annotated)
        print(f"Saved annotated image to: {output_path}")
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Image: {os.path.basename(image_path)}")
    if angle is not None:
        print(f"Box Orientation (x-axis to longer side): {angle:.2f}°")
    if sticker_pos:
        print(f"Sticker Coordinates (x, y): ({sticker_pos[0]}, {sticker_pos[1]})")
    print(f"{'='*50}\n")
    
    return {
        'angle': angle,
        'position': sticker_pos,
        'annotated_image': annotated,
        'rect': rect
    }


def process_directory(input_dir, output_dir):
    """
    Process all images in a directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save annotated images
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f"*{ext}"))
        image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} image(s) to process\n")
    
    for image_path in image_files:
        process_image(str(image_path), output_dir, sticker_path=None)


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) < 2:
        print("Usage: python process_image.py <image_path> [output_dir]")
        print("   or: python process_image.py --dir <input_dir> <output_dir>")
        sys.exit(1)
    
    if sys.argv[1] == "--dir":
        # Process directory
        if len(sys.argv) < 4:
            print("Usage: python process_image.py --dir <input_dir> <output_dir>")
            sys.exit(1)
        input_dir = sys.argv[2]
        output_dir = sys.argv[3]
        process_directory(input_dir, output_dir)
    else:
        # Process single image
        image_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
        sticker_path = sys.argv[3] if len(sys.argv) > 3 else None
        process_image(image_path, output_dir, sticker_path)

