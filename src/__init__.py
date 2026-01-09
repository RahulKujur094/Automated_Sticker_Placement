"""
Sticker Placement Computer Vision Package
"""

from .detect_box import detect_box
from .orientation import calculate_orientation, get_box_center, get_box_corners
from .sticker_position import calculate_sticker_position, calculate_sticker_position_on_face

__all__ = [
    'detect_box',
    'calculate_orientation',
    'get_box_center',
    'get_box_corners',
    'calculate_sticker_position',
    'calculate_sticker_position_on_face',
]


