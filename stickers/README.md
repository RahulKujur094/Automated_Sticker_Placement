# Sticker Images

Place your sticker image files in this directory.

## Supported Formats
- PNG (recommended - supports transparency)
- JPG/JPEG
- BMP

## Default Sticker
The system will look for a file named `sticker.png` (or `sticker.jpg`, etc.) in this directory.

If no sticker image is found, the system will fall back to a red circle marker.

## Usage
1. Place your sticker image file in this directory
2. Name it `sticker.png` (or any supported format)
3. The sticker will be automatically:
   - Rotated to match the box orientation
   - Scaled appropriately
   - Placed at the calculated position

## Custom Sticker Path
You can also specify a custom sticker path when running the script:
```bash
python src/process_image.py images/box.jpg output/annotated_images path/to/custom/sticker.png
```


