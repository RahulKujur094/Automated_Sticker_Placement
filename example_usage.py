"""
Example usage script for sticker placement detection.
Run this to see how to use the CV pipeline programmatically.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from process_image import process_image

if __name__ == "__main__":
    # Example: Process a single image
    # Make sure you have an image in the images/ directory first
    
    image_path = "images/box_0.jpg"  # Change this to your image path
    output_dir = "output/annotated_images"
    
    print("Processing image...")
    results = process_image(image_path, output_dir)
    
    if results:
        print("\n✅ Detection successful!")
        print(f"   Angle: {results['angle']:.2f}°")
        print(f"   Position: {results['position']}")
    else:
        print("\n❌ No box detected. Make sure:")
        print("   - Image has a clear rectangular box")
        print("   - Good contrast between box and background")
        print("   - Box is the largest object in the frame")


