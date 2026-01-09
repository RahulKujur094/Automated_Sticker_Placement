# Automated Sticker Placement (Computer Vision)

![annotated_Image_30_light](https://github.com/user-attachments/assets/2ae18836-97b5-46b4-a756-7ed467008cb0)

![annotated_Image-raw](https://github.com/user-attachments/assets/24157029-45e1-4c50-bfdb-1562ddb15c45)


## 🎯 Goal

Given an image of a shoebox:
- **Detect orientation** (rotation angle)
- **Find (x, y) position** for optimal sticker placement

## 🧠 Key Insight

A shoebox is a **rectangle** → orientation can be solved using **classical Computer Vision**, no ML needed!

This approach is:
- ✅ **Fast** - Real-time processing
- ✅ **Robust** - Works with different lighting and distances
- ✅ **Simple** - No training data required
- ✅ **Explainable** - Clear geometric reasoning

## 📁 Project Structure

```
sticker_placement/
│── images/
│   ├── box_0.jpg      # Box at 0° rotation
│   ├── box_45.jpg     # Box at 45° rotation
│   └── ...            # More test images
│
│── src/
│   ├── __init__.py
│   ├── detect_box.py      # Box detection using contours
│   ├── orientation.py      # Angle estimation
│   ├── sticker_position.py # Sticker coordinate calculation
│   └── process_image.py    # Main pipeline script
│
│── output/
│   └── annotated_images/   # Output with visualizations
│
│── README.md
│── requirements.txt
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Usage

**Process a single image:**
```bash
cd sticker_placement
python src/process_image.py images/box_0.jpg output/annotated_images
```

**Process all images in a directory:**
```bash
python src/process_image.py --dir images output/annotated_images
```

**From Python code:**
```python
from src.process_image import process_image

results = process_image('images/box_45.jpg', 'output/annotated_images')
print(f"Angle: {results['angle']}°")
print(f"Position: {results['position']}")
```

## 🔍 Step-by-Step CV Pipeline

### 1️⃣ Preprocessing
```
RGB Image → Grayscale → Gaussian Blur → Canny Edge Detection
```

**Purpose:** Reduce noise and extract edges for contour detection.

### 2️⃣ Box Detection

- Find all contours using `cv2.findContours()`
- Filter for rectangular shapes using `cv2.approxPolyDP()`
- Select the largest rectangular contour (assumed to be the box)

**Why this works:** Shoeboxes have clear rectangular edges that stand out from the background.

### 3️⃣ Orientation Estimation

Use **minimum area rectangle**:
```python
rect = cv2.minAreaRect(contour)
angle = rect[2]  # Rotation angle
```

**Key insight:** `cv2.minAreaRect()` finds the smallest rectangle that can enclose the contour. The angle of this rectangle directly gives us the box's orientation.

**Angle normalization:**
- `minAreaRect` returns angles in range `[-90°, 0°]`
- We normalize to `[0°, 90°]` for intuitive interpretation

### 4️⃣ Sticker Placement Coordinates

Calculate position based on:
- **Center of rectangle:** `(cx, cy)`
- **Offset:** Slight upward offset (configurable percentage of box height)
- **Final position:** `(cx, cy - offset)`

**Why center + offset?**
- Center ensures sticker is on the box surface
- Upward offset places it on the top face (most visible)

### 5️⃣ Visual Output

The pipeline draws:
- 🟢 **Green outline:** Detected box boundary
- 🔵 **Blue contour:** Original detected contour (for debugging)
- 🟡 **Yellow dot:** Box center point
- 🟢 **Green circle:** Sticker placement position
- 🔴 **Red text:** Angle and coordinates

## 🧪 Data Collection Guidelines

To capture good test images:

1. **Place shoebox** on floor/table with clear background
2. **Capture at different angles:**
   - 0° (aligned with image axes)
   - 30°, 45°, 60° (diagonal orientations)
   - 90° (rotated 90 degrees)
3. **Vary conditions:**
   - Different lighting (bright, dim, shadows)
   - Different distances (close-up, far away)
   - Different backgrounds (contrasting colors work best)

## 🏭 Real-World Application: Conveyor Systems

This CV pipeline maps directly to **industrial conveyor belt systems**:

### How it works in production:

1. **Camera Setup:** Overhead camera captures boxes on conveyor belt
2. **Real-time Processing:** Each box image is processed as it passes
3. **Robot Arm Control:** 
   - Orientation angle → Rotate sticker applicator
   - (x, y) position → Move to exact placement location
4. **Automation:** No human intervention needed

### Advantages over ML approaches:

| Classical CV | Machine Learning |
|-------------|------------------|
| ✅ No training data | ❌ Requires labeled dataset |
| ✅ Fast inference | ⚠️ Slower (neural network) |
| ✅ Works immediately | ❌ Needs training time |
| ✅ Explainable | ⚠️ Black box |
| ✅ Robust to lighting | ⚠️ May need data augmentation |

### Robustness Features:

- **Rotation invariant:** Works at any angle (0-90°)
- **Scale invariant:** Handles different box sizes
- **Lighting tolerant:** Edge detection works in various conditions
- **Background flexible:** Works with contrasting backgrounds

## 📊 Output Format

For each processed image, the system outputs:

```
==================================================
Image: box_45.jpg
Orientation Angle: 45.23°
Sticker Position: (320, 180)
==================================================
```

Annotated images are saved with:
- Visual markers for box, center, and sticker position
- Text overlay showing angle and coordinates

## 🔧 Configuration

Adjust sticker placement offset in `sticker_position.py`:

```python
# Default: 10% offset from center
sticker_pos = calculate_sticker_position(rect, offset_percent=0.1)

# Custom: 5% offset
sticker_pos = calculate_sticker_position(rect, offset_percent=0.05)
```

## 🐛 Troubleshooting

**No box detected:**
- Ensure good contrast between box and background
- Check that box edges are clear and visible
- Try adjusting Canny thresholds in `detect_box.py`

**Wrong angle detected:**
- Verify box is clearly rectangular (not occluded)
- Ensure lighting is even (avoid harsh shadows)
- Check that box is the largest object in frame

**Sticker position off:**
- Adjust `offset_percent` parameter
- Verify box center detection is correct
- Check image resolution (higher = more accurate)

## 📝 Technical Details

### Why Classical CV (Not ML)?

1. **Geometric Problem:** Box orientation is a pure geometric calculation
2. **Deterministic:** Same input → same output (no randomness)
3. **Fast:** Edge detection + contour analysis is milliseconds
4. **No Training:** Works immediately on any rectangular object
5. **Interpretable:** Every step is explainable

### Algorithm Complexity

- **Time:** O(n) where n = number of pixels
- **Space:** O(n) for edge detection
- **Real-time capable:** Processes 30+ FPS on modern hardware

## 🎓 Learning Resources

- [OpenCV Contour Detection](https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html)
- [Minimum Area Rectangle](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga3d476a3417130ae5154aea421ca7ead9)
- [Canny Edge Detection](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html)

## 📄 License

This project is provided as-is for educational and industrial use.

---

**Built with:** OpenCV, NumPy  
**Approach:** Classical Computer Vision  
**Use Case:** Automated sticker placement on conveyor systems


