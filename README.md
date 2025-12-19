# imprepro - Image Processing Pro

**imprepro** is a powerful, user-friendly image analysis and processing tool. It allows you to apply various filters, adjust color properties, and perform geometric operations with real-time visual feedback and synchronized views.

## Features

- **Dual View**: Synchronized "Original" and "Processed" image views with absolute zoom and pan.
- **Geometric Operations**: Resize, Rotate, Flip, and Crop.
- **Color Adjustments**: Brightness, Contrast, Saturation, Hue, Gamma.
- **Filters**: Stackable filters including:
  - Gaussian, Median, Bilateral Blur
  - Sobel, Laplacian, Canny Edge Detection
  - Sharpen, Erosion, Dilation, Opening, Closing
  - **Binary Threshold** (New!)
- **Analytics**: View real-time image statistics (Mean/Std Dev, Histogram data, etc.).
- **Import/Export**: Save your settings as JSON presets and apply them to other images (preserves original image geometry).

## Installation

### From Source

1.  Ensure you have Python 3.8+ installed.
2.  Install dependencies:
    ```bash
    pip install opencv-python numpy PySide6
    ```
3.  Run the application:
    ```bash
    python main.py
    ```

### Standalone Executable

Simply run `imprepro.exe`. No installation required.

## Usage Guide

1.  **Load Image**:

    - Click **"Open Image"** or drag and drop an image onto the window.
    - Supported formats: JPG, PNG, BMP, etc.

2.  **Adjust View**:

    - Use the mouse wheel to **Zoom**.
    - Click and drag to **Pan**.
    - Check "Synchronize Zoom/Pan" in the sidebar to keep both views aligned.

3.  **Apply Operations**:

    - **Geometry**: Use the top bar to Rotate or Flip. Use the sidebar to Resize.
    - **Crop**: Draw a rectangle on the left image and click **"Crop to Selection"**.
    - **Color**: Adjust sliders in the "Color Adjustments" panel.
    - **Filters**: Click **"+ Add Filter"**, select a filter (e.g., Threshold), and adjust parameters.

4.  **Analytics**:

    - Enable "Show Analytics Panel" in the "View Settings" section to see detailed image data.

5.  **Save/Export**:
    - **"Save Image"**: Save the processed result.
    - **"Export Settings"**: Save your current processing pipeline to a JSON file.
    - **"Import Settings"**: Load a JSON file to apply the same look to a new image (geometry is preserved).

## settings.json Structure

When exporting settings, the file contains:

- `adjustments`: Color slider values.
- `filters`: List of active filters and their parameters.
- `geometric`: (Ignored on import to preserve image shape).
