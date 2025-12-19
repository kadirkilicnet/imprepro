import cv2
import numpy as np

def resize_image(image, width=None, height=None, method=cv2.INTER_LINEAR):
    """
    Resize image while optionally preserving aspect ratio.
    If both width and height are provided, ignore aspect ratio unless managed by caller.
    If only one is provided, calculate other to maintain aspect ratio.
    """
    if image is None: 
        return None
    
    (h, w) = image.shape[:2]
    
    if width is None and height is None:
        return image
        
    dim = None
    
    if width is None:
        # Calculate ratio
        r = height / float(h)
        dim = (int(w * r), height)
    elif height is None:
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        dim = (width, height)
        
    return cv2.resize(image, dim, interpolation=method)

def get_image_stats(image):
    """
    Returns a dictionary of statistics for the image.
    Color: Mean/Std per channel (RGB and HSV if applicable)
    Intensity: Min/Max/Mean/Std
    """
    if image is None:
        return {}
    
    stats = {}
    stats['shape'] = image.shape
    stats['dtype'] = str(image.dtype)
    stats['size'] = image.size
    
    # Helper to safe cast
    def safe_float(v):
        return float(v) if not np.isnan(v) else 0.0
    def safe_int(v):
        return int(v)

    # Check if color or grayscale
    if len(image.shape) == 3:
        # BGR
        b, g, r = cv2.split(image)
        stats['mean_r'] = safe_float(np.mean(r))
        stats['std_r'] = safe_float(np.std(r))
        stats['mean_g'] = safe_float(np.mean(g))
        stats['std_g'] = safe_float(np.std(g))
        stats['mean_b'] = safe_float(np.mean(b))
        stats['std_b'] = safe_float(np.std(b))
        
        # Whole intensity (grayscale equivalent)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        stats['min_intensity'] = safe_int(np.min(gray))
        stats['max_intensity'] = safe_int(np.max(gray))
        stats['mean_intensity'] = safe_float(np.mean(gray))
        stats['std_intensity'] = safe_float(np.std(gray))
        
        # HSV Stats
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        stats['mean_hue'] = safe_float(np.mean(h))
        stats['mean_sat'] = safe_float(np.mean(s))
        stats['mean_val'] = safe_float(np.mean(v))
        
    else:
        # Grayscale
        stats['min_intensity'] = safe_int(np.min(image))
        stats['max_intensity'] = safe_int(np.max(image))
        stats['mean_intensity'] = safe_float(np.mean(image))
        stats['std_intensity'] = safe_float(np.std(image))
        
    return stats

def rotate_image(image, angle):
    """Rotate image by angle in degrees."""
    if angle == 0: return image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def crop_image(image, x, y, w, h):
    """Crop image to specified rectangle."""
    if image is None: return None
    ih, iw = image.shape[:2]
    x2, y2 = x + w, y + h
    # Clamp to image boundaries
    x, y = max(0, x), max(0, y)
    x2, y2 = min(iw, x2), min(ih, y2)
    if x2 <= x or y2 <= y: return image
    return image[y:y2, x:x2]

def flip_image(image, flip_code):
    """
    Flip image: 0 for vertical, 1 for horizontal, -1 for both.
    """
    if flip_code is None: return image
    return cv2.flip(image, flip_code)

def apply_affine(image, matrix_str):
    """Apply affine transform from 2x3 matrix string."""
    try:
        vals = [float(x.strip()) for x in matrix_str.split(',')]
        if len(vals) != 6: return image
        M = np.array(vals).reshape((2, 3))
        (h, w) = image.shape[:2]
        return cv2.warpAffine(image, M, (w, h))
    except:
        return image

def apply_perspective(image, matrix_str):
    """Apply perspective transform from 3x3 matrix string."""
    try:
        vals = [float(x.strip()) for x in matrix_str.split(',')]
        if len(vals) != 9: return image
        M = np.array(vals).reshape((3, 3))
        (h, w) = image.shape[:2]
        return cv2.warpPerspective(image, M, (w, h))
    except:
        return image

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    import sys
    import os
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
