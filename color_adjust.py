import cv2
import numpy as np

class ColorAdjuster:
    """
    Handles color adjustments for images including brightness, contrast,
    saturation, hue, and gamma correction.
    """

    @staticmethod
    def apply_brightness(image, value):
        """
        Applies additive brightness.
        value: int [-255, 255]
        """
        if value == 0:
            return image
        
        # Using cv2.add/subtract to handle saturation automatically (clipping 0-255)
        if value > 0:
            # Create a matrix of the value to add
            matrix = np.full(image.shape, value, dtype=np.uint8)
            return cv2.add(image, matrix)
        else:
            matrix = np.full(image.shape, abs(value), dtype=np.uint8)
            return cv2.subtract(image, matrix)

    @staticmethod
    def apply_contrast(image, value):
        """
        Applies multiplicative contrast.
        value: float (scale factor). 1.0 is original.
        """
        if value == 1.0:
            return image
        
        return cv2.convertScaleAbs(image, alpha=value, beta=0)

    @staticmethod
    def apply_hsv_adjustments(image, saturation_scale=1.0, hue_shift=0):
        """
        Applies Saturation and Hue adjustments in HSV space.
        saturation_scale: float, 1.0 is original.
        hue_shift: int, degrees to shift hue [0-180 in OpenCV Hue scale].
        """
        if saturation_scale == 1.0 and hue_shift == 0:
            return image

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Saturation
        if saturation_scale != 1.0:
            s = s.astype(np.float32)
            s = s * saturation_scale
            s = np.clip(s, 0, 255).astype(np.uint8)

        # Hue
        if hue_shift != 0:
            # OpenCV Hue is 0-179.
            # We treat input hue_shift as degrees (0-360) usually, but simple shift is okay.
            # If input is meant to be degrees, we might need to map it. 
            # Let's assume input is simple additive shift for now.
            h = h.astype(np.int16)
            h = h + hue_shift
            h = h % 180 # Wrap around
            h = h.astype(np.uint8)

        # Merge back
        final_hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def apply_gamma(image, gamma=1.0):
        """
        Applies Gamma Correction.
        gamma: float
        """
        if gamma == 1.0:
            return image

        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in range(256)]).astype("uint8")
        
        return cv2.LUT(image, table)

    @staticmethod
    def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Applies Contrast Limited Adaptive Histogram Equalization.
        """
        if clip_limit <= 0:
            return image
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        res_gray = clahe.apply(gray)
        
        # Merge back if colorful
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = res_gray
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def process(image, brightness=0, contrast=1.0, saturation=1.0, hue=0, gamma=1.0, clahe_limit=0.0):
        """
        Applies all color adjustments in a specific pipeline order.
        Order: Resize (handled externally) -> Color Adjustments -> Filters
        Internal Color Order: CLAHE -> Gamma -> Brightness/Contrast -> HSV
        """
        if image is None:
            return None
        
        # Make a copy to avoid modifying original
        processed = image.copy()

        # 0. CLAHE
        if clahe_limit > 0:
            processed = ColorAdjuster.apply_clahe(processed, clahe_limit)

        # 1. Gamma (Good to do on raw-ish intensities)
        processed = ColorAdjuster.apply_gamma(processed, gamma)

        # 2. Brightness & Contrast
        processed = ColorAdjuster.apply_contrast(processed, contrast)
        processed = ColorAdjuster.apply_brightness(processed, brightness)

        # 3. HSV Adjustments (Saturation, Hue)
        processed = ColorAdjuster.apply_hsv_adjustments(processed, saturation, hue)

        return processed
