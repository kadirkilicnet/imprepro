import cv2
import numpy as np
from abc import ABC, abstractmethod

class Filter(ABC):
    """
    Abstract Base Class for Image Filters.
    """
    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def description(self):
        pass

    @abstractmethod
    def get_params(self):
        """
        Returns a list of parameter dictionaries.
        Format:
        {
            'name': 'kernel_size',
            'type': 'int' | 'float' | 'option' | 'matrix' | 'text',
            'label': 'Kernel Size',
            'default': 5,
            'min': 1,       # Optional for numbers
            'max': 31,      # Optional for numbers
            'step': 2,      # Optional
            'options': [],  # For 'option' type
        }
        """
        pass

    @abstractmethod
    def apply(self, image, params):
        """
        Apply filter to image using params.
        """
        pass

class GaussianBlur(Filter):
    name = "Gaussian Blur"
    description = "Smooths image using Gaussian kernel to reduce noise."

    def get_params(self):
        return [
            {'name': 'ksize', 'type': 'int', 'label': 'Kernel Size', 'default': 5, 'min': 1, 'max': 51, 'step': 2},
            {'name': 'sigma', 'type': 'float', 'label': 'Sigma', 'default': 0.0, 'min': 0.0, 'max': 20.0, 'step': 0.1}
        ]

    def apply(self, image, params):
        k = params['ksize']
        if k % 2 == 0: k += 1 # Ensure odd
        s = params['sigma']
        return cv2.GaussianBlur(image, (k, k), s)

class SobelFilter(Filter):
    name = "Sobel Edge Detection"
    description = "Detects edges using Sobel operator."

    def get_params(self):
        return [
            {'name': 'ksize', 'type': 'int', 'label': 'Kernel Size', 'default': 3, 'min': 1, 'max': 7, 'step': 2},
            {'name': 'direction', 'type': 'option', 'label': 'Direction', 'default': 'Magnitude', 'options': ['X-Axis', 'Y-Axis', 'Magnitude']},
        ]

    def apply(self, image, params):
        k = params['ksize']
        direction = params['direction']
        
        gray = image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        dx, dy = 0, 0
        if direction == 'X-Axis':
            dx = 1
            res = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
            return cv2.convertScaleAbs(res)
        elif direction == 'Y-Axis':
            dy = 1
            res = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
            return cv2.convertScaleAbs(res)
        
        # Magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
        
        # Calculate magnitude
        grad_x_abs = cv2.convertScaleAbs(grad_x)
        grad_y_abs = cv2.convertScaleAbs(grad_y)
        
        # We can also compute true magnitude: sqrt(x^2 + y^2)
        # But commonly in standard viz we can blend them
        mag = cv2.addWeighted(grad_x_abs, 0.5, grad_y_abs, 0.5, 0)
        return mag

class LaplacianFilter(Filter):
    name = "Laplacian"
    description = "Detects edges using Laplacian operator."

    def get_params(self):
        return [
            {'name': 'ksize', 'type': 'int', 'label': 'Kernel Size', 'default': 3, 'min': 1, 'max': 7, 'step': 2},
        ]

    def apply(self, image, params):
        k = params['ksize']
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F, ksize=k))

class MedianBlur(Filter):
    name = "Median Blur"
    description = "Non-linear filter efficiently removes salt-and-pepper noise."

    def get_params(self):
        return [
            {'name': 'ksize', 'type': 'int', 'label': 'Kernel Size', 'default': 5, 'min': 1, 'max': 31, 'step': 2}
        ]

    def apply(self, image, params):
        k = params['ksize']
        if k % 2 == 0: k += 1
        return cv2.medianBlur(image, k)

class BilateralFilter(Filter):
    name = "Bilateral Filter"
    description = "Edge-preserving smoothing filter."

    def get_params(self):
        return [
            {'name': 'd', 'type': 'int', 'label': 'Diameter', 'default': 9, 'min': 1, 'max': 20, 'step': 1},
            {'name': 'sigmaColor', 'type': 'float', 'label': 'Sigma Color', 'default': 75.0, 'min': 10.0, 'max': 150.0, 'step': 5},
            {'name': 'sigmaSpace', 'type': 'float', 'label': 'Sigma Space', 'default': 75.0, 'min': 10.0, 'max': 150.0, 'step': 5}
        ]

    def apply(self, image, params):
        return cv2.bilateralFilter(image, params['d'], params['sigmaColor'], params['sigmaSpace'])

class CannyEdge(Filter):
    name = "Canny Edge Detection"
    description = "Multi-stage edge detector."

    def get_params(self):
        return [
            {'name': 'threshold1', 'type': 'int', 'label': 'Min Threshold', 'default': 100, 'min': 0, 'max': 255},
            {'name': 'threshold2', 'type': 'int', 'label': 'Max Threshold', 'default': 200, 'min': 0, 'max': 255}
        ]

    def apply(self, image, params):
        # Canny expects grayscale usually, but CV2 handles BGR fine too (uses Intensity)
        # However, for consistency we usually pass grayscale
        return cv2.Canny(image, params['threshold1'], params['threshold2'])

class SharpenFilter(Filter):
    name = "Sharpen"
    description = "Enhances edges and details."

    def get_params(self):
        return [
            {'name': 'strength', 'type': 'float', 'label': 'Strength', 'default': 1.0, 'min': 0.1, 'max': 5.0, 'step': 0.1}
        ]

    def apply(self, image, params):
        strength = params['strength']
        # Standard sharpen kernel
        #    0 -1  0
        #   -1  5 -1
        #    0 -1  0
        # For variable strength we can use:
        # Original + alpha * (Original - Gaussian) (Unsharp masking)
        # Or a weighted kernel. Let's do simple kernel for now.
        
        # Adjustable convolution kernel
        # base is identity
        # [0 0 0] [0 1 0] [0 0 0]
        # detail is Laplacian-like
        # [0 -1 0] [-1 4 -1] [0 -1 0]
        
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        
        # But this is fixed. Let's use weighted addWeighted approach which is unsharp mask
        # sharpened = original + strength * (original - blurred)
        gaussian = cv2.GaussianBlur(image, (0, 0), 3)
        return cv2.addWeighted(image, 1.0 + strength, gaussian, -strength, 0)

class CustomKernel(Filter):
    name = "Custom Kernel"
    description = "Convolve image with a custom 3x3 matrix."

    def get_params(self):
        return [
            # Flattened matrix params. 
            # In a real heavy app we might ask for a matrix input widget.
            # Here we might just hack it or ask for string.
            # Let's try 9 float inputs? A bit cluttered. parameters string?
            {'name': 'matrix', 'type': 'text', 'label': 'Kernel (Row-major, comma sep)', 'default': '0,-1,0,-1,5,-1,0,-1,0'}
        ]

    def apply(self, image, params):
        import numpy as np
        try:
            s = params['matrix']
            vals = [float(x.strip()) for x in s.split(',')]
            if len(vals) != 9:
                return image # invalid
            kernel = np.array(vals).reshape((3, 3))
            return cv2.filter2D(image, -1, kernel)
        except:
            return image

class ErosionFilter(Filter):
    name = "Erosion"
    description = "Morphological erosion - shrinks foreground objects."

    def get_params(self):
        return [
            {'name': 'ksize', 'type': 'int', 'label': 'Kernel Size', 'default': 3, 'min': 1, 'max': 31, 'step': 2},
            {'name': 'iterations', 'type': 'int', 'label': 'Iterations', 'default': 1, 'min': 1, 'max': 10}
        ]

    def apply(self, image, params):
        k = params['ksize']
        kernel = np.ones((k, k), np.uint8)
        return cv2.erode(image, kernel, iterations=params['iterations'])

class DilationFilter(Filter):
    name = "Dilation"
    description = "Morphological dilation - expands foreground objects."

    def get_params(self):
        return [
            {'name': 'ksize', 'type': 'int', 'label': 'Kernel Size', 'default': 3, 'min': 1, 'max': 31, 'step': 2},
            {'name': 'iterations', 'type': 'int', 'label': 'Iterations', 'default': 1, 'min': 1, 'max': 10}
        ]

    def apply(self, image, params):
        k = params['ksize']
        kernel = np.ones((k, k), np.uint8)
        return cv2.dilate(image, kernel, iterations=params['iterations'])

class OpeningFilter(Filter):
    name = "Opening"
    description = "Morphological opening - erosion followed by dilation (removes noise)."

    def get_params(self):
        return [
            {'name': 'ksize', 'type': 'int', 'label': 'Kernel Size', 'default': 3, 'min': 1, 'max': 31, 'step': 2}
        ]

    def apply(self, image, params):
        k = params['ksize']
        kernel = np.ones((k, k), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

class ClosingFilter(Filter):
    name = "Closing"
    description = "Morphological closing - dilation followed by erosion (fills holes)."

    def get_params(self):
        return [
            {'name': 'ksize', 'type': 'int', 'label': 'Kernel Size', 'default': 3, 'min': 1, 'max': 31, 'step': 2}
        ]

    def apply(self, image, params):
        k = params['ksize']
        kernel = np.ones((k, k), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

class FilterRegistry:
    _filters = {}

    @classmethod
    def register(cls, filter_class):
        instance = filter_class()
        cls._filters[instance.name] = instance

    @classmethod
    def get_filter(cls, name):
        return cls._filters.get(name)

    @classmethod
    def get_filter_names(cls):
        return sorted(list(cls._filters.keys()))

# Register all
class ThresholdFilter(Filter):
    name = "Threshold"
    description = "Applies binary thresholding to the image."

    def get_params(self):
        return [
            {'name': 'thresh', 'type': 'int', 'label': 'Threshold', 'default': 127, 'min': 0, 'max': 255},
            {'name': 'maxval', 'type': 'int', 'label': 'Max Value', 'default': 255, 'min': 0, 'max': 255},
            {'name': 'type', 'type': 'option', 'label': 'Type', 'default': 'Binary', 
             'options': ['Binary', 'Binary Inv', 'Trunc', 'ToZero', 'ToZero Inv']}
        ]

    def apply(self, image, params):
        thresh = params['thresh']
        maxval = params['maxval']
        t_str = params['type']
        
        type_map = {
            'Binary': cv2.THRESH_BINARY,
            'Binary Inv': cv2.THRESH_BINARY_INV,
            'Trunc': cv2.THRESH_TRUNC,
            'ToZero': cv2.THRESH_TOZERO,
            'ToZero Inv': cv2.THRESH_TOZERO_INV
        }
        t_val = type_map.get(t_str, cv2.THRESH_BINARY)
        
        gray = image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        _, res = cv2.threshold(gray, thresh, maxval, t_val)
        return res

FilterRegistry.register(GaussianBlur)
FilterRegistry.register(SobelFilter)
FilterRegistry.register(LaplacianFilter)
FilterRegistry.register(MedianBlur)
FilterRegistry.register(BilateralFilter)
FilterRegistry.register(CannyEdge)
FilterRegistry.register(ThresholdFilter) # Added
FilterRegistry.register(SharpenFilter)
FilterRegistry.register(CustomKernel)
FilterRegistry.register(ErosionFilter)
FilterRegistry.register(DilationFilter)
FilterRegistry.register(OpeningFilter)
FilterRegistry.register(ClosingFilter)
