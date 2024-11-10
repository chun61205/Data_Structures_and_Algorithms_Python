import cv2
import numpy as np

class HistogramEqualization:
    def __init__(self):
        pass

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Applying histogram equalization to a grayscale image
        
        Args:
            image (np.ndarray): Grayscale image as a 2D NumPy array
        
        Returns:
            np.ndarray: Equalized image
        """

        histogram = np.zeros(256, dtype=int)
        for pixel in image.ravel():
            histogram[pixel] += 1

        cdf = np.cumsum(histogram)

        cdfMin = cdf[cdf > 0].min()
        cdfNormalized = (cdf - cdfMin) / (cdf[-1] - cdfMin) * 255
        cdfNormalized = cdfNormalized.astype(np.uint8)

        equalizedImage = cdfNormalized[image]

        return equalizedImage