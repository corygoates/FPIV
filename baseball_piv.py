import cv2
import os
import numpy as np

from base_piv import BasePIVAnalysis
from image_handling import get_double_image_from_file, display_image_array


class BaseballPIVAnalysis(BasePIVAnalysis):
    """A class for PIV analysis on baseballs.
    """

    def __init__(self, filename, dt, x_lims=None, y_lims=None, pixel_threshold=np.inf):

        # Get image arrays
        image1, image2 = get_double_image_from_file(filename)
        self.Ny, self.Nx = image1.shape
        self.N = 1

        # Store other params
        self.dt = dt
        self.x_lims = x_lims
        self.y_lims = y_lims
        if self.x_lims == None:
            self.x_lims = [0.0, 1.0]
            self.y_lims = [0.0, self.Ny/self.Nx] # Assume the pixels are square

        # Initialize array
        self.data = np.zeros((2, self.Ny, self.Nx))
        self.data[0] = image1
        self.data[1] = image2

        # Threshold image
        #self.data = np.where(self.data > pixel_threshold, 0.0, self.data)

        # Remove baseball
        cv2.imwrite("temp.png", self.data[0])
        img = cv2.imread("temp.png", flags=0)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.3, 100)


    def process(self, e_thresh, e0, window_size, vector_spacing, N_passes=1, max_shift_in_pixels=None):
        """Processes the data.

        Parameters
        ----------
        e_thresh : float
            Normalized threshold for filtering the data.

        e0 : float
            Normalizer (to avoid division by zero).

        window_size : int
            Interrogation window size.

        vector_spacing : int, optional
            Spacing between output vectors in pixels. Defaults to the interrogation window size.

        subtract_background : bool, optional
            Whether to perform average background subtraction. Defaults to True.

        N_passes : int, optional
            Number of window shifting passes to perform. Defaults to 1.
        
        max_shift_in_pixels : int, optional
            Displacement threshold for throwing out vectors which are too large. Defaults to keeping all vectors.
        """

        # Determine vector locations
        self.set_up_vectors(window_size, vector_spacing)

        # Calculate shifts
        self.calculate_shifts(e_thresh, e0, N_passes, max_shift_in_pixels=max_shift_in_pixels)

        # Compute velocities
        self.convert_shifts_to_velocities()

        # Compute vorticities
        self.calculate_vorticities()