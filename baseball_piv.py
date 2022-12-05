import cv2
import os
import math as m
import numpy as np

from base_piv import BasePIVAnalysis
from image_handling import get_double_image_from_file, display_image_array


class BaseballPIVAnalysis(BasePIVAnalysis):
    """A class for PIV analysis on baseballs.
    """

    def __init__(self, filename, dt, pixel_threshold=np.inf, remove_baseball=True, D_baseball=2.9/12.0, scale_vals=0.01):

        # Get image arrays
        image1, image2 = get_double_image_from_file(filename)
        self.Ny, self.Nx = image1.shape
        self.N = 1
        print("Got {0} x {1} image.".format(self.Nx, self.Ny))

        # Store other params
        self.dt = dt

        # Initialize array
        self.data = np.zeros((2, self.Ny, self.Nx))
        self.data[0] = image1*scale_vals
        self.data[1] = image2*scale_vals
        self.data = np.where(self.data > 50, 50, self.data)

        print("Detecting baseball...")

        # Load images
        cv2.imwrite("temp.png", self.data[0])
        img1 = cv2.imread("temp.png", flags=0)
        cv2.imwrite("temp.png", self.data[1])
        img2 = cv2.imread("temp.png", flags=0)
        os.remove("temp.png")

        # Locate circles
        image_dim = min(self.Nx, self.Ny)
        baseball1 = cv2.HoughCircles(img1, cv2.HOUGH_GRADIENT, 1.3, 100, param2=60, minRadius=image_dim//4, maxRadius=image_dim)
        baseball2 = cv2.HoughCircles(img2, cv2.HOUGH_GRADIENT, 1.3, 100, param2=60, minRadius=image_dim//4, maxRadius=image_dim)
        baseball1 = baseball1.flatten()
        baseball2 = baseball2.flatten()
        self.R_baseball = 0.5*(baseball1[2] + baseball2[2])

        # Scale image to match baseball diameter and make pixels square
        self.x_lims = [0.0, self.Nx/(2.0*self.R_baseball)*D_baseball]
        self.y_lims = [0.0, self.Ny**2/(self.Nx*2.0*self.R_baseball)*D_baseball]

        # Remove baseball
        if remove_baseball:

            print("Removing baseballs from images...")

            # Get rid of pixels within baseball
            indices = np.indices((self.Ny, self.Nx))
            within_ball_1 = np.ceil((indices[0]-baseball1[1])**2) + np.ceil((indices[1]-baseball1[0])**2) < baseball1[2]**2
            within_ball_2 = np.ceil((indices[0]-baseball2[1])**2) + np.ceil((indices[1]-baseball2[0])**2) < baseball2[2]**2
            self.data[0] = np.where(within_ball_1, 0.0, self.data[0])
            self.data[1] = np.where(within_ball_2, 0.0, self.data[1])

        # Threshold image
        self.data = np.where(self.data > pixel_threshold, 0.0, self.data)
        display_image_array(self.data[0])
        display_image_array(self.data[1])



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