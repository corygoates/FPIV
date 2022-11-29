import numpy as np

from base_piv import BasePIVAnalysis
from image_handling import get_array_from_file, get_sample_filename, display_image_array
from helpers import OneLineProgress


class TimeSeriesPIVAnalysis(BasePIVAnalysis):
    """Class for performing an storing a PIV analysis on time-series data.
    Loads a set of time series data into a single array. Assumes the files are stored using 1-based indexing.

    Parameters
    ----------
    directory : str
        Folder the data are stored in.

    file_root_name : str
        First part of filename which does not change between the data.

    extension : str
        File extension.

    N_files : int
        Total number of files to read in.

    dt : float
        Timestep.

    x_lims : list
        Dimensional limits of the image array in the x-direction. Defaults to [0.0, 1.0].

    y_lims : list
        Dimensional limits of the image array in the y-direction. Defaults to [0.0, 1.0].
    """

    def __init__(self, directory, file_root_name, extension, N_files, dt, x_lims=[0.0, 1.0], y_lims=[0.0, 1.0]):

        # Get shape of image arrays
        filename = get_sample_filename(directory,file_root_name, extension, 1, N_files)
        image_array = get_array_from_file(filename)
        self.Ny, self.Nx = image_array.shape
        self.N = N_files-1

        # Store other params
        self.dt = dt
        self.x_lims = x_lims
        self.y_lims = y_lims

        # Initialize array
        self.data = np.zeros((N_files, self.Ny, self.Nx))

        # Now get the data
        prog = OneLineProgress(N_files, "Reading in data...")
        for i in range(N_files):
            filename = get_sample_filename(directory, file_root_name, extension, i+1, N_files)
            self.data[i] = get_array_from_file(filename)
            prog.display()


    def process(self, e_thresh, e0, window_size, vector_spacing, subtract_background=True, N_passes=1):
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
        """

        # Background subtraction
        if subtract_background:
            self.perform_background_subtraction()

        # Determine vector locations
        self.set_up_vectors(window_size, vector_spacing)

        # Calculate shifts
        self.calculate_shifts(e_thresh, e0, N_passes)

        # Compute velocities
        self.convert_shifts_to_velocities()

        # Compute vorticities
        self.calculate_vorticities()


    def perform_background_subtraction(self):
        """Performs average background subtraction on the data."""

        # Compute average
        self.average_image = np.average(self.data, axis=0)

        # Subtract
        self.subtracted_data = self.data - self.average_image