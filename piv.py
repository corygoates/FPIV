import os

import numpy as np
import scipy.signal as sig

from image_handling import get_array_from_file, get_sample_filename, display_image_array
from helpers import OneLineProgress


def get_correlation_peak(array1, array2):
    """Locates the correlation peak between two arrays.
    
    Parameters
    ----------
    array1 : ndarray
        First array.
        
    array2 : ndarray
        Second array.

    Returns
    -------
    list
        Coordinates of correlation peak.
    """

    # Cross-correlate
    avg1 = np.average(array1.flatten()).item()
    avg2 = np.average(array2.flatten()).item()
    corr = sig.correlate(array1-avg1, array2-avg2, method='fft', mode='same')

    # Find maximum
    max_corr = 0.0
    for  i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            if corr[i,j] > max_corr:
                i_max = i
                j_max = j
                max_corr = corr[i,j]

    return [i_max-array1.shape[0]//2, j_max-array1.shape[1]//2]


def get_frame_vorticity(V_array, dx, dy):
    """Calculates the vorticity of the given velocity field.

    Parameters
    ----------
    V_array : ndarray
        Array of the velocity components at each point in space.

    dx : float
        Dimensional vector spacing in the x-direction.

    dy : float
        Dimensional vector spacing in the y-direction.

    Returns
    -------
    ndarray
        Vorticity array.
    """

    # Initialize storage
    Ny, Nx, _ = V_array.shape
    zeta_array = np.zeros((Ny, Nx))

    # Loop
    for i in range(Ny):
        for j in range(Nx):

            # Get dv/dx
            if j == 0:
                dv_dx = (V_array[i,j+1,1] - V_array[i,j,1])/dx
            elif j == Nx-1:
                dv_dx = (V_array[i,j,1] - V_array[i,j-1,1])/dx
            else:
                dv_dx = 0.5*(V_array[i,j+1,1] - V_array[i,j-1,1])/dx

            # Get du/dy
            if i == 0:
                du_dy = (V_array[i,j,0] - V_array[i+1,j,0])/dy
            elif i == Ny-1:
                du_dy = (V_array[i-1,j,0] - V_array[i,j,0])/dy
            else:
                du_dy = (V_array[i-1,j,0] - V_array[i+1,j,0])/dy
        
            # Calculate vorticity
            zeta_array[i,j] = du_dy - dv_dx

    return zeta_array


class TimeSeriesPIVAnalysis:
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
        self.N = N_files

        # Store other params
        self.dt = dt
        self.x_lims = x_lims
        self.y_lims = y_lims

        # Initialize array
        self.time_series_array = np.zeros((N_files, self.Ny, self.Nx))

        # Now get the data
        prog = OneLineProgress(N_files, "Reading in data...")
        for i in range(N_files):
            filename = get_sample_filename(directory, file_root_name, extension, i+1, N_files)
            self.time_series_array[i] = get_array_from_file(filename)
            prog.display()


    def process(self, window_size, vector_spacing=None, sutract_background=True):
        """Processes the data.

        Parameters
        ----------
        window_size : int
            Interrogation window size.

        vector_spacing : int, optional
            Spacing between output vectors in pixels. Defaults to the interrogation window size.

        subtract_background : bool, optional
            Whether to perform average background subtraction. Defaults to True.

        """

        # Background subtraction
        if sutract_background:
            self.perform_background_subtraction()

        # Get vector spacing
        if vector_spacing == None:
            vector_spacing = window_size

        # Determien vector locations
        self.calc_vector_locations(window_size, vector_spacing)

        # Compute velocities
        self.compute_velocities(window_size, vector_spacing)

        # Compute vorticities
        self.calculate_vorticities()


    def perform_background_subtraction(self):
        """Performs average background subtraction on the data."""

        # Compute average
        self.average_image = np.average(self.time_series_array, axis=0)

        # Subtract
        self.subtracted_data = self.time_series_array - self.average_image


    def calc_vector_locations(self, window_size, vector_spacing):
        """Calculates the dimensional locations of each velocity vector.

        Parameters
        ----------
        vector_spacing : int
            Spacing between output vectors in pixels.

        window_size : int
            Interrogation window size.
        """

        # Get number of vectors
        self.Nu = (self.Nx-window_size)//vector_spacing + 1
        self.Nv = (self.Ny-window_size)//vector_spacing + 1

        # Determine pixel sizes
        self.dx = (self.x_lims[1] - self.x_lims[0]) / self.Nx
        self.dy = (self.y_lims[1] - self.y_lims[0]) / self.Ny

        # Determine dimensional vector spacing
        self.vec_spacing_x = vector_spacing*self.dx
        self.vec_spacing_y = vector_spacing*self.dy

        # Initialize location storage
        self.x_vec = np.zeros(self.Nu)
        self.y_vec = np.zeros(self.Nv)

        # Calculate x locations
        for i in range(self.Nu):
            self.x_vec[i] = (window_size//2 + i*vector_spacing)*self.dx

        # Calculate y locations
        for i in range(self.Nv):
            self.y_vec[i] = -(window_size//2 + i*vector_spacing)*self.dy


    def compute_velocities(self, window_size, vector_spacing):
        """Computes the cross-correlations of the given time-series data.

        Parameters
        ----------
        window_size : int
            Interrogation window size.

        vector_spacing : int, optional
            Spacing between output vectors in pixels. Defaults to the interrogation window size.
        """

        prog = OneLineProgress(self.Nu*self.Nv, "Calculating {0} velocity vectors for {1} samples...".format(self.Nu*self.Nv, self.N-1))

        # Initialize memory
        self.V = np.zeros((self.N-1, self.Nu, self.Nv, 2))

        # Loop through in x direction
        for j in range(self.Nu):

            # Loop through in y direction
            for k in range(self.Nv):
                    
                # Figure out our window indices
                j0 = j*vector_spacing
                j1 = j0 + window_size
                k0 = k*vector_spacing
                k1 = k0 + window_size

                # Loop through samples
                for i in range(self.N-1):

                    # Get windows
                    window1 = self.time_series_array[i,j0:j1,k0:k1]
                    window2 = self.time_series_array[i+1,j0:j1,k0:k1]

                    # Cross-correlate
                    peak = get_correlation_peak(window1, window2)
                    self.V[i,j,k,0] = -peak[1]*self.dx/self.dt
                    self.V[i,j,k,1] =  peak[0]*self.dy/self.dt

                prog.display()

    
    def calculate_vorticities(self):
        """Calculates the vorticities from the data."""

        # Initialize storage
        self.zeta = np.zeros((self.N-1, self.Nu, self.Nv))

        prog = OneLineProgress(self.N-1, "Calculating vorticities...")
        for i in range(self.N-1):
            self.zeta[i] = get_frame_vorticity(self.V[i], self.vec_spacing_x, self.vec_spacing_y)
            prog.display()


    def write_to_csv(self, output_file_root_name):
        """Writes the data to a series of csv files.
        
        Parameters
        ----------
        output_file_root_name : str
            Root name for the output files.
        """

        # Loop through data
        prog = OneLineProgress(self.N-1, "Writing velocity data to file...")
        for i in range(self.N-1):

            # Get filename
            filename = "{0}{1}.csv".format(output_file_root_name, str(i).zfill(len(str(self.N))))

            # Open file
            with open(filename, 'w') as output_handle:

                # Header
                print("x,y,z,u,v,zeta", file=output_handle)

                # Loop through points
                for j in range(self.Nu):
                    for k in range(self.Nv):
                        print("{0},{1},{2},{3},{4},{5}".format(self.x_vec[j], self.y_vec[k], 0.0, self.V[i,j,k,0], self.V[i,j,k,1], self.zeta[i,j,k]), file=output_handle)

            prog.display()