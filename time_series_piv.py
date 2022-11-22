import numpy as np

from base_piv import BasePIVAnalysis
from image_handling import get_array_from_file, get_sample_filename, display_image_array


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


    def process(self, window_size, vector_spacing, sutract_background=True):
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

        # Determine vector locations
        self.set_up_vectors(window_size, vector_spacing)

        # Compute velocities
        self.compute_velocities()

        # Compute vorticities
        self.calculate_vorticities()


    def perform_background_subtraction(self):
        """Performs average background subtraction on the data."""

        # Compute average
        self.average_image = np.average(self.time_series_array, axis=0)

        # Subtract
        self.subtracted_data = self.time_series_array - self.average_image

    
    def compute_frame_velocities(self, i):
        """Calculates the velocities from two frames.

        Parameters
        ----------
        i : int
            Index of first frame.
        """

        V = np.zeros((self.N_vels_in_y, self.N_vels_in_x, 2))

        # Loop through in x direction
        for j in range(self.N_vels_in_x):

            # Loop through in y direction
            for k in range(self.N_vels_in_y):
                    
                # Figure out our window indices
                j0 = j*self.vector_spacing
                j1 = j0 + self.window_size
                k0 = k*self.vector_spacing
                k1 = k0 + self.window_size

                # Calculate offset
                j_offset = int(self.V[i,j,k,0]*self.dt/self.dx)
                k_offset = int(-self.V[i,j,k,1]*self.dt/self.dy)

                # Get windows
                window1 = self.time_series_array[i,j0:j1,k0:k1]
                window2 = self.time_series_array[i+1,j0+j_offset:j1+j_offset,k0+k_offset:k1+k_offset]

                # Cross-correlate
                peak = get_correlation_peak(window1, window2)
                V[j,k,0] = -peak[1]*self.dx/self.dt
                V[j,k,1] =  peak[0]*self.dy/self.dt

        return V


    def compute_velocities(self):
        """Computes the raw velocities from the given time-series data."""

        # Initialize memory
        self.V = np.zeros((self.N-1, self.N_vels_in_x, self.N_vels_in_y, 2))

        # Window offset loop
        for iteration in range(2):

            # Calculate velocities
            prog = OneLineProgress(self.N-1, "Calculating {0} velocity vectors for {1} samples...".format(self.N_vels_in_x*self.N_vels_in_y, self.N-1))
            for i in range(self.N-1):
                self.V[i] = self.compute_frame_velocities(i)
                prog.display()

            # Apply median filter
            prog = OneLineProgress(self.N-1, "Applying median filter...")
            for i in range(self.N-1):
                self.V[i] = apply_median_filter(self.V[i], 2.0, 1.0)
                prog.display()

    
    def calculate_vorticities(self):
        """Calculates the vorticities from the data."""

        # Initialize storage
        self.zeta = np.zeros((self.N-1, self.N_vels_in_x, self.N_vels_in_y))

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
                for j in range(self.N_vels_in_x):
                    for k in range(self.N_vels_in_y):
                        print("{0},{1},{2},{3},{4},{5}".format(self.x_vec[j], self.y_vec[k], 0.0, self.V[i,j,k,0], self.V[i,j,k,1], self.zeta[i,j,k]), file=output_handle)

            prog.display()