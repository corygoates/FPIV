import numpy as np
import scipy.interpolate as interp

from analysis import get_correlation_peak
from helpers import OneLineProgress

class BasePIVAnalysis:
    """Base class for performing PIV analysis."""


    def __init__(self):
        
        self.dt = None
        self.dx = None
        self.dy = None
        self.N = None
        self.data = None
        self.shifts = None
        self.V = None
        self.zeta = None
        self.N_vels_in_x = None
        self.N_vels_in_y = None
        self.vec_spacing_x = None
        self.vec_spacing_y = None


    def set_up_vectors(self, window_size, vector_spacing=None):
        """Initializes the shift and velocity vectors to be calculated.

        Parameters
        ----------
        window_size : int
            Interrogation window size.

        vector_spacing : int, optional
            Spacing between output vectors in pixels. Defaults to the interrogation window size.
        """

        self.window_size = window_size
        if vector_spacing == None:
            self.vector_spacing = self.window_size
        else:
            self.vector_spacing = vector_spacing

        # Get number of vectors
        self.N_vels_in_x = (self.Nx-self.window_size)//self.vector_spacing + 1
        self.N_vels_in_y = (self.Ny-self.window_size)//self.vector_spacing + 1

        # Determine pixel sizes
        self.dx = (self.x_lims[1] - self.x_lims[0]) / self.Nx
        self.dy = (self.y_lims[1] - self.y_lims[0]) / self.Ny

        # Get pixel locations
        self.x_pix = np.linspace(self.x_lims[0]+0.5*self.dx, self.x_lims[1]-0.5*self.dx, self.Nx)
        self.y_pix = np.linspace(self.y_lims[1]-0.5*self.dy, self.y_lims[0]+0.5*self.dy, self.Ny)

        # Determine dimensional vector spacing
        self.vec_spacing_x = self.vector_spacing*self.dx
        self.vec_spacing_y = self.vector_spacing*self.dy

        # Initialize location storage
        self.x_vec = np.zeros(self.N_vels_in_x)
        self.y_vec = np.zeros(self.N_vels_in_y)

        # Calculate x locations
        for i in range(self.N_vels_in_x):
            self.x_vec[i] = (self.window_size//2 + i*self.vector_spacing)*self.dx

        # Calculate y locations
        for i in range(self.N_vels_in_y):
            self.y_vec[i] = self.y_lims[1] - (self.window_size//2 + i*self.vector_spacing)*self.dy


    def convert_shifts_to_velocities(self):
        """Converts the stored shifts to velocities."""

        # Calculate velocities
        self.V = np.zeros_like(self.shifts)
        self.V[:,:,:,0] = -self.shifts[:,:,:,1]*self.dx/self.dt
        self.V[:,:,:,1] =  self.shifts[:,:,:,0]*self.dy/self.dt


    def calculate_shifts(self, e_thresh, e0, N_passes=0):
        """Calculates the shift fields between the raw data.
        
        Parameters
        ----------
        e_thresh : float
            Normalized threshold for filtering the data.

        e0 : float
            Normalizer (to avoid division by zero).

        N_passes : int, optional
            Number of passes using window offsetting. Defaults to 1.
        """

        # Initialize storage
        self.shifts = np.zeros((self.N, self.N_vels_in_y, self.N_vels_in_x, 2))

        # Loop through passes
        for i in range(N_passes):

            print("Pass ", i+1)

            # Loop through samples
            for l in range(self.N):

                # Loop through in x direction
                for j in range(self.N_vels_in_y):

                    # Loop through in y direction
                    for k in range(self.N_vels_in_x):

                        # Figure out our window indices
                        j0 = j*self.vector_spacing
                        j1 = j0 + self.window_size
                        k0 = k*self.vector_spacing
                        k1 = k0 + self.window_size

                        # Get window offset from previous pass
                        j_shift = int(round(self.shifts[l,j,k,0]))
                        k_shift = int(round(self.shifts[l,j,k,1]))

                        # Limit shifts to stay inside the data bounds
                        if j0 + j_shift < 0:
                            j_shift = -j0
                        elif j1 + j_shift > self.Ny:
                            j_shift = self.Ny - j1

                        if k0 + k_shift < 0:
                            k_shift = -k0
                        elif k1 + k_shift > self.Nx:
                            k_shift = self.Nx - k1

                        # Get windows
                        window1 = self.data[l,j0:j1,k0:k1]
                        window2 = self.data[l+1,j0+j_shift:j1+j_shift,k0+k_shift:k1+k_shift]

                        # Cross-correlate
                        self.shifts[l,j,k,:] = get_correlation_peak(window1, window2)

            # Filter
            if i != 0:
                self.filter_shifts(e_thresh, e0)


    def calculate_vorticities(self):
        """Calculates the vorticities from the data."""

        # Initialize storage
        self.zeta = np.zeros((self.N, self.N_vels_in_y, self.N_vels_in_x))

        # Loop
        for k in range(self.N):
            for i in range(self.N_vels_in_y):
                for j in range(self.N_vels_in_x):

                    # Get dv/dx
                    if j == 0:
                        dv_dx = (self.V[k,i,j+1,1] - self.V[k,i,j,1])/self.vec_spacing_x
                    elif j == self.N_vels_in_x-1:
                        dv_dx = (self.V[k,i,j,1] - self.V[k,i,j-1,1])/self.vec_spacing_x
                    else:
                        dv_dx = 0.5*(self.V[k,i,j+1,1] - self.V[k,i,j-1,1])/self.vec_spacing_x

                    # Get du/dy
                    if i == 0:
                        du_dy = (self.V[k,i,j,0] - self.V[k,i+1,j,0])/self.vec_spacing_y
                    elif i == self.N_vels_in_y-1:
                        du_dy = (self.V[k,i-1,j,0] - self.V[k,i,j,0])/self.vec_spacing_y
                    else:
                        du_dy = 0.5*(self.V[k,i-1,j,0] - self.V[k,i+1,j,0])/self.vec_spacing_y

                    # Calculate vorticity
                    self.zeta[k,i,j] = du_dy - dv_dx

    
    def filter_shifts(self, e_thresh, e0):
        """Applies median filtering to the shift array(s).

        Parameters
        ----------
        e_thresh : float
            Normalized threshold for filtering the data.

        e0 : float
            Normalizer (to avoid division by zero).

        Returns
        -------
        ndarray
            Filtered velocity array.
        """

        # Initialize new array
        filtered_shifts = np.zeros_like(self.shifts[0])

        # Loop
        for k in range(self.N):
            for i in range(self.N_vels_in_y):
                for j in range(self.N_vels_in_x):

                    # Get neighbors
                    i_min = max(0, i-1)
                    i_max = min(self.Ny, i+2)
                    j_min = max(0, j-1)
                    j_max = min(self.Nx, j+2)

                    # Get statistics
                    y_shift_med = np.median(self.shifts[k,i_min:i_max,j_min:j_max,0].flatten()).item()
                    x_shift_med = np.median(self.shifts[k,i_min:i_max,j_min:j_max,1].flatten()).item()
                    u_std = np.std(self.shifts[k,i_min:i_max,j_min:j_max,0].flatten(), ddof=1).item()
                    v_std = np.std(self.shifts[k,i_min:i_max,j_min:j_max,1].flatten(), ddof=1).item()

                    # Check
                    if abs(self.shifts[k,i,j,0]-y_shift_med)/(u_std*e0) > e_thresh or abs(self.shifts[k,i,j,1]-x_shift_med)/(v_std*e0) > e_thresh:
                        filtered_shifts[i,j,:] = [y_shift_med, x_shift_med]
                    else:
                        filtered_shifts[i,j,:] = self.shifts[k,i,j,:]

            # Replace
            self.shifts[k] = filtered_shifts

    
    def write_to_csv(self, output_file_root_name):
        """Writes the data to a series of csv files.
        
        Parameters
        ----------
        output_file_root_name : str
            Root name for the output files.
        """

        # Loop through data
        for i in range(self.N):

            # Get filename
            filename = "{0}{1}.csv".format(output_file_root_name, str(i).zfill(len(str(self.N))))

            # Open file
            with open(filename, 'w') as output_handle:
                
                # Get image interpolator
                interpolator1 = interp.interp2d(self.y_pix, self.x_pix, self.data[i].T)
                interpolator2 = interp.interp2d(self.y_pix, self.x_pix, self.data[i+1].T)

                # Header
                print("x,y,z,u,v,zeta,raw_data_1,raw_data_2", file=output_handle)

                # Loop through points
                for j in range(self.N_vels_in_y):
                    for k in range(self.N_vels_in_x):

                        # Get interpolated image
                        pix1 = float(interpolator1(self.y_vec[j], self.x_vec[k]).item())
                        pix2 = float(interpolator2(self.y_vec[j], self.x_vec[k]).item())

                        # Write out to file
                        print("{0},{1},{2},{3},{4},{5},{6},{7}".format(self.x_vec[k], self.y_vec[j], 0.0, self.V[i,j,k,0], self.V[i,j,k,1], self.zeta[i,j,k], pix1, pix2), file=output_handle)