import numpy as np

from analysis import get_correlation_peak

class BasePIVAnalysis:
    """Base class for performing PIV analysis."""


    def __init__(self):
        pass


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
            self.y_vec[i] = -(self.window_size//2 + i*self.vector_spacing)*self.dy


    def convert_shifts_to_velocities(self, shift_array):
        """Takes the given correlation shifts and converts them to velocities.
        
        Parameters
        ----------
        shift_array : ndarray
            Array of correlation shifts.
        """

        # Initialize storage
        V = np.zeros_like(shift_array)

        # Calculate velocities
        V[:,:,0] = -shift_array[:,:,1]*self.dx/self.dt
        V[:,:,1] =  shift_array[:,:,0]*self.dx/self.dt

        return V


    def calculate_shifts(self, frame1, frame2, N_passes=1):
        """Calculates the shift field between two image frames.
        
        Parameters
        ----------
        frame1 : ndarray
            First image array.
        
        frame2 : ndarray
            Second image array.

        N_passes : int, optional
            Number of passes using window offsetting. Defaults to 1.
        """

        # Initialize storage
        shifts = np.zeros((self.N_vels_in_y, self.N_vels_in_x, 2))


    def get_frame_vorticity(V_array):
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
        zeta_array = np.zeros((self.N_vels_in_y, self.N_vels_in_x))

        # Loop
        for i in range(Ny):
            for j in range(Nx):

                # Get dv/dx
                if j == 0:
                    dv_dx = (V_array[i,j+1,1] - V_array[i,j,1])/self.vec_spacing_x
                elif j == Nx-1:
                    dv_dx = (V_array[i,j,1] - V_array[i,j-1,1])/self.vec_spacing_x
                else:
                    dv_dx = 0.5*(V_array[i,j+1,1] - V_array[i,j-1,1])/self.vec_spacing_x

                # Get du/dy
                if i == 0:
                    du_dy = (V_array[i,j,0] - V_array[i+1,j,0])/self.vec_spacing_y
                elif i == Ny-1:
                    du_dy = (V_array[i-1,j,0] - V_array[i,j,0])/self.vec_spacing_y
                else:
                    du_dy = 0.5*(V_array[i-1,j,0] - V_array[i+1,j,0])/self.vec_spacing_y

                # Calculate vorticity
                zeta_array[i,j] = du_dy - dv_dx

        return zeta_array