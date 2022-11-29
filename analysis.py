import numpy as np
import scipy.signal as sig

from image_handling import display_image_array

# Set up matrices for correlation peak fitting
x = np.array([[-1.0, 0.0, 1.0],
              [-1.0, 0.0, 1.0],
              [-1.0, 0.0, 1.0]])

y = np.array([[-1.0, -1.0, -1.0],
              [ 0.0,  0.0,  0.0],
              [ 1.0,  1.0,  1.0]])

Z = np.zeros((9,6))
Z[:,0] = 1.0
Z[:,1] = x.flatten()
Z[:,2] = y.flatten()
Z[:,3] = 0.5*Z[:,1]**2
Z[:,4] = Z[:,1]*Z[:,2]
Z[:,5] = 0.5*Z[:,2]**2


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
    i_max = -1
    for  i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            if corr[i,j] > max_corr:
                i_max = i
                j_max = j
                max_corr = corr[i,j]

    # Check if we've found a maximum
    if i_max == -1:
        return [0, 0]
    else:

        # Get array around peak
        peak_array = np.zeros((3,3))
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if not (i_max+i < 0 or i_max+i >= corr.shape[0] or j_max+j < 0 or j_max+j >= corr.shape[1]):
                    peak_array[i+1,j+1] = corr[i_max+i,j_max+j]

        # Get subpixel peak location
        peak = center_of_fit_parabola(peak_array)

        # Reject erroneous peak fits (Shouldn't be further from the original peak than 1 pixel in any direction)
        if np.linalg.norm(peak) > 1.414:
            peak = [0.0, 0.0]
        return [i_max-array1.shape[0]//2+peak[1], j_max-array1.shape[1]//2+peak[0]]


def center_of_fit_parabola(corr_peak):
    """Returns the center of a parabola fit to the correlation peak.

    Parameters
    ----------
    corr_peak : ndarray
        3x3 square of pixels bracketing the correlation peak.

    Returns
    -------
    ndarray
        Gives the coordinates of the fit peak. Returns [0,0] if the fit peak is exactly aligned with the original peak.
    """

    # Get parabolic coefficients
    coefs, _, _, _ = np.linalg.lstsq(Z, corr_peak.flatten(), rcond=None)

    # Get peak location
    A = np.zeros((2,2))
    b = np.zeros(2)
    A[0,0] = coefs[3]
    A[0,1] = coefs[4]
    A[1,0] = coefs[4]
    A[1,1] = coefs[5]
    b[0] = -coefs[1]
    b[1] = -coefs[2]
    peak = np.linalg.solve(A, b)
    return peak