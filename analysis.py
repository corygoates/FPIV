import numpy as np
import scipy.signal as sig

from image_handling import display_image_array


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
    corr = np.where(corr < 0.0, 0.0001, corr)

    # Find maximum (we're not going to check the edges here)
    max_corr = 0.0
    i_max = -1
    for  i in range(1, corr.shape[0]-1):
        for j in range(1, corr.shape[1]-1):
            if corr[i,j] > max_corr:
                i_max = i
                j_max = j
                max_corr = corr[i,j]

    # Check if we've found a maximum
    if i_max == -1:
        return [0, 0]
    else:

        # Get array around peak
        peak_array = corr[i_max-1:i_max+2,j_max-1:j_max+2]

        # Get subpixel peak location
        peak = center_of_fit_parabola(peak_array)
        if any(np.isnan(peak)):
            display_image_array(corr)

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

    # y-direction
    lnR_m = np.log(corr_peak[1,0])
    lnR_0 = np.log(corr_peak[1,1])
    lnR_p = np.log(corr_peak[1,2])
    e_y = 0.5*(lnR_m - lnR_p) / (lnR_m - 2.0*lnR_0 + lnR_p)

    # x-direction
    lnR_m = np.log(corr_peak[0,1])
    lnR_p = np.log(corr_peak[2,1])
    e_x = 0.5*(lnR_m - lnR_p) / (lnR_m - 2.0*lnR_0 + lnR_p)

    return np.nan_to_num([e_y, e_x])