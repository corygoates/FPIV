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
        return [i_max-array1.shape[0]//2, j_max-array1.shape[1]//2]
