import numpy as np
import scipy.signal as sig


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


def apply_median_filter(V_array, e_thresh, e0):
    """Applies median filtering to the given velocity array.

    Parameters
    ----------
    V_array : ndarray
        Velocity array on which to apply median filtering.

    e_thresh : float
        Normalized threshold for filtering the data.

    e0 : float
        Normalizer (to avoid division by zero).

    Returns
    -------
    ndarray
        Filtered velocity array.
    """

    # Get data limits
    Ni, Nj, _ = V_array.shape

    # Initialize new array
    V_filtered = np.zeros_like(V_array)

    # Loop
    for i in range(Ni):
        for j in range(Nj):

            # Get neighbors
            i_min = max(0, i-1)
            i_max = min(Ni, i+2)
            j_min = max(0, j-1)
            j_max = min(Ni, j+2)

            # Get statistics
            u_med = np.median(V_array[i_min:i_max,j_min:j_max,0].flatten()).item()
            v_med = np.median(V_array[i_min:i_max,j_min:j_max,1].flatten()).item()
            u_std = np.std(V_array[i_min:i_max,j_min:j_max,0].flatten(), ddof=1).item()
            v_std = np.std(V_array[i_min:i_max,j_min:j_max,1].flatten(), ddof=1).item()

            # Check
            if abs(V_array[i,j,0]-u_med)/(u_std*e0) > e_thresh or abs(V_array[i,j,1]-v_med)/(v_std*e0) > e_thresh:
                V_filtered[i,j,:] = [u_med, v_med]
            else:
                V_filtered[i,j,:] = V_array[i,j,:]

    return V_filtered