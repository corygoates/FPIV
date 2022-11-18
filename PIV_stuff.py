import numpy as np
import scipy.signal as sig


def compute_cross_correlations(time_series_data, window_size):
    """Computes the cross-correlations of the given time-series data.
    
    Parameters
    ----------
    time_series_data : ndarray
        Array of time-series data.

    window_size : int
        Interrogation window size.

    Returns
    -------
    ndarray
        Computed cross-correlation arrays.
    """

    return


def subtract_background(time_series_data, show_average=False):
    """Performs average background subtraction on the given time-series data array.
    
    Parameters
    ----------
    time_series_data : ndarray
        Array of time-series data.

    show_average : bool, optional
        Whether to display the computed average image. Defaults to False.

    Returns
    -------
    ndarray
        Data with average background subtracted.
    """

    # Compute average
    average_image = np.average(time_series_data, axis=0)

    # Display
    if show_average:
        display_image_array(average_image)

    # Subtract
    subtracted_data = time_series_data - average_image

    return subtracted_data