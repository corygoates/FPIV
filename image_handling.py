import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def get_array_from_file(filename):
    # Reads in an image file and returns an array containing the same data

    image = Image.open(filename)
    image_array = np.array(image, dtype=np.float64)
    return image_array.T


def display_image_array(image_array):
    # Displays the given image array

    plt.figure()
    plt.imshow(image_array)
    plt.show()


def load_time_series(directory, file_root_name, extension, N_files):
    """Loads a set of time series data into a single array. Assumes the files are stored using 1-based indexing.
    
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
    """

    # Get shape of image arrays
    filename = get_sample_filename(directory,file_root_name, extension, 1, N_files)
    image_array = get_array_from_file(filename)
    image_shape = image_array.shape

    # Initialize array
    time_series_array = np.zeros((N_files, *image_shape))

    # Now get the data
    for i in range(N_files):
        filename = get_sample_filename(directory, file_root_name, extension, i+1, N_files)
        time_series_array[i] = get_array_from_file(filename)

    return time_series_array


def get_sample_filename(directory, file_root_name, extension, i, N_files):
    """Loads a set of time series data into a single array.
    
    Parameters
    ----------
    directory : str
        Folder the data are stored in.

    file_root_name : str
        First part of filename which does not change between the data.

    extension : str
        File extension.

    i : int
        File number.

    N_files : int
        Total number of files in the sample.
    """

    filename = "{0}{1}{2}".format(file_root_name, str(i).zfill(len(str(N_files))), extension)
    return os.path.join(directory, filename)


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