import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from PIL import Image


def get_array_from_file(filename):
    # Reads in an image file and returns an array containing the same data

    image = Image.open(filename)
    image_array = np.array(image, dtype=np.float64)
    return image_array


def get_double_image_from_file(filename):
    # Reads in an image file where two images are concatenated on top of each other

    double_array = get_array_from_file(filename)
    N_rows = double_array.shape[0]
    first_image = double_array[:N_rows//2,:]
    second_image = double_array[N_rows//2:,:]

    return first_image, second_image


def display_image_array(image_array):
    # Displays the given image array

    plt.figure()
    plt.imshow(image_array)
    plt.show()


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