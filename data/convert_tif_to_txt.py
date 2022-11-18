import os
import sys
import cv2
import numpy as np
from PIL import Image

def convert_file(filename):
    # Reads in data and converts to txt format

    # Read in data
    image = Image.open(filename)

    # Convert to numpy array
    im_array = np.array(image, dtype=np.float64)

    # Write out
    out_filename = filename.replace(".tif", ".txt")
    np.savetxt(out_filename, im_array, header="{0} {1}".format(im_array.shape[0], im_array.shape[1]))


if __name__=="__main__":

    # Loop through files in specified directory
    target_dir = "data/synthetic_jet/"
    for filename in os.listdir(target_dir):
        convert_file(os.path.join(target_dir, filename))