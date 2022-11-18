import os
import matplotlib.pyplot as plt
from image_handling import load_time_series, display_image_array
from PIV_stuff import subtract_background, compute_cross_correlations


if __name__=="__main__":

    # Loop through files in specified directory
    target_dir = "data/synthetic_jet/"
    data = load_time_series(target_dir, "C001H001S0001000", ".tif", 240)

    # Perform background subtraction
    data_wo_background = subtract_background(data, show_average=False)