import os
import matplotlib.pyplot as plt
from image_handling import load_time_series, display_image_array, subtract_background


if __name__=="__main__":

    # Loop through files in specified directory
    target_dir = "data/synthetic_jet/"
    data = load_time_series(target_dir, "C001H001S0001000", ".tif", 240)

    # Perform background subtraction
    data_wo_background = subtract_background(data, show_average=True)
    display_image_array(data[23])
    display_image_array(data_wo_background[23])