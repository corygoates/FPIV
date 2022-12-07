import os

import matplotlib.pyplot as plt

from time_series_piv import TimeSeriesPIVAnalysis
from baseball_piv import BaseballPIVAnalysis
from paraview_visualization import render_csv_with_paraview


if __name__=="__main__":

    #target_dir = "data/synthetic_jet/"
    #my_piv = TimeSeriesPIVAnalysis(target_dir, "C001H001S00010000", ".tif", 10, 0.01)
    #my_piv.process(2.0, 0.5, 32, vector_spacing=8)
    #my_piv.write_to_csv("results/synthetic_jet/velocity")

    my_piv = BaseballPIVAnalysis("data/baseball_data/baseballs2.tif", dt=10.0e-6, pixel_threshold=10)
    my_piv.process(1.0, 0.1, 24, vector_spacing=3, N_passes=1)#, max_shift_in_pixels=10)
    my_piv.write_to_csv("results/baseballs/baseballs2_")
    my_piv.create_velocity_histogram()
    my_piv.plot_slice_in_y(0, -1)
    my_piv.plot_quiver(0, background='velocity')

    #render_csv_with_paraview("results/baseballs/baseballs1_0.csv", image_name='baseball.png', arrow_scale_factor=0.005)