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

    my_piv = BaseballPIVAnalysis("data/baseball_data/baseballs1.tif", 0.001, pixel_threshold=10)
    my_piv.process(2.0, 0.1, 32, vector_spacing=8, N_passes=9)
    my_piv.write_to_csv("results/baseballs/baseballs1_")

    render_csv_with_paraview("results/baseballs/baseballs1_0.csv", image_name='baseball.png')