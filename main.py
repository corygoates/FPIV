import os
import matplotlib.pyplot as plt
from piv import TimeSeriesPIVAnalysis


if __name__=="__main__":

    target_dir = "data/synthetic_jet/"
    my_piv = TimeSeriesPIVAnalysis(target_dir, "C001H001S00010000", ".tif", 10, 0.01)
    my_piv.process(32, vector_spacing=8)
    my_piv.write_velocities_to_csv("results/synthetic_jet/velocity")