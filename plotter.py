import math
import sys
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.image as image

def line_plot(file_path):
    data = np.loadtxt(file_path)
    t = data[:,0]
    _, ax = plt.subplots()
    for col in data[:,1:].T:
        ax.plot(t,col)
    plt.show()


def ground_tracking(file_path: str, img_path):
    """
    Plots ground tracking data from file.

    The file is expected to contain the following data:
        1. Image of ground
        2. Longitude and Latitude of satellite

    :param file_path: File path to ground tracking data
    """

    # Load data
    data = np.loadtxt(file_path)
    _, lons, lats = data.T

    # Load ground track image
    ground = image.imread(img_path)

    # Find jumps larger than pi
    jumps = np.abs(np.diff(lons)) > np.pi
    split_indices = np.where(jumps)[0] + 1

    # Create segments from each split segment
    lon_segments = np.split(lons, split_indices)
    lat_segments = np.split(lats, split_indices)

    _, ax = plt.subplots()
    ax.imshow(ground, extent=(-180.0, 180.0, -90.0, 90.0), zorder=0)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)

    for lo, la in zip(lon_segments, lat_segments):
        ax.plot(lo/np.pi*180, la/np.pi*180, color='r', zorder=1)
    plt.show()

def main(argv):
    if len(argv) == 3:
        plot_type = argv[1]
        file_path = argv[2]
        if plot_type == 'lineplot':
            line_plot(file_path)
        else:
            print("Plot type not supported yet.")
    else:
        print("Wrong number of arguments. Expected 2 (plot_type, file_path) got {}".format(len(argv)-1))

if __name__ == "__main__":
    main(sys.argv)
