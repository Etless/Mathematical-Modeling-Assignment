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


def ground_tracking(file_path: str):
    """
    Plots ground tracking data from file.

    The file is expected to contain the following data:
        1. Image of ground
        2. Longitude and Latitude of satellite

    :param file_path: File path to ground tracking data
    """

    ## Temp load image directly
    ground = image.imread("3DModels/earth_8k.jpg")

    plt.plot(200, 350, marker='v', color="white")

    plt.imshow(ground)
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
