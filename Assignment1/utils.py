import math
import numpy as np

### Extra Utils used by me ###

# Convert between polar coordinates to
# cartesian coordinates
def polar2coord(r, theta, out=None):
    if out is None:
        out = np.empty(3)

    out[0] = r * math.cos(theta)
    out[1] = r * math.sin(theta)
    out[2] = 0
    return out