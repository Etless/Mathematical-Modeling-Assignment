import numpy as np
import orbit_lib as ol
import math

satellite_distance = ol.R_E + 400 # From earth center [km]
T = 2 * math.pi * math.sqrt(satellite_distance ** 3 / ol.mu) # Orbital period [s]

class RigidBody:
    def __init__(self):
        pass

    def update(self, t, dt, et):
        pass


