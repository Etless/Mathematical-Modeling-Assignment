import numpy as np
import orbit_lib as ol
import simutils as su
import math

satellite_distance = ol.R_E + 400 # From earth center [km]
T = 2 * math.pi * math.sqrt(satellite_distance ** 3 / ol.mu) # Orbital period [s]

class RigidBody:
    def __init__(self, ri: np.ndarray, vi: np.ndarray, m: float, J: np.ndarray, w0: np.ndarray, q0: su.Quaternion):
        self.m  = m # Mass [float]
        self.r  = np.linalg.norm(ri) # Radius [float]
        self.vi = vi # Velocity [array]


        # Equivalent of position [q], velocity [w] and force [tau] # TODO: Remove
        self.q = q0.normalized() # Initial quaternion [array]
        self.w = w0              # Initial angular velocity [array]
        self.tau = np.zeros(3)   # Torque acting on body [array]

        # Maps torque to angular acceleration
        self.J = J  # Inertia matrix [array]

    def update(self, t, dt, fu):


    def f(self, t, x, ae: np.ndarray=None):
        p = self.r * x[:3] / np.linalg.norm(x[:3]) # Normalize
        dot_p = x[3:]

        w = np.cross(np.cross(p, self.fs), p)

        norm_w = np.linalg.norm(w)
        if norm_w > 1e-6: # Accommodates floating-point error
            w /= norm_w

        fc = np.dot(self.fs, w) * w
        dot_v = fc / self.m - self.k * dot_p
        return np.concat((dot_p, dot_v))


