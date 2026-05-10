import numpy as np
import orbit_lib as ol
import simutils as su
import math

satellite_distance = ol.R_E + 400 # From earth center [km]
T = 2 * math.pi * math.sqrt(satellite_distance ** 3 / ol.mu) # Orbital period [s]

###################################
# Assignment 5 | Algorithms       #
###################################

class RigidBody:
    def __init__(self, r0: np.ndarray, v0: np.ndarray, m: float, q0: su.Quaternion, w0: np.ndarray, J: np.ndarray) -> None:
        """
        Rigid body class used for attitude simulation.
        :param r0: Initial position vector [km]
        :param v0: Initial velocity vector [km/s]
        :param m: The satellites mass [kg]
        :param q0: Initial quaternion (normalized)
        :param q0: Initial quaternion (normalized)
        :param w0: Initial angular velocity [rad/s]
        :param J: 3x3 Inertia matrix
        """
        if J.shape != (3, 3):
            raise ValueError("J must be a 3x3 matrix")

        self.ri = r0             # Initial position [array]
        self.vi = v0             # Initial velocity [array]
        self.force = np.zeros(3) # Force acting on body [array]

        self.m = m               # Mass of body

        self.q = q0.normalized() # Initial quaternion [array] (normalized creates new class)
        self.w = w0              # Initial angular velocity [array]
        self.tau = np.zeros(3)   # Torque acting on body [array]

        # Maps torque to angular acceleration
        self.J = J  # Inertia matrix [array]
        self.J_inv = np.linalg.inv(self.J) # Inverse (precalculated)

    def update(self, t: float, dt: float, force: np.ndarray,tau: np.ndarray) -> None:
        """
        Update the state of the rigid body.
        :param t: Time [s]
        :param dt: Time step [s]
        :param force: Force vector [N*m]
        :param tau: Torque vector acting on body [N*m]
        """

        #self.force = force # Update force
        #self.tau   = tau   # Update torque

        self.force = np.array(force, dtype=np.float64, copy=True)
        self.tau = np.array(tau, dtype=np.float64, copy=True)

        assert np.all(np.isfinite(self.q))
        assert np.all(np.isfinite(self.w))
        assert np.all(np.isfinite(self.tau))

        # State vector containing both kinematics and dynamics
        x = np.concatenate([self.ri, self.vi, self.q, self.w])

        x = su.step_RK4(dt, t, x, self.f) # Perform step (dont include ae)

        self.ri   = x[  : 3] # Update position
        self.vi   = x[ 3: 6] # Update velocity
        self.q.q = x[ 6:10] # Update values in Quaternion class
        self.w    = x[10:  ] # Update angular velocity

        self.q.normalize()  # Normalize values in class

    def get_state(self):
        return self.ri, self.vi, self.q, self.w

    # noinspection PyUnusedLocal
    def f(self, t: float, x: np.ndarray, _: None=None, u: float=ol.mu)  -> np.ndarray:
        """
        Compute the time derivative of the state vector.

        The state vector x contains both the position, velocity, quaternion and angular velocity,
        formatted as: [rx, ry, rz, vx, vy, vz, q0, q1, q2, q3, w0, w1, w2].

        :param t: Time [s]
        :param x: State vector containing position, velocity, quaternion and angular velocity
        :param _: Unused
        :param u: Standard gravitational parameter (default: Earth's μ) [km**3/s**2]
        :return: Time derivative of the state vector
        """

        # Temp is used to differentiate it from the class self counterparts
        vi_temp = x[3:6]
        q_temp  = x[6:10]
        w_temp  = x[10:]

        ai = self.force / self.m # Get acceleration vector

        dq = 0.5 * (su.Quaternion(q_temp) @ su.Quaternion([0,*w_temp])) # Normalize quaternion
        dw = self.J_inv @ (self.tau - np.cross(w_temp, self.J @ w_temp))

        return np.concatenate([vi_temp, ai, dq, dw])


class Satellite:
    def __init__(self, q_ib, w_bib, J, ri=np.zeros(3), vi=np.zeros(3), m=1, orbit=None, substeps=0) -> None:
        self.orbit = orbit

        if self.orbit is not None:
            ri, vi = self.orbit.get_state()

        self.body = RigidBody(ri, vi, m, q_ib, w_bib, J)
        self.N = substeps + 1
        self.ADCS = ADCS_PD(1e-5, 2e-4, J)

    def update(self, t: float, dt: float) -> None:
        """
        Update the state of the satellite.
        :param t: Time [s]
        :param dt: Time step [s]
        """
        if self.orbit:
            self.update_with_orbit(t, dt)
        else:
            self.update_with_dynamics(t, dt)

    def get_state(self):
        return self.body.get_state()

    def get_orbit_frame(self):
        if self.orbit:
            return self.orbit.get_orbit_frame()
        else:
            ri, vi, _, _ = self.body.get_state()
            return ol.orbit_frame_from_state(ri, vi)

    def update_with_orbit(self, t: float, dt: float) -> None:
        r0, v0 = self.orbit.get_state()
        self.orbit.propagate(dt)
        r1, v1 = self.orbit.get_state()
        t_sub = dt / self.N
        for n in range(0, self.N):
            ri = r0 + n / self.N * (r1 - r0)
            vi = v0 + n / self.N * (v1 - v0)
            _, _, q_ib, w_bib = self.get_state()
            q_io, w_iio, _ = ol.orbit_frame_from_state(ri, vi)
            self.ADCS.update(t, q_ib, w_bib, q_io, w_iio, np.zeros(3))
            tau_u = self.ADCS.get_control()
            self.body.update(t, t_sub, np.zeros(3), tau_u)
            t += t_sub
        self.body.ri, self.body.vi = self.orbit.get_state()

    def update_with_dynamics(self, t, dt):
        t_sub = dt / self.N
        for n in range(0, self.N):
            ri, vi, q_ib, w_bib = self.get_state()
            q_io, w_iio, dw_iio = self.get_orbit_frame()
            self.ADCS.update(t, q_ib, w_bib, q_io, w_iio, dw_iio)
            f = -ol.mu / np.linalg.norm(ri) ** 3 * ri
            tau_u = self.ADCS.get_control()
            self.body.update(t, t_sub, f, tau_u)
            t += t_sub


###################################
# Assignment 5 | Algorithms       #
###################################

class ADCS_PD:
    def __init__(self, k1, k2, J):
        self.k1 = k1
        self.k2 = k2
        self.J = J
        self.tau = np.zeros(3)

    def update(self, t, q_ib, w_bib, q_io, w_iio, dw_iio):
        # Quaternion error (desired -> body)
        q_db = q_io.conjugated() @ q_ib
        if q_db[0] < 0:  # Shortest way/direction to rotate
            q_db *= -1

        # Angular velocity error (desired -> body)
        w_db = w_bib - q_db.conjugated().rotate(w_iio)

        # Simple PD controller for torque
        self.tau = -self.k1 * q_db[1:] - self.k2 * w_db

        # Important commands. If function does
        # not print 3 times then the ADCS breaks.
        # No clue why
        print("help")
        print("me")
        print("plz")

    def get_control(self):
        return self.tau

