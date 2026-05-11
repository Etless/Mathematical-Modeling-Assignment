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
        :param w0: Initial angular velocity [rad/s]
        :param J: 3x3 Inertia matrix
        """
        if J.shape != (3, 3):
            raise ValueError("J must be a 3x3 matrix")

        self.ri = r0.copy() # Initial position [array]
        self.vi = v0.copy() # Initial velocity [array]
        self.force = np.zeros(3) # Force acting on body [array]

        self.m = m               # Mass of body

        self.q = q0.normalized() # Initial quaternion [array] (normalized creates new class)
        self.w = w0.copy()       # Initial angular velocity [array]
        self.tau = np.zeros(3)   # Torque acting on body [array]

        # Maps torque to angular acceleration
        self.J = J.copy() # Inertia matrix [array]
        self.J_inv = np.linalg.inv(self.J) # Inverse (precalculated)

        # Build state vector
        self.x = np.concatenate([self.ri, self.vi, self.q, self.w])

    def update(self, t: float, dt: float, force: np.ndarray,tau: np.ndarray) -> None:
        """
        Update the state of the rigid body.
        :param t: Time [s]
        :param dt: Time step [s]
        :param force: Force vector [N]
        :param tau: Torque vector acting on body [N*m]
        """
        self.force = force # Update force
        self.tau   = tau   # Update torque

        self.x = su.step_RK4(dt, t, self.x, self.f) # Perform step (dont include ae)

        self.ri   = self.x[  : 3] # Update position
        self.vi   = self.x[ 3: 6] # Update velocity
        self.q.q  = self.x[ 6:10] # Update values in Quaternion class
        self.w    = self.x[10:  ] # Update angular velocity

        self.q.normalize()  # Normalize values in class
        self.x[6:10] = self.q.q # Update state vector quaternion

    def get_state(self) -> tuple[np.ndarray, np.ndarray, su.Quaternion, np.ndarray]:
        """
        Returns the state of the rigid body.
        :return: Tuple containing state elements:
                 - Position vector in ECI frame [km]
                 - Velocity vector in ECI frame [km/s]
                 - Rotation quaternion
                 - Angular velocity in body frame [rad/s]
        """
        # Copy values so no unwanted changes is passed back
        return self.ri.copy(), self.vi.copy(), self.q.normalized(), self.w.copy()

    # noinspection PyUnusedLocal
    def f(self, t: float, x: np.ndarray, _: None=None)  -> np.ndarray:
        """
        Compute the time derivative of the state vector.

        The state vector x contains both the position, velocity, quaternion and angular velocity,
        formatted as: [rx, ry, rz, vx, vy, vz, q0, q1, q2, q3, w0, w1, w2].

        :param t: Time [s]
        :param x: State vector containing position, velocity, quaternion and angular velocity
        :param _: Unused
        :return: Time derivative of the state vector
        """
        # Extract values from state vector
        ri_ = x[:3] # Unused
        vi_ = x[3:6]
        q_  = x[6:10]
        w_  = x[10:]

        # Get acceleration
        ai = self.force / self.m # Force returned from class (self)
        dq = 0.5 * (su.Quaternion(q_).normalized() @ su.Quaternion(w_)) # Normalize quaternion
        dw = self.J_inv @ (self.tau - np.cross(w_, self.J @ w_))

        return np.concatenate([vi_, ai, dq, dw])

class Satellite:
    def __init__(self, q_ib: su.Quaternion, w_bib: np.ndarray, J: np.ndarray, ri: np.ndarray=np.zeros(3), vi: np.ndarray=np.zeros(3), m: float=1, orbit=None, substeps: int=0) -> None:
        """
        Satellite class used to represent a satellite.

        If an orbit object is provided, then the satellite position and velocity
        follows the orbit model.

        :param q_ib: Initial attitude quaternion
        :param w_bib: Initial angular velocity [rad/s]
        :param J: 3x3 Inertia matrix
        :param ri: Initial position vector (default: zero) [km]
        :param vi: Initial velocity vector (default: zero) [km/s]
        :param m: Satellite mass (default: 1) [kg]
        :param orbit: Orbit propagation object (default: None)
        :param substeps: Number of ADCS substeps per update (default: 0)
        """
        # Set orbit class to be used for the orbit
        self.orbit = orbit

        # Get state from orbit
        if self.orbit is not None:
            ri, vi, _, _ = self.orbit.get_state()

        # Create rigid body of satellite
        self.body = RigidBody(ri, vi, m, q_ib, w_bib, J)

        # Substep used for ADCS integration
        self.N = substeps + 1 # Add one to make sure it runs at least once
        self.ADCS = ADCS_PD(1e-5, 2e-4, J)

    def update(self, t: float, dt: float) -> None:
        """
        Update the state of the satellite.
        :param t: Time [s]
        :param dt: Time step [s]
        """
        if self.orbit is not None:
            self.update_with_orbit(t, dt)
        else:
            self.update_with_dynamics(t, dt)

    def get_state(self) -> tuple[np.ndarray, np.ndarray, su.Quaternion, np.ndarray]:
        """
        Returns the state of the satellite.
        :return: Tuple containing state elements:
                 - Position vector in ECI frame [km]
                 - Velocity vector in ECI frame [km/s]
                 - Rotation quaternion
                 - Angular velocity in body frame [rad/s]
        """
        return self.body.get_state() # Values are a copy

    def get_orbit_frame(self) -> tuple[su.Quaternion, np.ndarray, np.ndarray]:
        """
        Returns orbital frame attitude and kinematics of satellite.
        :return: Tuple containing orbital frame parameters:
                 - Orbit to inertial quaternion
                 - Angular velocity of orbit frame [rad/s]
                 - Angular acceleration of orbit frame [rad/s**2]
        """
        if self.orbit is not None:
            return self.orbit.get_orbit_frame()
        else:
            ri, vi, _, _ = self.body.get_state()
            return ol.orbit_frame_from_state(ri, vi)

    ### update methods ###
    def update_with_orbit(self, t: float, dt: float) -> None:
        """
        Update the state of the satellite using orbit propagation and attitude control.
        :param t: Time [s]
        :param dt: Time step [s]
        """
        # Get initial and propagated states
        r0, v0 = self.orbit.get_state()
        self.orbit.propagate(dt)
        r1, v1 = self.orbit.get_state()

        # Calculate the sub delta time
        dt_sub = dt / self.N

        # Perform substeps
        for n in range(0, self.N):
            # Propagate position and velocity linearly for each substep
            ri = r0 + n / self.N * (r1 - r0)
            vi = v0 + n / self.N * (v1 - v0)

            # Get current states
            _, _, q_ib, w_bib = self.get_state()
            q_io, w_iio, _ = ol.orbit_frame_from_state(ri, vi)

            # Get ADCS torqe based on states
            self.ADCS.update(t, q_ib, w_bib, q_io, w_iio, np.zeros(3))
            tau_u = self.ADCS.get_control()

            # Update satellites body with forces
            self.body.update(t, dt_sub, np.zeros(3), tau_u)
            t += dt_sub

        # Manualy update rigid body position and velocity from orbit
        self.body.ri, self.body.vi = self.orbit.get_state()

    def update_with_dynamics(self, t: float, dt: float) -> None:
        """
        Update the state of the satellite based on dynamics propagation.
        :param t: Time [s]
        :param dt: Time step [s]
        """
        # Calculate the sub delta time
        dt_sub = dt / self.N

        # Perform substeps
        for n in range(0, self.N):
            # Get current states
            ri, vi, q_ib, w_bib = self.get_state()
            q_io, w_iio, dw_iio = self.get_orbit_frame()

            # Get ADCS torqe based on states
            self.ADCS.update(t, q_ib, w_bib, q_io, w_iio, dw_iio)
            tau_u = self.ADCS.get_control()

            # Calculate orbit force (two body problem)
            f = self.body.m * (-ol.mu / np.linalg.norm(ri) ** 3 * ri)

            # Update satellites body with forces
            self.body.update(t, dt_sub, f, tau_u)
            t += dt_sub


###################################
# Assignment 5 | Algorithms       #
###################################

# noinspection PyPep8Naming
class ADCS_PD:
    def __init__(self, k1, k2, J):
        self.k1 = k1
        self.k2 = k2
        self.J = J.copy()
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
        #print("help")
        #print("me")
        #print("plz")

    def get_control(self):
        return self.tau

