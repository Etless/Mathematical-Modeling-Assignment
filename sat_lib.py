import numpy as np
import orbit_lib as ol
import simutils as su
import math

###################################
# Assignment 7 | Sensors ++       #
###################################

# Class serves as the baseline for the other sensor classes
class Sensor:
    def update(self, t: float, dt: float, ri: np.ndarray, vi: np.ndarray, q_ib: su.Quaternion, w_bib: np.ndarray) -> None:
        """
        Update sensor values.
        :param t: Time [s]
        :param dt: Time step [s]
        :param ri: Position vector in ECI frame [km]
        :param vi: Velocity vector in ECI frame [km/s]
        :param q_ib: Quaternion rotating body frame to ECI frame
        :param w_bib: Angular velocity in body frame [rad/s]
        """
        pass
    def output(self, body_frame: bool = False) -> np.ndarray:
        """
        Returns the simulated sensor measurement.

        Output can be choosen to either be in sensor frame or
        body frame by configuring the body_frame parameter.

        :param body_frame: If measurement should be in body frame (default: False)
        :return: Sensor measurement
        """
        pass

class Gyro(Sensor):
    def __init__(self, q_bs: su.Quaternion, p_b: np.ndarray=np.zeros(3), z0: np.ndarray=np.zeros(3), sigma_g2: float=0.0, sigma_bg2: float=0.0, beta_w0: np.ndarray=np.zeros(3)) -> None:
        """
        Gyro sensor model with additive white noise and time-varying bias drift.

        :param q_bs: Quaternion orientation of sensor frame relative to body frame
        :param p_b: Position vector of sensor relative to body frame origin (default: zero vector) [km]
        :param z0: Initial gyro angular measurement (default: zero vector) [rad/s]
        :param sigma_g2: Variance of gyro noise (default: 0)
        :param sigma_bg2: Variance of the bias drift noise (default: 0)
        :param beta_w0: Initial time-varying offset (default: zero vector) [rad/s]
        """
        # Variance
        self.sigma_g = math.sqrt(sigma_g2)
        self.sigma_bg = math.sqrt(sigma_bg2)

        # Sensor position and orientation
        self.q_bs = q_bs
        self.p_b = p_b

        self.beta_w = beta_w0

        # Output
        self.z = z0

    def update(self, t: float, dt: float, ri: np.ndarray, vi: np.ndarray, q_ib: su.Quaternion, w_bib: np.ndarray) -> None:
        """
        Update sensor values.
        :param t: Time [s]
        :param dt: Time step [s]
        :param ri: Position vector in ECI frame [km]
        :param vi: Velocity vector in ECI frame [km/s]
        :param q_ib: Quaternion rotating body frame to ECI frame
        :param w_bib: Angular velocity in body frame [rad/s]
        """
        # Split into three parts (true, offset & noise)
        eta_w = np.random.normal(0, self.sigma_g, 3)

        # Time-varying error/offset also referred to as the bias (dependent on delta time)
        self.beta_w += np.random.normal(0, self.sigma_bg, 3) * dt

        # Rotate angular velocity from body frame to sensor frame
        w_t = self.q_bs.conjugated().rotate(w_bib)

        # Simulated gyro measurement
        self.z = w_t + self.beta_w + eta_w

    def output(self, body_frame: bool=False) -> np.ndarray:
        """
        Returns the simulated gyro measurement.

        Output can be choosen to either be in sensor frame or
        body frame by configuring the body_frame parameter.

        :param body_frame: If measurement should be in body frame (default: False)
        :return: Gyro angular velocity measurement [rad/s]
        """
        return self.q_bs.rotate(self.z) if body_frame else self.z
class Magnetometer(Sensor):
    def __init__(self, q_bs: su.Quaternion, p_b: np.ndarray=np.zeros(3), z0: np.ndarray=np.zeros(3), sigma_B2: float=0.0, JD: float=0.0) -> None:
        """
        Magnetometer sensor model with additive white noise.

        :param q_bs: Quaternion orientation of sensor frame relative to body frame
        :param p_b: Position vector of sensor relative to body frame origin (default: zero vector) [km]
        :param z0: Initial measured magnetometer field vector (default: zero vector) [T]
        :param sigma_B2: Variance of white noise (default: 0)
        :param JD: Julian Date (days since J2000.0, can include fractional day)
        """
        # Variance
        self.sigma_B = math.sqrt(sigma_B2)

        # Sensor position and orientation
        self.q_bs = q_bs
        self.p_b = p_b

        # Output
        self.z = z0

        self.JD = JD

    def update(self, t: float, dt: float, ri: np.ndarray, vi: np.ndarray, q_ib: su.Quaternion, w_bib: np.ndarray) -> None:
        """
        Update sensor values.
        :param t: Time [s]
        :param dt: Time step [s]
        :param ri: Position vector in ECI frame [km]
        :param vi: Velocity vector in ECI frame [km/s]
        :param q_ib: Quaternion rotating body frame to ECI frame
        :param w_bib: Angular velocity in body frame [rad/s]
        """
        # Noise
        eta_B = np.random.normal(0, self.sigma_B, 3)

        # Get the magnetix flux density
        B_iE = ol.magnetic_field_dipol(ri, self.JD + t / (24.0 * 3600.0))

        # Internal frame -> body frame -> sensor frame
        q_is = q_ib @ self.q_bs

        # Rotate magnetix flux density to sensor frame and add noise
        self.z = q_is.conjugated().rotate(B_iE) + eta_B

    def output(self, body_frame: bool=False) -> np.ndarray:
        """
        Returns the simulated magnetometer measurement.

        Output can be choosen to either be in sensor frame or
        body frame by configuring the body_frame parameter.

        :param body_frame: If measurement should be in body frame (default: False)
        :return: Measured magnetometer field vector [T]
        """
        return self.q_bs.rotate(self.z) if body_frame else self.z
class FineSunSensor(Sensor):
    def __init__(self, q_bs: su.Quaternion, p_b: np.ndarray=np.zeros(3), z0: np.ndarray=np.zeros(3), sigma_s2: float=0.0, alpha: float=math.pi, JD: float=0.0):
        """
        Fine sun sensor model with additive white noise.

        :param q_bs: Quaternion orientation of sensor frame relative to body frame
        :param p_b: Position vector of sensor relative to body frame origin (default: zero vector) [km]
        :param z0: Initial measured sun direction vector (default: zero vector)
        :param sigma_s2: Variance of white noise (default: 0)
        :param alpha: FOV (default: pi) [radians]
        :param JD: Julian Date (days since J2000.0, can include fractional day)
        """
        # Variance
        self.sigma_s = math.sqrt(sigma_s2)

        # Sensor position and orientation
        self.p_b = p_b
        self.q_bs = q_bs

        # Output
        self.z = z0

        # FOV (threshold)
        self.alpha = alpha

        self.JD = JD

    def update(self, t: float, dt: float, ri: np.ndarray, vi: np.ndarray, q_ib: su.Quaternion, w_bib: np.ndarray) -> None:
        """
        Update sensor values.
        :param t: Time [s]
        :param dt: Time step [s]
        :param ri: Position vector in ECI frame [km]
        :param vi: Velocity vector in ECI frame [km/s]
        :param q_ib: Quaternion rotating body frame to ECI frame
        :param w_bib: Angular velocity in body frame [rad/s]
        """
        # Get sun vector and transform from ECI to body frame
        si = ol.sun_vector(self.JD + t / (24.0 * 3600.0)) # Best would be to calculate this outside class and pass it to all sun sensors
        s_b_sensor = q_ib.conjugated().rotate(si - ri)

        # Normilize value
        s_bt = s_b_sensor / np.linalg.norm(s_b_sensor)

        # Transform value into sensor frame
        s_st = self.q_bs.conjugated().rotate(s_bt)

        x, y ,z = s_st
        # Check FOV
        if z > 0 and math.atan(math.sqrt(x ** 2 + y ** 2) / 2) < self.alpha / 2:
            eta_s = np.random.normal(0, self.sigma_s, 3)
            self.z = s_st + eta_s
        else:
            self.z = np.zeros(3)

    def output(self, body_frame: bool=False) -> np.ndarray:
        """
        Returns the simulated sun measurement.

        Output can be choosen to either be in sensor frame or
        body frame by configuring the body_frame parameter.

        :param body_frame: If measurement should be in body frame (default: False)
        :return: Measured Sun direction vector
        """
        return self.q_bs.rotate(self.z) if body_frame else self.z

# Estimation classes
class AttitudeEstimator:
    def estimate_attitude(self, M_B: list[np.ndarray], M_A: list[np.ndarray]) -> su.Quaternion:
        pass

class TRIAD(AttitudeEstimator):
    def estimate_attitude(self, M_B: list[np.ndarray], M_A: list[np.ndarray]) -> su.Quaternion:
        """
        Attitude estimation using TRIAD's algorithm.

        M_B is the body frame vectors and M_A is the reference frame vectors.

        :param M_B: List of vectors in frame B
        :param M_A: List of vectors in frame A
        :return: Estimated coordinate transformation from A to B
        """
        # Only take the two first vectors from M_A and M_B (Filters out excess sun sensors)
        # Refrence frame
        UA = M_A[0]
        VA = M_A[1]

        # Body frame
        UB = M_B[0]
        VB = M_B[1]

        ta1 = su.unit(UA)
        ta2 = su.unit(np.cross(ta1, VA))
        ta3 = np.cross(ta2, ta1)

        Ta = np.array([ta1, ta2, ta3])

        tb1 = su.unit(UB)
        tb2 = su.unit(np.cross(tb1, VB))
        tb3 = np.cross(tb2, tb1)

        Tb = np.array([tb1, tb2, tb3])

        # Convert rotation matrix to queterion
        return su.dcm_to_quaternion(Ta.T @ Tb).normalized()
class Davenport(AttitudeEstimator):
    def estimate_attitude(self, M_B: list[np.ndarray], M_A: list[np.ndarray], weights: list[float] | None=None) -> su.Quaternion:
        """
        Attitude estimation using Davenport's algorithm.

        M_B is the body frame vectors and M_A is the reference frame vectors.

        :param M_B: List of vectors in frame B
        :param M_A: List of vectors in frame A
        :param weights: Weights for each sensor vector (default: Uniform)
        :return: Estimated coordinate transformation from A to B
        """
        N = len(M_B)

        # If weights are not given then make them uniform
        if weights is None:
            weights = np.ones(N) / N

        B = np.zeros((3,3))
        z = np.zeros(3)

        # Compute B, z
        for i in range(N):
            B += weights[i] * np.outer(M_B[i], M_A[i])
            z += weights[i] * np.cross(M_B[i], M_A[i])

        tr_B = np.trace(B)

        # Davenport K-matrix
        K = np.zeros((4,4))
        K[0, 0] = tr_B
        K[0, 1:] = z
        K[1:, 0] = z
        K[1:, 1:] = B + B.T - tr_B * np.eye(3)

        # Get the eigenvectors from K-matrix
        _, evecs = np.linalg.eigh(K)

        # Construct rotation quatarion from eigenvectors
        return su.Quaternion(evecs[:, -1]).normalized()


###################################
# Assignment 7 | Satellite ++     #
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
    def __init__(self, q_ib: su.Quaternion, w_bib: np.ndarray, J: np.ndarray, sensors: list[Sensor], JD: float, ri: np.ndarray | None=None, vi: np.ndarray | None=None, m: float=1, orbit=None, substeps: int=0, estimator: AttitudeEstimator=Davenport()) -> None:
        """
        Satellite class used to represent a satellite.

        If an orbit object is provided, then the satellite position and velocity
        follows the orbit model.

        :param q_ib: Initial attitude quaternion
        :param w_bib: Initial angular velocity [rad/s]
        :param J: 3x3 Inertia matrix
        :param sensors: All attitude sensors connected to the satellite (default: None)
        :param JD: Julian Date (days since J2000.0, can include fractional day)
        :param ri: Initial position vector (default: zero) [km]
        :param vi: Initial velocity vector (default: zero) [km/s]
        :param m: Satellite mass (default: 1) [kg]
        :param orbit: Orbit propagation object (default: None)
        :param substeps: Number of ADCS substeps per update (default: 0)
        :param estimator: Attitude estimator (default: Davenport)
        """
        # Handle None
        if ri is None:
            ri = np.zeros(3)
        if vi is None:
            vi = np.zeros(3)

        # Set orbit class to be used for the orbit
        self.orbit = orbit

        # Get state from orbit
        if self.orbit is not None:
            ri, vi = self.orbit.get_state()

        # Create rigid body of satellite
        self.body = RigidBody(ri, vi, m, q_ib, w_bib, J)

        # Substep used for ADCS integration
        self.N = substeps + 1 # Add one to make sure it runs at least once
        self.ADCS = ADCS_PD(1e-5, 2e-4, J, JD=JD, estimator=estimator, sensors=sensors)

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
            self.ADCS.update(t, dt_sub, ri, vi, q_ib, w_bib, q_io, w_iio, np.zeros(3))
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
            self.ADCS.update(t, dt_sub, ri, vi, q_ib, w_bib, q_io, w_iio, dw_iio)
            tau_u = self.ADCS.get_control()

            # Calculate orbit force (two body problem)
            f = self.body.m * (-ol.mu / np.linalg.norm(ri) ** 3 * ri)

            # Update satellites body with forces
            self.body.update(t, dt_sub, f, tau_u)
            t += dt_sub


###################################
# Assignment 7 | ADCS             #
###################################

# noinspection PyPep8Naming
class ADCS_PD:
    def __init__(self, k1: float, k2: float, J: np.ndarray, estimator: AttitudeEstimator, JD: float, sensors: list[Sensor]) -> None:
        """
        PD-based ADCS controller.

        :param k1: Proportional gain
        :param k2: Derivative gain
        :param J: Inertia matrix (deprecated)
        :param estimator: Attitude estimator (TRIAD, Davenport, etc.)
        :param JD: Julian Date (days since J2000.0, can include fractional day)
        :param sensors: List of sensor objects
        """
        # Coeficence varaibles
        self.k1 = k1
        self.k2 = k2

        self.J = J.copy() # Not used anymore

        # Torque
        self.tau = np.zeros(3)

        # Estimator class used for attitude estimation
        self.estimator = estimator
        self.JD = JD


        # Add sensor from list to its respective class
        self.sun_sensors = []
        self.mag_sensor  = None
        self.gyro_sensor = None
        self.sensors = sensors # Used for an easy update loop

        for s in sensors:
            if isinstance(s, FineSunSensor):
                self.sun_sensors.append(s)
            elif isinstance(s, Magnetometer):
                self.mag_sensor = s
            elif isinstance(s, Gyro):
                self.gyro_sensor = s

    def update(self, t: float, dt: float, ri: np.ndarray, vi: np.ndarray, q_ib: su.Quaternion, w_bib: np.ndarray, q_io: su.Quaternion, w_iio: np.ndarray, dw_iio: np.ndarray=np.zeros(3)) -> None:
        """
        Performs one ADCS control update iteration.

        :param t: Time [s]
        :param dt: Time step [s]
        :param ri: Position vector in ECI frame [km]
        :param vi: Velocity vector in ECI frame [km/s]
        :param q_ib: Quaternion rotating body frame to ECI frame
        :param w_bib: Angular velocity in body frame [rad/s]
        :param q_io: Desired attitude quaternion
        :param w_iio: Desired angular velocity [rad/s]
        :param dw_iio: Desired angular acceleration (default: zero) [rad/s**2]
        """
        # Get refrence vectors
        Bi = ol.magnetic_field_dipol(ri, self.JD + t / (24.0 * 3600.0))
        Si = ol.sun_vector(self.JD + t / (24.0 * 3600.0))

        # Update sensors
        for sensor in self.sensors:
            sensor.update(t, dt, ri, vi, q_ib, w_bib)

        # Create empty list for estimator
        M_B = []
        M_A = []

        # Magnometer
        M_B.append(su.unit(self.mag_sensor.output(body_frame=True)))
        M_A.append(su.unit(Bi))

        # Fine sun sensors
        for sun in self.sun_sensors:

            # Only care about non-zero values
            measurement = sun.output(body_frame=True)
            if np.linalg.norm(measurement) == 0:
                continue

            M_B.append(su.unit(measurement))
            M_A.append(su.unit(Si))

        # Estimate attitude
        q_ib_estimate = self.estimator.estimate_attitude(M_B, M_A)

        # Quaternion error (desired -> body)
        q_db = q_io.conjugated() @ q_ib_estimate
        if q_db[0] < 0:  # Shortest way/direction to rotate
            q_db *= -1

        # Angular velocity error (desired -> body)
        w_db = self.gyro_sensor.output(body_frame=True) - q_db.conjugated().rotate(w_iio)

        # Simple PD controller for torque
        self.tau = -self.k1 * q_db[1:] - self.k2 * w_db

    def get_control(self) -> np.ndarray:
        """
        Returns the current control torque computed by the ADCS controller.
        :return: Control torque vector in body frame [N*m]
        """
        return self.tau