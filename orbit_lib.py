import math
import simutils as su

import numpy as np

from utils import polar2coord

import plotter as pl

mu = 398600.4418 # Standard gravitational parameter [km**3/s**-2]
R_E = 6378.1363  # Radius of earth [km]
w_E = 7.292115e-5 # Angular speed of earth [rad/s]

###################################
# Assignment 2 | Helper functions #
###################################

# Conversion between anomalies
def mean_anomaly_from_eccentric_anomaly(E: float, e: float) -> float:
    """
    Converts eccentric anomaly into mean anomaly.
    :param E: Eccentric anomaly [radians]
    :param e: Eccentricity
    :return: Mean anomaly [radians]
    """
    return E - e * math.sin(E)
def mean_anomaly_from_true_anomaly(theta: float, e: float) -> float:
    """
    Converts true anomaly into mean anomaly.
    :param theta: True anomaly [radians]
    :param e: Eccentricity
    :return: Mean anomaly [radians]
    """
    return mean_anomaly_from_eccentric_anomaly(eccentric_anomaly_from_true_anomaly(theta, e), e)
def true_anomaly_from_eccentric_anomaly(E: float, e: float) -> float:
    """
    Converts eccentric anomaly into true anomaly.
    :param E: Eccentric anomaly [radians]
    :param e: Eccentricity
    :return: True anomaly [radians]
    """
    return 2 * math.atan(
        math.sqrt((1+e)/(1-e)) * math.tan(E/2)
    )
def eccentric_anomaly_from_true_anomaly(theta: float, e: float) -> float:
    """
    Converts true anomaly into eccentric anomaly.
    :param theta: True anomaly [radians]
    :param e: Eccentricity
    :return: Eccentric anomaly [radians]
    """
    return 2 * math.atan(
        math.sqrt((1-e)/(1+e)) * math.tan(theta/2)
    )
def eccentric_anomaly_from_mean_anomaly(Me: float, e: float, delta: float=1e-10, N: int=50) -> float:
    """
    Converts mean anomaly into eccentric anomaly using newtons method.
    :param Me: Mean anomaly [radians]
    :param e: Eccentricity
    :param delta: Tolerance for Newton iteration (default: 1e-10)
    :param N: Maximum number of iterations (default: 50)
    :return: Eccentric anomaly [radians]
    """
    E = Me + (e/2 if Me < math.pi else -e/2) # Initial guess

    while True:
        # Newtons method
        dE = (E - e*math.sin(E) - Me) / (1 - e * math.cos(E))
        E -= dE

        # Check if tolerance is exceeded
        if abs(dE) < delta:
            return E

        # Check if max iterations is exceeded
        N -= 1
        if N <= 0:
            print("Unable to calculate eccentric anomaly in given iterations!!!")
            return E

# Orbital period
def orbital_period_from_semi_major_axis(a: float, u: float=mu) -> float:
    """
    Calculates orbital period using semi-major axis.
    :param a: Semi-major axis [km]
    :param u: Standard gravitational parameter (default: Earth's μ) [km**3/s**2]
    :return: Orbital period [s]
    """
    return 2 * math.pi * math.sqrt(a**3 / u)
def orbital_period_from_revs_per_day(x: float) -> float:
    """
    Calculates orbital period from orbital revolution per day.
    :param x: Revolutions per day
    :return: Orbital period [s]
    """
    return 24 * 3600 / x

# Guess he wants h, e, theta, omega, i, w
def orbit_params_from_tle_params(e, x, me, omega, i, w):
    pass
def tle_params_from_orbit_params():
    pass

# Matrix functions
# Note: rotations are intrinsic (body-fixed) axes
def _matrix_r1(alpha: float) -> np.ndarray:
    """
    Returns the 3x3 rotation matrix for rotation about the first axis.
    :param alpha: Rotation angle [radians]
    :return: 3x3 rotation matrix
    """
    return np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)]
    ])
def _matrix_r2(beta: float) -> np.ndarray:
    """
    Returns the 3x3 rotation matrix for rotation about the second axis.
    :param beta: Rotation angle [radians]
    :return: 3x3 rotation matrix
    """
    return np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])
def _matrix_r3(gamma: float) -> np.ndarray:
    """
    Returns the 3x3 rotation matrix for rotation about the third axis.
    :param gamma: Rotation angle [radians]
    :return: 3x3 rotation matrix
    """
    return np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]
    ])

def rotation_matrix_from_classical_euler_sequence(omega: float, i: float, w: float) -> np.ndarray:
    """
    Returns the 3x3 rotation matrix for a classical Euler sequence.

    Rotations are applied in the followingorder:
    - Omega [RAAN] (around z-axis)
    - i     [Inclination] (around x-axis)
    - w     [Argument of perihelion] (around z-axis)

    :param omega: Right Ascension of the Ascending Node (RAAN) [radians]
    :param i: Inclination [radians]
    :param w: Argument of perihelion [radians]
    :return: 3x3 rotation matrix
    """
    return _matrix_r3(omega) @ _matrix_r1(i) @ _matrix_r3(w)
def rotation_matrix_from_roll_pitch_yaw_sequence(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Returns the 3x3 rotation matrix for a roll-pitch-yaw (RPY) sequence.

    Rotations are applied in the following order:
    - Roll  (rotation about x-axis)
    - Pitch (rotation about y-axis)
    - Yaw   (rotation about z-axis)

    :param roll: Roll angle [radians]
    :param pitch: Pitch angle [radians]
    :param yaw: Yaw angle [radians]
    :return: 3x3 rotation matrix
    """
    return _matrix_r3(yaw) @ _matrix_r2(pitch) @ _matrix_r1(roll)

# Quaternion functions
# Note: rotations are intrinsic (body-fixed) axes
def _quaternion_q1(alpha: float) -> su.Quaternion:
    """
    Return the quaternion representing a rotation about the first axis of a rotation sequence.
    :param alpha: Rotation angle [radians]
    :return: Quaternion representing the rotation
    """
    return su.Quaternion([math.cos(alpha / 2), math.sin(alpha / 2), 0, 0])
def _quaternion_q2(beta: float) -> su.Quaternion:
    """
    Return the quaternion representing a rotation about the second axis of a rotation sequence.
    :param beta: Rotation angle [radians]
    :return: Quaternion representing the rotation
    """
    return su.Quaternion([math.cos(beta / 2), 0, math.sin(beta / 2), 0])
def _quaternion_q3(gamma: float) -> su.Quaternion:
    """
    Return the quaternion representing a rotation about the third axis of a rotation sequence.
    :param gamma: Rotation angle [radians]
    :return: Quaternion representing the rotation
    """
    return su.Quaternion([math.cos(gamma / 2), 0, 0, math.sin(gamma / 2)])

def quaternion_from_classical_euler_sequence(omega: float, i: float, w: float) -> su.Quaternion:
    """
    Return the quaternion for a classical Euler sequence.

    Rotations are applied in the following order:
    - Omega [RAAN] (around z-axis)
    - i     [Inclination] (around x-axis)
    - w     [Argument of perihelion] (around z-axis)

    :param omega: Right Ascension of the Ascending Node (RAAN) [radians]
    :param i: Inclination [radians]
    :param w: Argument of perihelion [radians]
    :return: Quaternion representing the rotation
    """
    return _quaternion_q3(omega) @ _quaternion_q1(i) @ _quaternion_q3(w)
def quaternion_from_roll_pitch_yaw_sequence(roll: float, pitch: float, yaw: float) -> su.Quaternion:
    """
    Return the quaternion for a roll-pitch-yaw (RPY) sequence.

    Rotations are applied in the following order:
    - Roll  (rotation about x-axis)
    - Pitch (rotation about y-axis)
    - Yaw   (rotation about z-axis)

    :param roll: Roll angle [radians]
    :param pitch: Pitch angle [radians]
    :param yaw: Yaw angle [radians]
    :return: Quaternion representing the rotation
    """
    return _quaternion_q3(yaw) @ _quaternion_q2(pitch) @ _quaternion_q1(roll)

# Degree & radians functions
def deg2rad(deg: float) -> float:
    """
    Converts the given angle from degrees to radians.
    :param deg: Angle [degrees]
    :return: Angle [radians]
    """
    return deg * math.pi / 180
def rad2deg(rad: float) -> float:
    """
    Converts the given angle from radians to degrees.
    :param rad: Angle [radians]
    :return: Angle [degrees]
    """
    return rad * 180 / math.pi
def angle_wrap_radians(rad: float) -> float:
    """
    Wraps the given angle in radians to the range [0, 2pi].
    :param rad: Angle [radians]
    :return: Equivalent angle in range [0, 2pi]
    """
    return rad % (2 * math.pi)
def angle_wrap_degrees(deg: float) -> float:
    """
    Wraps the given angle in degrees to the range [0, 360].
    :param deg: Angle [degrees]
    :return: Equivalent angle in range [0, 360]
    """
    return deg % 360


###################################
# Assignment 2 | Algorithms       #
###################################

# Algorithm 1
def sidereal_angle(JD: float) -> float:
    """
    Calculates the Greenwich sidereal angle of the Earth from a Julian date.

    :param JD: Julian Date (days since J2000.0, can include fractional day)
    :return: Greenwich sidereal angle [radians]
    """
    T0 = (int(JD) - 2451545) / 36525
    theta_G0 = deg2rad(
        100.4606184 + 36000.77005361 * T0 + 0.00038793 * T0**2 - 2.6e-8 * T0**3
    )

    # Add 0.5 to JD because Julian days start at noon
    theta_G  = theta_G0 + w_E * (24 * 3600 * (JD + 0.5) % 1.0)

    return angle_wrap_radians(theta_G)

# Algorithm 2
def state_from_orbit_params(h: float, e: float, theta: float, omega: float, i: float, w: float, u: float=mu) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the satellite's position and velocity vectors in Earth-Centered Inertial (ECI) frame
    from classical orbital elements.

    :param h: Specific angular momentum [km**2/s]
    :param e: Eccentricity
    :param theta: True anomaly [radians]
    :param omega: Right Ascension of Ascending Node (RAAN) [radians]
    :param i: Inclination [radians]
    :param w: Argument of perihelion [radians]
    :param u: Standard gravitational acceleration (default: Earth's μ) [m/s**2]
    :return: Tuple of two np.ndarray elements:
             - Position vector in ECI frame [km]
             - Velocity vector in ECI frame [km/s]
    """
    r = h ** 2 / u / (1 + e * math.cos(theta)) # Get distance of satellite to planet center
    rp = polar2coord(r, theta) # Convert to XY
    vp = (u/h) * np.array([-np.sin(theta), e + np.cos(theta), 0])

    R = rotation_matrix_from_classical_euler_sequence(omega, i, w)
    return R @ rp, R @ vp

# Algorithm 3
# All arguments from row 2 of tle (Note: given as string!)
# [0]Inclination | [1]RAAN | [2]Eccentricity | [3]Argument of perigee | [4]Mean anomaly | [5](Revs per day | Revs | Checksum)
def state_from_tle_params(args):
    # Get satellite position and velocity in ECI frame from TLE parameters

    # NN.NNNNNNNN | NNNNN | N [Revs per day | Revs | Checksum]
    T = orbital_period_from_revs_per_day(float(args[5][:11]))

    a = (math.sqrt(mu)*T/(2*math.pi)) ** (2/3) # Semi-major axis

    # 0.NNNNNNN [Eccentricity]
    e = float(f"0.{args[2]}")

    h = math.sqrt(a * mu * (1 - e ** 2))

    # float [Mean anomaly]
    Me = deg2rad(float(args[4]))
    # Convert mean anomaly to true anomaly
    E = eccentric_anomaly_from_mean_anomaly(Me, e)
    theta = true_anomaly_from_eccentric_anomaly(E, e)

    # TODO: Notes
    i     = deg2rad(float(args[0]))
    omega = deg2rad(float(args[1]))
    w     = deg2rad(float(args[3]))

    # Get satellite position and velocity
    ri, vi = state_from_orbit_params(h, e, theta, omega, i, w)

    return ri, vi


# Algorithm 4
def orbit_params_from_state(ri, vi):
    r  = np.linalg.norm(ri)
    v  = np.linalg.norm(vi)
    vr = np.dot(ri ,vi) / r

    # Angular momentum
    hi = np.cross(ri, vi)
    h = np.linalg.norm(hi)

    # Inclination
    i = math.acos(hi[2] / h)

    k = np.array([0, 0, 1])
    Ni = np.cross(k, hi)
    N = np.linalg.norm(Ni)

    # RAAN
    omega = math.acos(Ni[0]/N)
    if Ni[1] < 0:
        omega = 2 * math.pi - omega

    # Eccentricity
    ei = ((v ** 2 - mu / r) * ri - vi * vr * r) / mu
    e = np.linalg.norm(ei)

    # Argument of perigee
    w = math.acos(np.dot(Ni, ei) / (N * e))
    if ei[2] < 0:
        w = 2 * math.pi - w

    # True anomaly
    theta = math.acos(np.dot(ei, ri) / (e * r))
    if vr < 0:
        theta = 2 * math.pi - theta

    return h, e, theta, omega, i, w

# Algorithm 5
def orbit_propagation(ri, vi):
    h, e, theta, omega, i, w = orbit_params_from_state(ri, vi)
    print(e)

    # TODO: Unsure about step 2 in algorithm 5
    # Get mean anomaly
    Me = mean_anomaly_from_eccentric_anomaly(eccentric_anomaly_from_true_anomaly(theta, e), e)

    # Mean motion
    a = h ** 2 / (mu * (1 - e ** 2))
    T = orbital_period_from_semi_major_axis(a)
    n = 2 * math.pi / T
    print(a, T / 60)
    # Test
    pos_plot = np.concatenate(([0], ri))  # Initialize the plot data

    # Some for loop??? Propagation loop
    dt = 1
    for t in range(0, int(T), dt):
        Me = angle_wrap_radians(Me + n * dt)
        # Is the eccentric anomaly wanted full iteration or just 1?
        E = eccentric_anomaly_from_mean_anomaly(Me, e)
        # Safe to overwrite theta due to its values already being calculated
        theta = true_anomaly_from_eccentric_anomaly(E, e)

        # Get the new ri, vi
        ri, vi = state_from_orbit_params(h, e, theta, omega, i, w)
        pos_plot = np.vstack((pos_plot, np.concatenate(([t], ri))))

    file = su.log_pos("assignment2_position", pos_plot)
    pos_plot = None  # Clear the data after its saved
    pl.line_plot(file)


# Algorithm 6
# Epoch [int|int|float] (Note: given as string!)
# NN|NNN.NNNNNNNN [Year | Day in year . Fraction]
def epoch_to_julian_date(epoch):
    year = int(epoch[:2]) + 2000
    day  = float(epoch[2:]) # Includes fraction
    leap = 1 if year % 4 == 0 and day <= 60 else 0 # Uses day 60 due to UTC being included

    return 2451544.5 + year * 365 + year // 4 + day - leap