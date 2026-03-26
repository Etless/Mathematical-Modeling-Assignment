import math
import simutils as su

import numpy as np

from utils import polar2coord

import plotter as pl

mu = 398600.4418 # Standard gravitational parameter [km**3/s**-2]
R_E = 6378.1363  # Radius of earth [km]
w_E = 7.292115e-5 # Angular speed of earth [rad/s]

# Assignment 2 ### Helper functions
# Did not specify eccentricity being needed
def mean_anomaly_from_eccentric_anomaly(E, e):
    return E - e * math.sin(E)
def eccentric_anomaly_from_true_anomaly(theta, e):
    return 2 * math.atan(
        math.sqrt((1-e)/(1+e)) * math.tan(theta/2)
    )
def true_anomaly_from_eccentric_anomaly(E, e):
    return 2 * math.atan(
        math.sqrt((1+e)/(1-e)) * math.tan(E/2)
    )
def eccentric_anomaly_from_mean_anomaly(Me, e, delta=1e-10, N=10000):
    E = Me # TODO: Initial guess

    while True:
        dE = (E - e*math.sin(E) - Me) / (1 - e * math.cos(E))
        E -= dE

        # Check if tolerance is within threshold
        if abs(dE) < delta:
            return E

        # Check if max iterations has been passed
        N -= 1
        if N <= 0:
            print("Unable to calculate eccentric anomaly in given iterations!!!")
            return E

def orbital_period_from_semi_major_axis(a, u=mu):
    return 2 * math.pi * math.sqrt(a**3 / u)
def orbital_period_from_revs_per_day(x):
    return 24 * 3600 / x

# Guess he wants h, e, theta, omega, i, w
def orbit_params_from_tle_params(e, x, me, omega, i, w):
    pass
def tle_params_from_orbit_params():
    pass

# Presume its given in rads and not quaternions
# (feels redundant to write essentially the same code 4 times in a row)
def r1_matrix(alpha):
    return np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)]
    ])
def r2_matrix(beta):
    return np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ])
def r3_matrix(gamma):
    return np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1]
    ])

def rotation_matrix_from_classical_euler_sequence(omega, i, w):
    return r3_matrix(omega) @ r1_matrix(i) @ r3_matrix(w)

def quaternion_from_classical_euler_sequence(omega, i, w):
    q_1 = su.Quaternion([math.cos(w / 2), math.sin(w / 2), 0, 0])
    q_2 = su.Quaternion([math.cos(i / 2), 0, math.sin(i / 2), 0])
    q_3 = su.Quaternion([math.cos(omega / 2), 0, 0, math.sin(omega / 2)])
    return q_3 @ q_2 @ q_1

def rotation_matrix_from_roll_pitch_yaw_sequence(omega, i, w):
    R_1 = np.array([
        [1, 0, 0],
        [0, np.cos(w), -np.sin(w)],
        [0, np.sin(w), np.cos(w)]
    ])
    R_2 = np.array([
        [np.cos(i), 0, np.sin(i)],
        [0, 1, 0],
        [-np.sin(i), 0, np.cos(i)]
    ])
    R_3 = np.array([
        [np.cos(omega), -np.sin(omega), 0],
        [np.sin(omega), np.cos(omega), 0],
        [0, 0, 1]
    ])
    return R_3 @ R_2 @ R_1
def quaternion_from_roll_pitch_yaw_sequence(omega, i, w):
    q_1 = su.Quaternion([math.cos(w / 2), math.sin(w / 2), 0, 0])
    q_2 = su.Quaternion([math.cos(i / 2), 0, math.sin(i / 2), 0])
    q_3 = su.Quaternion([math.cos(omega / 2), 0, 0, math.sin(omega / 2)])
    return q_3 @ q_2 @ q_1

def angle_wrap_radians(rad):
    return rad % (2 * math.pi)
def angle_wrap_degrees(deg):
    return deg % 360

# Assignment 2 ### Algorithms functions
# Algorithm 1
# TODO: Add comment / notes
def sidereal_angle(JD):
    T0 = (int(JD) - 2451545) / 36525

    theta_G0 = deg2rad(100.4606184 + 36000.77005361 * T0 + 0.00038793 * T0 ** 2 - 2.6e-8 * T0 ** 3)
    theta_G  = theta_G0 + w_E * (24 * 3600 * (JD + 0.5) % 1.0)

    return angle_wrap_radians(theta_G)

# Algorithm 2
# TODO: Add comment / notes
def state_from_orbit_params(h, e, theta, omega, i, w):
    r = h ** 2 / mu / (1 + e * math.cos(theta))
    rp = polar2coord(r, theta)
    vp = (mu/h) * np.array([-np.sin(theta), e + np.cos(theta), 0])

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
    print(e)
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


# DTOR
def deg2rad(deg) -> float:
    return deg * math.pi / 180

# RTOD
def rad2deg(rad) -> float:
    return rad * 180 / math.pi