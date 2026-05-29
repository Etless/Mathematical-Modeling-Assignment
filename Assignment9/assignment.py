import math

import numpy as np
import simutils as su
import sat_lib as sl
import orbit_lib as ol
import Assignment9.simulator as sim

import plotter as pl

# Make tqdm not required to run program
try:
    import tqdm
except ImportError:
    tqdm = None


# Functions is only used for debug purposes and not strictly needed for simulation
if tqdm is not None:
    pbar: tqdm.tqdm | None=None
def progress_bar(total: float):
    global pbar
    if tqdm is not None:
        pbar = tqdm.tqdm(total=total, dynamic_ncols=True)
def progress_bar_update(steps: float=1, info: str | None=None):
    global pbar
    if tqdm is not None and pbar is not None:
        if info is not None:
            pbar.set_description(info, refresh=False)

        pbar.update(steps)
        pbar.refresh()
def progress_bar_close():
    global pbar
    if pbar is not None:
        pbar.close()
        pbar = None


##############################
# Assignment 9 Part 1        #
##############################

def PKepler_from_tle_params(file_path: str, index: int=0, debug: bool=False) -> tuple[ol.OrbitPKepler, float]:
    """
    Returns an orbit PKepler class from a TLE file.
    :param file_path: file path of the TLE file
    :param index: Satellite index of the TLE file (default: 0)
    :param debug: If debug output should be written (default: False)
    :return: OrbitPKepler class
    """
    # Load TLE file and get all elements assigned to it
    epoch, e, rev, Me, omega, i, w, dn, d2n = ol.orbit_params_from_tle_params(file_path, debug=debug, index=index)

    # Orbit params
    n = 2 * math.pi / ol.orbital_period_from_revs_per_day(rev)
    a = ol.semi_major_axis_from_mean_motion(n)

    JD = ol.epoch_to_julian_date(epoch)

    # Return orbit class
    return ol.OrbitPKepler(a, e, Me, omega, i, w, dn, d2n), JD

def print_state(orbit, JD: float, theta_E: float, text: str):
    ri, vi = orbit.get_state()
    a, e, Me, omega, i, w, dn, d2n = orbit.get_params()

    E = ol.eccentric_anomaly_from_mean_anomaly(Me, e)
    theta = ol.true_anomaly_from_eccentric_anomaly(E, e)

    n = 2 * math.pi / ol.orbital_period_from_semi_major_axis(a)

    # Note: Underlying functions use the more standard latitude = phi and longitude = lam
    lam, phi, height = ol.geodetic_from_xyz(ol.ecef_from_eci(ri, theta_E))

    q_io, w_iio, dw_iio = ol.orbit_frame_from_state(ri, vi)

    h, _, _, _, _, _ = ol.orbit_params_from_state(ri, vi)

    print(" === === TLE data and parameters (", text ,") === ===")
    print("Specific Relative Angular Momentum : h      ", h)
    print("True Anomaly                       : θ      ", ol.angle_wrap_radians(theta))
    print("Eccentric Anomaly                  : E      ", ol.angle_wrap_radians(E))
    print("Semi-Major Axis                    : a      ", a)
    print("Mean Motion                        : n      ", n)
    print("Derivative of Mean Motion          : dn     ", dn)
    print("Second Derivative of Mean Motion   : d2n    ", d2n)
    print("Position                           : ri     ", ri)
    print("Velocity                           : vi     ", vi)
    print("Julian Date                        : JD     ", JD)
    print("Sidereal Angle                     : θG0    ", ol.angle_wrap_radians(theta_E))
    print("Orbit frame                        : q_io   ", q_io[:])
    print("Orbit frame Angular Velocity       : w_i_io ", w_iio)
    print("Orbit frame Angular Acceleration   : dw_i_io", dw_iio)
    print("Geodetic Latitude                  : λ′     ", lam)
    print("Geodetic/Geocentric longitude      : ϕ      ", phi)
    print("Altitude                           : h      ", height)


##############################
# Assignment 9 Part 2        #
##############################

def error_in_arcsec(q: su.Quaternion):
    return  2 * 180 * 3600 / math.pi * np.arcsin(np.linalg.norm(q[1:]))

class Part1Task2(sim.BaseScenario):
    def __init__(self, file_path):
        self.ri = None
        self.q = None
        self.orbit1_theta_E = None
        self.orbit2_theta_E = None
        self.orbit3_theta_E = None

        self.orbit2_plot = None
        self.orbit3_plot = None
        self.orbit_difference_plot = None

        self.orbit1_ground_track_plot = None
        self.orbit2_ground_track_plot = None
        self.orbit3_ground_track_plot = None

        # Orbit to look 9 years in the future
        self.orbit1, self.JD1 = PKepler_from_tle_params(file_path, index=0, debug=False) # HST1

        # Orbit of current year
        self.orbit2, self.JD2 = PKepler_from_tle_params(file_path, index=0, debug=False) # HST1
        # Orbit 9 years ago to be propagated to current year
        self.orbit3, self.JD3 = PKepler_from_tle_params(file_path, index=1, debug=False) # HST2

    def init(self, t):

        print("Initial ISO Time:")
        print("Orbit 1:", ol.julian_date_to_iso(self.JD1))
        print("Orbit 2:", ol.julian_date_to_iso(self.JD2))
        print("Orbit 3:", ol.julian_date_to_iso(self.JD3))

        # Calculate delta time between orbit2 (HST1) and orbit3 (HST2)
        dt_23 = (24.0 * 3600.0) * (self.JD2 - self.JD3)

        # Calculate delta time for orbit1 (HST1) to propagate 9 years in the future
        dt_1  = 9.0 * 365.0 * (24.0 * 3600.0)

        substeps = 100000 # Number of substeps to perform the propagation

        # Propagate orbits to target date
        self.orbit1.propagate_timestep(dt_1, substeps)
        self.orbit3.propagate_timestep(dt_23, substeps)

        print("\nUpdated ISO Time:")
        print("Orbit 1:", ol.julian_date_to_iso(self.JD1 + dt_1 / (24.0 * 3600.0))) # Gives dubious year error but is only for the conversion to iso
        print("Orbit 2:", ol.julian_date_to_iso(self.JD2))
        print("Orbit 3:", ol.julian_date_to_iso(self.JD3 + dt_23 / (24.0 * 3600.0)))

        self.ri, vi = self.orbit1.get_state()
        self.q, _, _ = self.orbit1.get_orbit_frame()

        # !!! Not needed just to make simulation look more realistic
        self.orbit1_theta_E = ol.sidereal_angle(self.JD1 + dt_1 / (24.0 * 3600.0)) # Rotation of earth at updated orbit1 (HST1)

        self.orbit2_theta_E = ol.sidereal_angle(self.JD2)                            # Rotation of earth at orbit2 (HST1)
        self.orbit3_theta_E = ol.sidereal_angle(self.JD3 + dt_23 / (24.0 * 3600.0))  # Rotation of earth at updated orbit3 (HST2)

        # Used to plot the difference between orbit2 and orbit3
        ri2, _ = self.orbit2.get_state()
        r2 = np.linalg.norm(ri2)
        ri3, _ = self.orbit3.get_state()
        r3 = np.linalg.norm(ri3)
        self.orbit2_plot = np.concatenate(([t], [*ri2, r2]))
        self.orbit3_plot = np.concatenate(([t], [*ri3, r3]))

        r_diff = r2 - r3
        self.orbit_difference_plot = np.concatenate(([t], [r_diff]))

        # Used for plotting ground track of orbits
        lat1, lon1, _ = ol.geodetic_from_xyz(ol.ecef_from_eci(self.ri, self.orbit1_theta_E))
        lat2, lon2, _ = ol.geodetic_from_xyz(ol.ecef_from_eci(ri2, self.orbit2_theta_E))
        lat3, lon3, _ = ol.geodetic_from_xyz(ol.ecef_from_eci(ri3, self.orbit3_theta_E))
        self.orbit1_ground_track_plot = np.concatenate(([t], [lon1, lat1]))
        self.orbit2_ground_track_plot = np.concatenate(([t], [lon2, lat2]))
        self.orbit3_ground_track_plot = np.concatenate(([t], [lon3, lat3]))

        # Assignment 9 Part 1 Task 1
        print("\n\n")
        print_state(self.orbit2, self.JD2, self.orbit2_theta_E, "Now")
        # Assignment 9 Part 1 Task 2
        print("\n\n")
        print_state(self.orbit1, self.JD1 + dt_1 / (24.0 * 3600.0), self.orbit1_theta_E, "9 Years")


    def update(self, t, dt):

        # Propagate the orbit
        self.orbit1.propagate(dt)
        self.orbit2.propagate(dt)
        self.orbit3.propagate(dt)

        # Get states from orbit1 (It is the only one rendered)
        self.ri, _ = self.orbit1.get_state()
        self.q, _, _ = self.orbit1.get_orbit_frame()

        # Calculate earth's rotation from time step
        self.orbit1_theta_E += dt * ol.w_E

        # Also calculate it for the other orbits
        self.orbit2_theta_E += dt * ol.w_E
        self.orbit3_theta_E += dt * ol.w_E

        # Used to plot the difference between orbit2 and orbit3
        ri2, _ = self.orbit2.get_state()
        r2 = np.linalg.norm(ri2)
        ri3, _ = self.orbit3.get_state()
        r3 = np.linalg.norm(ri3)
        self.orbit2_plot = np.vstack((self.orbit2_plot, np.concatenate(([t], [*ri2, r2]))))
        self.orbit3_plot = np.vstack((self.orbit3_plot, np.concatenate(([t], [*ri3, r3]))))

        r_diff = r2 - r3
        self.orbit_difference_plot = np.vstack((self.orbit_difference_plot, np.concatenate(([t], [r_diff]))))

        # Used for plotting ground track of orbits
        lat1, lon1, _ = ol.geodetic_from_xyz(ol.ecef_from_eci(self.ri, self.orbit1_theta_E))
        lat2, lon2, _ = ol.geodetic_from_xyz(ol.ecef_from_eci(ri2, self.orbit2_theta_E))
        lat3, lon3, _ = ol.geodetic_from_xyz(ol.ecef_from_eci(ri3, self.orbit3_theta_E))
        self.orbit1_ground_track_plot = np.vstack((self.orbit1_ground_track_plot, np.concatenate(([t], [lon1, lat1]))))
        self.orbit2_ground_track_plot = np.vstack((self.orbit2_ground_track_plot, np.concatenate(([t], [lon2, lat2]))))
        self.orbit3_ground_track_plot = np.vstack((self.orbit3_ground_track_plot, np.concatenate(([t], [lon3, lat3]))))

    def get(self):
        temp = ol.polar2xyz(1, self.orbit1_theta_E / 2)  # Normalized XY from q_E
        q_E = su.Quaternion([temp[0], 0, 0, temp[1]])

        return [
            ['satellite', self.ri, self.q],
            ['body_frame', self.ri, self.q],
            ['earth', np.zeros(3), q_E],
            ['ECEF frame', np.zeros(3), q_E],
            ['ECI frame', np.zeros(3), su.Quaternion()]
        ]

    def post_process(self, t, dt):
        file = su.log_pos("assignment9_orbit2", self.orbit2_plot)
        self.orbit2_plot = None  # Clear the data after its saved
        pl.line_plot(file, labels=["X", "Y", "Z", "R"], x_axis="Time [s]", y_axis="Height [km]", titel="Orbit height of HST1", linestyle=["-", "-", "-", "--"])

        file = su.log_pos("assignment9_orbit3", self.orbit3_plot)
        self.orbit3_plot = None  # Clear the data after its saved
        pl.line_plot(file, labels=["X", "Y", "Z", "R"], x_axis="Time [s]", y_axis="Height [km]", titel="Orbit height of HST2", linestyle=["-", "-", "-", "--"])

        file = su.log_pos("assignment9_orbit_difference", self.orbit_difference_plot)
        self.orbit_difference_plot = None  # Clear the data after its saved
        pl.line_plot(file, labels=None, x_axis="Time [s]", y_axis="Difference [km]", titel="Difference in height")

        # Plot ground tracks
        grid_img = "Assignment9/earth_grid.jpg" # Path to image of ground

        file = su.log_pos("assignment9_ground_track_orbit1", self.orbit1_ground_track_plot)
        self.orbit1_ground_track_plot = None  # Clear the data after its saved
        pl.ground_tracking(file, grid_img)

        file = su.log_pos("assignment9_ground_track_orbit2", self.orbit2_ground_track_plot)
        self.orbit2_ground_track_plot = None  # Clear the data after its saved
        pl.ground_tracking(file, grid_img)

        file = su.log_pos("assignment9_ground_track_orbit3", self.orbit3_ground_track_plot)
        self.orbit3_ground_track_plot = None  # Clear the data after its saved
        pl.ground_tracking(file, grid_img)

class Part2Task1(sim.BaseScenario):
    def __init__(self, file_path):
        self.sat1 = None
        self.sat2 = None

        # Are the same for both satellites
        self.theta_E = None
        self.target = None

        self.pointing_error = None
        self.pointing_error1 = None
        self.pointing_error2 = None

        self.orbit1, self.JD1 = PKepler_from_tle_params(file_path, index=0, debug=False)  # HST1 (ADCS_PD)
        self.orbit2, self.JD2 = PKepler_from_tle_params(file_path, index=0, debug=False)  # HST1 (ADCS_SM)

        #self.fine = False

    def init(self, t):
        q_ib = su.Quaternion([1, 0, 0, 0])
        w_bib = np.array([0.3, -0.1, 2]) * 1E-3
        q_id = su.Quaternion([0.5, 0.5, 0.5, 0.5])
        w_did = np.zeros(3)
        dw_did = np.zeros(3)

        self.target = (q_id, w_did, dw_did) # Set target tuple

        # kg * m ** 2
        J = np.array([
            [36046,  -706,  1491],
            [ -706, 86868,   449],
            [ 1491,   449, 93848]
        ])

        # Add Star-sensor & gyro
        sensors1 = [
            sl.StarTracker(su.Quaternion([1, 0, 0, 0]), su.Quaternion([1, 0, 0, 0]), 0, np.zeros(3), 1e-2),  # 1E-2
            sl.Gyro(su.Quaternion([1, 0, 0, 0]), np.zeros(3), np.zeros(3), 1e-6, 0, np.zeros(3))  # 1E-6
        ]

        sensors2 = [
            sl.StarTracker(su.Quaternion([1, 0, 0, 0]), su.Quaternion([1, 0, 0, 0]), 0, np.zeros(3), 1e-2),  # 1E-2
            sl.Gyro(su.Quaternion([1, 0, 0, 0]), np.zeros(3), np.zeros(3), 1e-6, 0, np.zeros(3))  # 1E-6
        ]

        self.theta_E = ol.sidereal_angle(self.JD1)  # Offset to the rotation

        # Create ADCS
        ADCS1 = sl.ADCS_PD(1E-4, 2E-2, J, estimator=sl.Davenport(), JD=self.JD1, sensors=sensors1)
        self.sat1 = sl.Satellite(q_ib, w_bib, J, sensors=sensors1, ADCS=ADCS1, JD=self.JD1,orbit=self.orbit1, substeps=50, estimator=sl.Davenport())

        # Due to sensor noise make targeting coefficient aggressive 2e-2, 2e-5, 2e-5
        ADCS2 = sl.ADCS_SM(2e-2, 2e-5, 2e-5, J, JD=self.JD2, estimator=sl.Davenport(), sensors=sensors2)
        #ADCS = sl.ADCS_SM(1.2e-3, 1.0e-5, 3.0e-6, J, JD=self.JD, estimator=sl.Davenport(), sensors=[*star_sensors, gyro_sensor])
        self.sat2 = sl.Satellite(q_ib, w_bib, J, sensors=sensors2, ADCS=ADCS2, JD=self.JD2, orbit=self.orbit2, substeps=50, estimator=sl.Davenport())

        # Used for plotting Error
        ri1, _, q1, _ = self.sat1.get_state()
        ri2, _, q2, _ = self.sat2.get_state()

        q_oG = su.Quaternion([0, 1, 0, 0]) # Gaussian frame
        arcsec_err1 = error_in_arcsec(q_oG.conjugated() @ self.target[0].conjugated() @ q1)
        arcsec_err2 = error_in_arcsec(q_oG.conjugated() @ self.target[0].conjugated() @ q2)

        self.pointing_error1 = np.concatenate(([t], [arcsec_err1]))
        self.pointing_error2 = np.concatenate(([t], [arcsec_err2]))

        self.pointing_error = np.concatenate(([t], [arcsec_err1, arcsec_err2]))

    def update(self, t, dt):
        self.sat1.update(t, dt, self.target)
        self.sat2.update(t, dt, self.target)

        # Calculate earth's rotation from time step
        self.theta_E += dt * ol.w_E

        # Used for plotting Error
        ri1, _, q1, _ = self.sat1.get_state()
        ri2, _, q2, _ = self.sat2.get_state()

        q_oG = su.Quaternion([0, 1, 0, 0]) # Gaussian frame
        arcsec_err1 = error_in_arcsec(q_oG.conjugated() @ self.target[0].conjugated() @ q1)
        arcsec_err2 = error_in_arcsec(q_oG.conjugated() @ self.target[0].conjugated() @ q2)

        self.pointing_error1 = np.vstack((self.pointing_error1, np.concatenate(([t], [arcsec_err1]))))
        self.pointing_error2 = np.vstack((self.pointing_error2, np.concatenate(([t], [arcsec_err2]))))
        self.pointing_error = np.vstack((self.pointing_error, np.concatenate(([t], [arcsec_err1, arcsec_err2]))))

        progress_bar_update(dt, f"ADCS PD: {arcsec_err1:.5f} arcsec || ADCS SM: {arcsec_err2:.5f} arcsec :") # Update progress bar

    def get(self):
        ri, _, q, _ = self.sat2.get_state() # Render satellite 2 (ADCS SM)

        temp = ol.polar2xyz(1, self.theta_E / 2)  # Normalized XY from q_E
        q_E = su.Quaternion([temp[0], 0, 0, temp[1]])

        return [
            ['satellite', ri, q],
            ['body_frame', ri, q],
            ['earth', np.zeros(3), q_E],
            ['ECEF frame', np.zeros(3), q_E],
            ['ECI frame', np.zeros(3), su.Quaternion()]
        ]

    def post_process(self, t, dt):
        progress_bar_close() # Close progress bar

        file = su.log_pos("assignment9_pointing_error1", self.pointing_error1)
        #self.pointing_error1 = None  # Clear the data after its saved
        pl.line_plot(file, labels=None, x_axis="Time [s]", y_axis="Arcsec",titel="Pointing error (PD)", linestyle=None)

        file = su.log_pos("assignment9_pointing_error_sub1", self.pointing_error1[(3600//dt):])
        self.pointing_error1 = None  # Clear the data after its saved
        pl.line_plot(file, labels=None, x_axis="Time [s]", y_axis="Arcsec", titel="Pointing error (PD)", linestyle=None)

        file = su.log_pos("assignment9_pointing_error2", self.pointing_error2)
        # self.pointing_error2 = None  # Clear the data after its saved
        pl.line_plot(file, labels=None, x_axis="Time [s]", y_axis="Arcsec", titel="Pointing error (SM)", linestyle=None)

        file = su.log_pos("assignment9_pointing_error_sub2", self.pointing_error2[(3600//dt):])
        self.pointing_error2 = None  # Clear the data after its saved
        pl.line_plot(file, labels=None, x_axis="Time [s]", y_axis="Arcsec", titel="Pointing error (SM)", linestyle=None)

        file = su.log_pos("assignment9_pointing_error", self.pointing_error)
        # self.pointing_error = None  # Clear the data after its saved
        pl.line_plot(file, labels=["PD", "SM"], x_axis="Time [s]", y_axis="Arcsec", titel="Pointing error", linestyle=None)

        file = su.log_pos("assignment9_pointing_error_sub", self.pointing_error[(3600//dt):])
        self.pointing_error = None  # Clear the data after its saved
        pl.line_plot(file, labels=["PD", "SM"], x_axis="Time [s]", y_axis="Arcsec", titel="Pointing error", linestyle=None)

class Part2Task2(sim.BaseScenario):
    def __init__(self, file_path):
        self.sat1 = None
        self.sat2 = None

        # Are the same for both satellites
        self.theta_E = None
        self.target = None

        self.pointing_error = None
        self.pointing_error1 = None
        self.pointing_error2 = None

        self.orbit1, self.JD1 = PKepler_from_tle_params(file_path, index=0, debug=False)  # HST1 (ADCS_PD)
        self.orbit2, self.JD2 = PKepler_from_tle_params(file_path, index=0, debug=False)  # HST1 (ADCS_SM)

        #self.fine = False

    def init(self, t):
        q_ib = su.Quaternion([1, 0, 0, 0])
        w_bib = np.array([0.3, -0.1, 2]) * 1E-3
        q_id = su.Quaternion([0.5, 0.5, 0.5, 0.5])
        w_did = np.zeros(3)
        dw_did = np.zeros(3)

        self.target = (q_id, w_did, dw_did) # Set target tuple

        # kg * m ** 2
        J = np.array([
            [36046,  -706,  1491],
            [ -706, 86868,   449],
            [ 1491,   449, 93848]
        ])

        # Add Star-sensor & gyro
        sensors1 = [
            sl.StarTracker(su.Quaternion([1, 0, 0, 0]), su.Quaternion([1, 0, 0, 0]), 0, np.zeros(3), 1e-2),  # 1E-2
            sl.Gyro(su.Quaternion([1, 0, 0, 0]), np.zeros(3), np.zeros(3), 1e-6, 0, np.zeros(3))  # 1E-6
        ]

        sensors2 = [
            sl.StarTracker(su.Quaternion([1, 0, 0, 0]), su.Quaternion([1, 0, 0, 0]), 0, np.zeros(3), 1e-2),  # 1E-2
            sl.StarTracker(su.Quaternion([1, 0, 0, 0]), su.Quaternion([1, 0, 0, 0]), 0, np.zeros(3), 1e-2),  # 1E-2
            sl.StarTracker(su.Quaternion([1, 0, 0, 0]), su.Quaternion([1, 0, 0, 0]), 0, np.zeros(3), 1e-2),  # 1E-2
            sl.Gyro(su.Quaternion([1, 0, 0, 0]), np.zeros(3), np.zeros(3), 1e-6, 0, np.zeros(3))  # 1E-6
        ]

        self.theta_E = ol.sidereal_angle(self.JD1)  # Offset to the rotation

        # Create ADCS
        ADCS1 = sl.ADCS_SM(2e-2, 2e-5, 2e-5, J, JD=self.JD2, estimator=sl.Davenport(), sensors=sensors1)
        self.sat1 = sl.Satellite(q_ib, w_bib, J, sensors=sensors1, ADCS=ADCS1, JD=self.JD1,orbit=self.orbit1, substeps=50, estimator=sl.Davenport())

        # Due to sensor noise make targeting coefficient aggressive 2e-2, 2e-5, 2e-5
        ADCS2 = sl.ADCS_SM(2e-2, 2e-5, 2e-5, J, JD=self.JD2, estimator=sl.Davenport(), sensors=sensors2)
        #ADCS = sl.ADCS_SM(1.2e-3, 1.0e-5, 3.0e-6, J, JD=self.JD, estimator=sl.Davenport(), sensors=[*star_sensors, gyro_sensor])
        self.sat2 = sl.Satellite(q_ib, w_bib, J, sensors=sensors2, ADCS=ADCS2, JD=self.JD2, orbit=self.orbit2, substeps=50, estimator=sl.Davenport())

        # Used for plotting Error
        ri1, _, q1, _ = self.sat1.get_state()
        ri2, _, q2, _ = self.sat2.get_state()

        q_oG = su.Quaternion([0, 1, 0, 0]) # Gaussian frame
        arcsec_err1 = error_in_arcsec(q_oG.conjugated() @ self.target[0].conjugated() @ q1)
        arcsec_err2 = error_in_arcsec(q_oG.conjugated() @ self.target[0].conjugated() @ q2)

        self.pointing_error1 = np.concatenate(([t], [arcsec_err1]))
        self.pointing_error2 = np.concatenate(([t], [arcsec_err2]))

        self.pointing_error = np.concatenate(([t], [arcsec_err1, arcsec_err2]))

    def update(self, t, dt):
        self.sat1.update(t, dt, self.target)
        self.sat2.update(t, dt, self.target)

        # Calculate earth's rotation from time step
        self.theta_E += dt * ol.w_E

        # Used for plotting Error
        ri1, _, q1, _ = self.sat1.get_state()
        ri2, _, q2, _ = self.sat2.get_state()

        q_oG = su.Quaternion([0, 1, 0, 0]) # Gaussian frame
        arcsec_err1 = error_in_arcsec(q_oG.conjugated() @ self.target[0].conjugated() @ q1)
        arcsec_err2 = error_in_arcsec(q_oG.conjugated() @ self.target[0].conjugated() @ q2)

        self.pointing_error1 = np.vstack((self.pointing_error1, np.concatenate(([t], [arcsec_err1]))))
        self.pointing_error2 = np.vstack((self.pointing_error2, np.concatenate(([t], [arcsec_err2]))))
        self.pointing_error = np.vstack((self.pointing_error, np.concatenate(([t], [arcsec_err1, arcsec_err2]))))

        progress_bar_update(dt, f"ADCS 1 Tracker: {arcsec_err1:.5f} arcsec || ADCS 3 Tracker: {arcsec_err2:.5f} arcsec :") # Update progress bar

    def get(self):
        ri, _, q, _ = self.sat2.get_state() # Render satellite 2 (ADCS SM)

        temp = ol.polar2xyz(1, self.theta_E / 2)  # Normalized XY from q_E
        q_E = su.Quaternion([temp[0], 0, 0, temp[1]])

        return [
            ['satellite', ri, q],
            ['body_frame', ri, q],
            ['earth', np.zeros(3), q_E],
            ['ECEF frame', np.zeros(3), q_E],
            ['ECI frame', np.zeros(3), su.Quaternion()]
        ]

    def post_process(self, t, dt):
        progress_bar_close()  # Close progress bar

        file = su.log_pos("assignment9_pointing_error1_2", self.pointing_error1)
        # self.pointing_error1 = None  # Clear the data after its saved
        pl.line_plot(file, labels=None, x_axis="Time [s]", y_axis="Arcsec", titel="Pointing error 1 star tracker", linestyle=None)

        file = su.log_pos("assignment9_pointing_error_sub1_2", self.pointing_error1[(3600 // dt):])
        self.pointing_error1 = None  # Clear the data after its saved
        pl.line_plot(file, labels=None, x_axis="Time [s]", y_axis="Arcsec", titel="Pointing error 1 star tracker", linestyle=None)

        file = su.log_pos("assignment9_pointing_error2_2", self.pointing_error2)
        # self.pointing_error2 = None  # Clear the data after its saved
        pl.line_plot(file, labels=None, x_axis="Time [s]", y_axis="Arcsec", titel="Pointing error 1 star tracker", linestyle=None)

        file = su.log_pos("assignment9_pointing_error_sub2_2", self.pointing_error2[(3600 // dt):])
        self.pointing_error2 = None  # Clear the data after its saved
        pl.line_plot(file, labels=None, x_axis="Time [s]", y_axis="Arcsec", titel="Pointing error 3 star tracker", linestyle=None)

        file = su.log_pos("assignment9_pointing_error_2", self.pointing_error)
        # self.pointing_error = None  # Clear the data after its saved
        pl.line_plot(file, labels=["1 star tracker", "3 star tracker"], x_axis="Time [s]", y_axis="Arcsec", titel="Pointing error",
                     linestyle=None)

        file = su.log_pos("assignment9_pointing_error_sub_2", self.pointing_error[(3600 // dt):])
        self.pointing_error = None  # Clear the data after its saved
        pl.line_plot(file, labels=["1 star tracker", "3 star tracker"], x_axis="Time [s]", y_axis="Arcsec", titel="Pointing error",
                     linestyle=None)

def main():
  file_path = "Assignment9/TLE.txt"

  ##############################
  # Assignment 9 Part 1 Task 1 #
  ##############################

  # Load TLE file and get all elements assigned to it
  epoch, e, rev, Me, omega, i, w, dn, d2n = ol.orbit_params_from_tle_params(file_path, debug=False, index=0)

  # Do Part 1 Task 1 & 2
  scenario = Part1Task2(file_path)
  T = ol.orbital_period_from_revs_per_day(rev) # Assume the other orbits are the same as this
  sim_config = {'t_0': 0, 't_e': T, 't_step': 2, 'speed_factor': 100, 'anim_dt': 0.04, 'scale_factor': 1000,'visualise': True}
  #sim.create_and_start_simulation(sim_config,scenario) # Don't use when working in part 2

  # Do Part 2 Task 1
  scenario = Part2Task1(file_path)
  progress_bar(T*4) # Create progress bar
  sim_config = {'t_0': 0, 't_e': T*4, 't_step': 10, 'speed_factor': 100, 'anim_dt': 0.04, 'scale_factor': 1000,'visualise': True}
  #sim.create_and_start_simulation(sim_config, scenario)
  progress_bar_close()

  # Do Part 2 Task 2
  scenario = Part2Task2(file_path)
  progress_bar(T * 4)  # Create progress bar
  sim_config = {'t_0': 0, 't_e': T * 2, 't_step': 10, 'speed_factor': 100, 'anim_dt': 0.04, 'scale_factor': 1000,'visualise': True}
  sim.create_and_start_simulation(sim_config, scenario)

if __name__ == "__main__":
    main()
