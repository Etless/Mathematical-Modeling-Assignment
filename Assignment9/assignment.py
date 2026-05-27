import math

import numpy as np
import simutils as su
import sat_lib as sl
import orbit_lib as ol
import Assignment9.simulator as sim

import plotter as pl

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


class Part1Task2(sim.BaseScenario):
    def __init__(self, file_path):
        self.ri = None
        self.q = None
        self.theta_E = None

        self.orbit2_plot = None
        self.orbit3_plot = None
        self.orbit_difference_plot = None

        # Orbit to look 9 years in the future
        self.orbit1, self.JD1 = PKepler_from_tle_params(file_path, index=0, debug=False) # HST1

        # Orbit of current year
        self.orbit2, self.JD2 = PKepler_from_tle_params(file_path, index=0, debug=False) # HST1
        # Orbit 9 years ago to be propigated to current year
        self.orbit3, self.JD3 = PKepler_from_tle_params(file_path, index=1, debug=False) # HST2

    def init(self, t):

        print("Initial ISO Time:")
        print("Orbit 1:", ol.julian_date_to_iso(self.JD1))
        print("Orbit 2:", ol.julian_date_to_iso(self.JD2))
        print("Orbit 3:", ol.julian_date_to_iso(self.JD3))

        # Calculate delta time between orbit2 (HST1) and orbit3 (HST2)
        dt_23 = (24.0 * 3600.0) * (self.JD2 - self.JD3)

        # Calculate delta time for orbit1 (HST1) to propigate 9 years in the future
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

        # !!! Not needed just to make simultaion look more realistic
        self.theta_E = ol.sidereal_angle(self.JD1 + dt_1 / (24.0 * 3600.0)) # Rotation of earth at updated orbit1 (HST1)

        # Used to plot the difference between orbit2 and orbit3
        ri2, _ = self.orbit2.get_state()
        r2 = np.linalg.norm(ri2)
        ri3, _ = self.orbit3.get_state()
        r3 = np.linalg.norm(ri3)
        self.orbit2_plot = np.concatenate(([t], [*ri2, r2]))
        self.orbit3_plot = np.concatenate(([t], [*ri3, r3]))

        r_diff = r2 - r3
        self.orbit_difference_plot = np.concatenate(([t], [r_diff]))

        a, e, Me, omega, i, w, dn, d2n = self.orbit1.get_params()

        E = ol.eccentric_anomaly_from_mean_anomaly(Me, e)
        theta = ol.true_anomaly_from_eccentric_anomaly(E, e)

        n = 2 * math.pi / ol.orbital_period_from_semi_major_axis(a)

        # Note: Underlying functions use the more standard latitude = phi and longitude = lam
        lam, phi, height = ol.geodetic_from_xyz(self.ri)

        q_io, w_iio, dw_iio = ol.orbit_frame_from_state(self.ri, vi)

        h, _, _, _, _, _ = ol.orbit_params_from_state(self.ri, vi)

        print(" === === Assignment 9 Part 1 Task 2 === ===")
        print("Specific Relative Angular Momentum : h      ", h)
        print("True Anomaly                       : θ      ", ol.rad2deg(ol.angle_wrap_radians(theta)))
        print("Eccentric Anomaly                  : E      ", ol.rad2deg(ol.angle_wrap_radians(E)))
        print("Semi-Major Axis                    : a      ", a)
        print("Mean Motion                        : n      ", n)
        print("Derivative of Mean Motion          : dn     ", dn)
        print("Second Derivative of Mean Motion   : d2n    ", d2n)
        print("Position                           : ri     ", self.ri)
        print("Velocity                           : vi     ", vi)
        print("Julian Date                        : JD     ", self.JD1 + dt_1 / (24.0 * 3600.0))
        print("Sidereal Angle                     : θG0    ", ol.rad2deg(ol.angle_wrap_radians(self.theta_E)))
        print("Orbit frame                        : q_io   ", q_io[:])
        print("Orbit frame Angular Velocity       : w_i_io ", w_iio)
        print("Orbit frame Angular Acceleration   : dw_i_io", dw_iio)
        print("Geodetic Latitude                  : λ′     ", lam)
        print("Geodetic/Geocentric longitude      : ϕ      ", phi)
        print("Altitude                           : h      ", height)

    def update(self, t, dt):

        # Propagate the orbit
        self.orbit1.propagate(dt)
        self.orbit2.propagate(dt)
        self.orbit3.propagate(dt)

        # Get states from orbit1
        self.ri, _ = self.orbit1.get_state()
        self.q, _, _ = self.orbit1.get_orbit_frame()

        # Calculate earth's rotation from time step
        self.theta_E += dt * ol.w_E

        # Used to plot the difference between orbit2 and orbit3
        ri2, _ = self.orbit2.get_state()
        r2 = np.linalg.norm(ri2)
        ri3, _ = self.orbit3.get_state()
        r3 = np.linalg.norm(ri3)
        self.orbit2_plot = np.vstack((self.orbit2_plot, np.concatenate(([t], [*ri2, r2]))))
        self.orbit3_plot = np.vstack((self.orbit3_plot, np.concatenate(([t], [*ri3, r3]))))

        r_diff = r2 - r3
        self.orbit_difference_plot = np.vstack((self.orbit_difference_plot, np.concatenate(([t], [r_diff]))))

    def get(self):
        temp = ol.polar2xyz(1, self.theta_E / 2)  # Normalized XY from q_E
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
        pl.line_plot(file, labels=None, x_axis="Time [s]", y_axis="Difference [km]", titel="Differenece in height")

class Part2Task1(sim.BaseScenario):
    def __init__(self, file_path):
        self.target = None

    def init(self, t):
        q_ib = su.Quaternion([1, 0, 0, 0])
        w_bib = np.array([0.3, -0.1, 2]) * 1E-3
        q_id = su.Quaternion([0.5, 0.5, 0.5, 0.5])
        w_did = np.zeros(3)
        dw_did = np.zeros(3)

        self.target = (q_id, w_did, dw_did)

        # kg * m ** 2 -> kg * km ** 2
        J = np.array([
            [36046,  -706,  1491],
            [ -706, 86868,   449],
            [ 1491,   449, 93848]
        ]) / 1000 ** 2

        # Add Star-sensor
        star_sensors = [
            sl.StarTracker(su.Quaternion([1, 0, 0, 0]), su.Quaternion([1, 0, 0, 0]), 0, np.zeros(3), 1E-2)
        ]

        # Add a gyro
        gyro_sensor = sl.Gyro(su.Quaternion([1, 0, 0, 0]), np.zeros(3), np.zeros(3), 1E-6, 0, np.zeros(3))

        # Create ADCS


        self.sat = sl.Satellite(q_ib, w_bib, J, sensors=[*star_sensors, gyro_sensor], ADCS=None, JD=0, orbit=None, substeps=0)

    def update(self, t, dt):
        self.sat.update(t, dt, self.target)

        # Calculate earth's rotation from time step
        self.theta_E += dt * ol.w_E

    def get(self):
        ri, _, q, _ = self.sat.get_state()

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
        pass


def main():
  file_path = "Assignment9/TLE.txt"

  ##############################
  # Assignment 9 Part 1 Task 1 #
  ##############################

  # Load TLE file and get all elements assigned to it
  epoch, e, rev, Me, omega, i, w, dn, d2n = ol.orbit_params_from_tle_params(file_path, debug=False, index=0)

  E     = ol.eccentric_anomaly_from_mean_anomaly(Me, e)
  theta = ol.true_anomaly_from_eccentric_anomaly(E, e)

  JD = ol.epoch_to_julian_date(epoch)
  theta_G0 = ol.sidereal_angle(JD)

  n = 2 * math.pi / ol.orbital_period_from_revs_per_day(rev)

  a = ol.semi_major_axis_from_mean_motion(n)

  ri, vi = ol.state_from_tle_params(e, n, Me, omega, i, w)  # Satellite position

  # Note: Underlying functions use the more standard latitude = phi and longitude = lam
  lam, phi, height = ol.geodetic_from_xyz(ri)

  q_io, w_iio, dw_iio = ol.orbit_frame_from_state(ri, vi)

  h, _, _, _, _, _= ol.orbit_params_from_state(ri, vi)

  print(" === === Assignment 9 Part 1 Task 1 === ===")
  print("Specific Relative Angular Momentum : h      ", h)
  print("True Anomaly                       : θ      ", ol.rad2deg(ol.angle_wrap_radians(theta)))
  print("Eccentric Anomaly                  : E      ", ol.rad2deg(ol.angle_wrap_radians(E)))
  print("Semi-Major Axis                    : a      ", a)
  print("Mean Motion                        : n      ", n)
  print("Derivative of Mean Motion          : dn     ", dn)
  print("Second Derivative of Mean Motion   : d2n    ", d2n)
  print("Position                           : ri     ", ri)
  print("Velocity                           : vi     ", vi)
  print("Julian Date                        : JD     ", JD)
  print("Sidereal Angle                     : θG0    ", ol.rad2deg(ol.angle_wrap_radians(theta_G0)))
  print("Orbit frame                        : q_io   ", q_io[:])
  print("Orbit frame Angular Velocity       : w_i_io ", w_iio)
  print("Orbit frame Angular Acceleration   : dw_i_io", dw_iio)
  print("Geodetic Latitude                  : λ′     ", lam)
  print("Geodetic/Geocentric longitude      : ϕ      ", phi)
  print("Altitude                           : h      ", height)


  # Do Part 1 Task 2
  scenario = Part1Task2(file_path)
  T = ol.orbital_period_from_revs_per_day(rev) # Assume the other orbits are the same as this
  sim_config = {'t_0': 0, 't_e': T, 't_step': 2, 'speed_factor': 100, 'anim_dt': 0.04, 'scale_factor': 1000,'visualise': True}
  sim.create_and_start_simulation(sim_config,scenario)

  # Do Part 2 Task 1


if __name__ == "__main__":
    main()
