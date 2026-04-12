import sys

import numpy as np
import simutils as su
import orbit_lib as ol
import simulator as sim
import math

import plotter as pl
from orbit_lib import ground_track

# Extends upon the Base Scenario template from simulator
class ScenarioAssignment1(sim.BaseScenario):
    def __init__(self, args: list[str]):

        ### Convert arguments to values ###
        JD = ol.epoch_to_julian_date(args[0])
        self.theta_G0 = ol.sidereal_angle(JD)

        # Julian Date information
        print(f"Julian Date: {JD}")
        print(f"Sidereal Angle: {self.theta_G0} [{ol.rad2deg(self.theta_G0):.2f}]")

        print(ol.julian_date_to_iso(JD))

        # Orbit params
        self.ri0, self.vi0 = ol.state_from_tle_params(args[1:])

        self.theta_E = None
        self.q_E = None
        self.ri = None
        self.q = None
        self.n = None
        self.Me = None
        self.w = None
        self.i = None
        self.omega = None
        self.e = None
        self.h = None
        self.pos_plot = None
        self.ground_track_plot = None


    def init(self, t):

        # Retrieve orbit parameters from initial ri and vi
        self.h, self.e, theta, self.omega, self.i, self.w = ol.orbit_params_from_state(self.ri0, self.vi0)

        # Conversion madness! True anomaly [theta] -> Eccentric anomaly [E] -> Mean anomaly [Me]
        self.Me = ol.mean_anomaly_from_true_anomaly(theta, self.e)

        # Mean motion
        a = self.h ** 2 / (ol.mu * (1 - self.e ** 2))
        T = ol.orbital_period_from_semi_major_axis(a)
        self.n = 2 * math.pi / T

        self.q = su.Quaternion() # Satellite rotation
        self.ri = self.ri0  # Satellite position

        # Earth rotation variables
        self.theta_E = self.theta_G0 # Offset to the rotation
        temp = ol.polar2xyz(1, self.theta_E / 2) # Normalized XY from q_E
        self.q_E = su.Quaternion([temp[0], 0, 0, temp[1]])
        #self.q_E = su.Quaternion()

        # Data logging variables
        self.pos_plot = np.concatenate(([t], self.ri)) # Initialize the plot data

        lon, lat = ol.ground_track(self.ri, self.theta_E)
        self.ground_track_plot = np.concatenate(([t], [lon, lat]))


    def update(self, t, dt):

        # Propagate mean anomaly to it's next value in regard to time step
        self.Me = ol.angle_wrap_radians(self.Me + self.n * dt)

        # Conversion madness! Mean anomaly [Me] -> Eccentric anomaly [E] -> True anomaly [theta]
        theta = ol.true_anomaly_from_eccentric_anomaly(ol.eccentric_anomaly_from_mean_anomaly(self.Me, self.e), self.e)

        # Get the new ri, vi from updated theta
        self.ri, vi = ol.state_from_orbit_params(self.h, self.e, theta, self.omega, self.i, self.w)

        # Calculate earth's rotation from time step
        self.theta_E += dt * ol.w_E
        temp = ol.polar2xyz(1, self.theta_E / 2) # Normalized XY from q_E
        self.q_E = su.Quaternion([temp[0], 0, 0, temp[1]])

        # Log orbit data
        self.pos_plot = np.vstack((self.pos_plot, np.concatenate(([t], self.ri))))

        lon, lat = ol.ground_track(self.ri, self.theta_E) # Ground track
        self.ground_track_plot = np.vstack((self.ground_track_plot, np.concatenate(([t], [lon, lat]))))


    def get(self):
        return [
            ['satellite', self.ri, self.q],
            ['body_frame', self.ri, self.q],
            ['earth', np.zeros(3), self.q_E],
            ['ECEF frame', np.zeros(3), self.q_E],
            ['ECI frame', np.zeros(3), su.Quaternion()]]


    def post_process(self, t, dt):
        # Plot orbit of satellite
        file = su.log_pos("assignment2_position", self.pos_plot)
        self.pos_plot = None # Clear the data after its saved
        pl.line_plot(file)

        file = su.log_pos("assignment2_ground_track", self.ground_track_plot)
        self.ground_track_plot = None  # Clear the data after its saved
        pl.ground_tracking(file, "3DModels/earth_8k.jpg")


def main():
  #scenario = sim.BaseScenario()
  #scenario = ScenarioAssignment1()
  #sim.create_and_start_simulation(sim_config,scenario)

  # Read the TLE file
  file_path = "Assignment2/VANGUARD_1_TLE.txt"
  #file_path = "Assignment2/SAT_40613_TLE.txt"
  #file_path = "Assignment2/MOLNIYA_1_91_TLE.txt"

  try:
      with open(file_path, "r") as f:
          tle_text = f.read()

  except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    return

  # Get all necessary fields as arguments
  args = ol.orbit_params_from_tle_params(tle_text, debug=True)

  T = ol.orbital_period_from_revs_per_day(float(args[1:][5][:11]))

  sim_config = {'t_0': 0, 't_e': T, 't_step': 1, 'speed_factor': 2000, 'anim_dt': 0.04, 'scale_factor': 2000, 'visualise': True}
  scenario = ScenarioAssignment1(args)
  sim.create_and_start_simulation(sim_config,scenario)



if __name__ == "__main__":
    main()
