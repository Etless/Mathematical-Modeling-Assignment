import numpy as np
import simutils as su
import orbit_lib as ol
import sat_lib as sl
import simulator as sim
import math

import utils
import utils as ut
import plotter as pl
from orbit_lib import epoch_to_julian_date, state_from_tle_params

# Global variables
ri0 = None
vi0 = None

# Extends upon the Base Scenario template from simulator
class ScenarioAssignment1(sim.BaseScenario):
    def __init__(self):
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


    def init(self, t):

        # Catch variables from outside of scope
        global ri0, vi0

        # Retrieve orbit parameters from initial ri and vi
        self.h, self.e, theta, self.omega, self.i, self.w = ol.orbit_params_from_state(ri0, vi0)

        # Conversion madness! True anomaly [theta] -> Eccentric anomaly [E] -> Mean anomaly [Me]
        self.Me = ol.mean_anomaly_from_eccentric_anomaly(ol.eccentric_anomaly_from_true_anomaly(theta, self.e), self.e)

        # Mean motion
        a = self.h ** 2 / (ol.mu * (1 - self.e ** 2))
        T = ol.orbital_period_from_semi_major_axis(a)
        self.n = 2 * math.pi / T

        self.q = su.Quaternion() # Satellite rotation
        self.ri = ri0  # Satellite position

        # Earth rotation variables
        self.theta_E = 0 # Offset to the rotation
        #temp = ut.polar2coord(1, self.theta_E/2) # Normalized XY from q_E
        #self.q_E = su.Quaternion([temp[0], 0, 0, temp[1]])
        self.q_E = su.Quaternion()

        # Data logging variables
        self.pos_plot = np.concatenate(([t], self.ri)) # Initialize the plot data


    def update(self, t, dt):

        # Propagate mean anomaly to it's next value in regard to time step
        self.Me = ol.angle_wrap_radians(self.Me + self.n * dt)

        # Conversion madness! Mean anomaly [Me] -> Eccentric anomaly [E] -> True anomaly [theta]
        theta = ol.true_anomaly_from_eccentric_anomaly(ol.eccentric_anomaly_from_mean_anomaly(self.Me, self.e), self.e)

        # Get the new ri, vi from updated theta
        self.ri, vi = ol.state_from_orbit_params(self.h, self.e, theta, self.omega, self.i, self.w)

        # Calculate earth's rotation from time step
        self.theta_E += dt * ol.w_E
        temp = ut.polar2coord(1, self.theta_E / 2) # Normalized XY from q_E
        self.q_E = su.Quaternion([temp[0], 0, 0, temp[1]])

        # Log orbit data
        self.pos_plot = np.vstack((self.pos_plot, np.concatenate(([t], self.ri))))


    def get(self):
        return [
            ['satellite', self.ri, self.q],
            ['body_frame', self.ri, self.q],
            ['earth', np.zeros(3), self.q_E],
            ['ECEF frame', np.zeros(3), self.q_E],
            ['ECI frame', np.zeros(3), su.Quaternion()]]


    def post_process(self, t, dt):
        # Plot orbit of satellite
        file = su.log_pos("assignment1_position", self.pos_plot)
        self.pos_plot = None # Clear the data after its saved
        pl.line_plot(file)

def main():
  sim_config = {'t_0':0,'t_e':sl.T,'t_step':1,'speed_factor':200,'anim_dt':0.04,'scale_factor':2000,'visualise':True}
  #scenario = sim.BaseScenario()
  #scenario = ScenarioAssignment1()
  #sim.create_and_start_simulation(sim_config,scenario)

  # Read the TLE file
  file_path = "Assignment2/tle.txt"
  rows = []
  try:
      with open(file_path, "r") as f:
          rows = f.read().splitlines()

  except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    return
  print(f"TLE Name: '{rows[0]}' Data:\n{rows[1]}\n{rows[2]}")

  # Get all necessary fields as arguments
  args = [""] * 7
  fields = (" ".join(rows[1].split())).split(' ') # Fields in row 1
  args[0] = fields[3] # Epoch                           [int|int|float]

  fields = (" ".join(rows[2].split())).split(' ') # Fields in row 2
  args[1] = fields[2] # Inclination                     [float]
  args[2] = fields[3] # RAAN                            [float]
  args[3] = fields[4] # Eccentricity                    [float]
  args[4] = fields[5] # Argument of perigee             [float]
  args[5] = fields[6] # Mean anomaly                    [float]
  args[6] = fields[7] # Revs per day | Revs | Checksum  [float|int|int]

  ### Convert arguments to values ###
  JD = ol.epoch_to_julian_date(args[0])
  theta_G = ol.sidereal_angle(JD)

  # Julian Date information
  print(f"Julian Date: {JD}")
  print(f"Sidereal Angle: {theta_G} [{ol.rad2deg(theta_G):.2f}]")

  # Orbit params
  ri, vi = ol.state_from_tle_params(args[1:])
  #print(ri, vi)

  #ol.orbit_propagation(ri, vi)

  global ri0, vi0
  ri0 = ri
  vi0 = vi

  scenario = ScenarioAssignment1()
  sim.create_and_start_simulation(sim_config,scenario)



if __name__ == "__main__":
    main()
