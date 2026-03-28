import numpy as np
import simutils as su
import orbit_lib as ol
import sat_lib as sl
import simulator as sim
import math

import plotter as pl


# Extends upon the Base Scenario template from simulator
class ScenarioAssignment1(sim.BaseScenario):
    def __init__(self):
        self.r = None
        self.theta_offset = None
        self.theta = None

        self.q = None

        self.theta_E = None
        self.q_E = None

        self.r_i = None
        self.pos_plot = None


    def init(self, t):

        # Orbit variables [Satellite]
        self.r = sl.satellite_distance # Distance from earth center

        self.theta_offset = 0 # Offset to the orbit angle
        self.theta = self.theta_offset + 2 * math.pi / sl.T * t # Orbit angle bias + time angle

        # Satellite rotation
        self.q = su.Quaternion()

        # Earth rotation variables
        self.theta_E = 0 # Offset to the rotation
        temp = ol.polar2xyz(1, self.theta_E / 2) # Normalized XY from q_E
        self.q_E = su.Quaternion([temp[0], 0, 0, temp[1]])

        # Data logging variables
        self.r_i = ol.polar2xyz(self.r, self.theta) # Convert to XY
        self.pos_plot = np.concatenate(([t], self.r_i)) # Initialize the plot data


    def update(self, t, dt):
        # Orbit angle calculated from time step or time
        self.theta += 2 * math.pi / sl.T * dt
        #self.theta = self.theta_offset + 2 * math.pi / sl.T * t

        # Calculate earth's rotation from time step
        self.theta_E += dt * ol.w_E
        temp = ol.polar2xyz(1, self.theta_E / 2) # Normalized XY from q_E
        self.q_E = su.Quaternion([temp[0], 0, 0, temp[1]])

        # Log orbit data
        self.r_i = ol.polar2xyz(self.r, self.theta) # Convert to XY
        self.pos_plot = np.vstack((self.pos_plot, np.concatenate(([t], self.r_i))))


    def get(self):
        return [
            ['satellite', self.r_i, self.q],
            ['body_frame', self.r_i, self.q],
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
  scenario = ScenarioAssignment1()
  sim.create_and_start_simulation(sim_config,scenario)

if __name__ == "__main__":
    main()
