import numpy as np
import simutils as su
import orbit_lib as ol
import simulator as sim
import math

import plotter as pl

# Extends upon the Base Scenario template from simulator
class ScenarioAssignment1(sim.BaseScenario):
    def __init__(self):
        self._x = None
        self.RK4_x = None
        self.orbit_energy_plot = None
        self.m = None
        self.vi = None
        self.ri = None
        self.verlet_x_past = None
        self.verlet_x = None
        self.leapfrog_x = None
        self.q = None
        self.q_E = None
        self.euler_x = None
        self.pos_plot = None
        self.ground_track_plot = None
        self.theta_E = None

    def init(self, t):

        # Satellite variables
        self.q = su.Quaternion() # Satellite rotation
        self.ri = np.array([7378, 0, 0]) # Satellite position
        self.vi = np.array([0, 0, 9]) # Satellite velocity

        self.m = 8000 # Mass of the satellite [kg]

        v_temp = math.sqrt(ol.mu / (ol.R_E + 800))

        # Position from each integration method
        self.euler_x = np.concatenate([[ol.R_E + 800, 0, 0],[0, v_temp, 0]])
        self.leapfrog_x = np.concatenate([[ol.R_E + 800, 0, 0],[0, v_temp, 0]])

        self.verlet_x_past = None
        self.verlet_x = np.concatenate([[ol.R_E + 800, 0, 0],[0, v_temp, 0]])

        self.RK4_x = np.concatenate([[ol.R_E + 800, 0, 0],[0, v_temp, 0]])
        # Used for Assignment 3.2
        self._x = np.concatenate([self.ri, self.vi])

        # Earth rotation variables
        self.theta_E = 0 # Offset to the rotation
        temp = ol.polar2xyz(1, self.theta_E / 2) # Normalized XY from q_E
        self.q_E = su.Quaternion([temp[0], 0, 0, temp[1]])
        #self.q_E = su.Quaternion()

        # Data logging variables
        self.pos_plot = np.concatenate(([t], self.ri)) # Initialize the plot data

        lon, lat = ol.ground_track(self.ri, self.theta_E)
        self.ground_track_plot = np.concatenate(([t], [lon, lat])) # Initialize ground track data

        # Convert all energies to array
        self.orbit_energy_plot = np.concatenate(([t], [ol.get_orbit_energy_state(self.euler_x, self.m), ol.get_orbit_energy_state(self.leapfrog_x, self.m), ol.get_orbit_energy_state(self.verlet_x, self.m), ol.get_orbit_energy_state(self.RK4_x, self.m)]))


    def update(self, t, dt):

        # Get the next "step" of the satellite
        self.euler_x = su.step_euler(dt, t, self.euler_x, su.two_body)
        self.leapfrog_x = su.step_leapfrog(dt, t, self.leapfrog_x, su.two_body)
        self.verlet_x, self.verlet_x_past = su.step_verlet(dt, t, self.verlet_x, self.verlet_x_past, su.two_body), self.verlet_x
        self.RK4_x = su.step_RK4(dt, t, self.RK4_x, su.two_body)

        # Used for Assignment 3.2
        k1 = 10e-3
        k2 = 10e-3
        ei = ol.get_orbit_eccentricity_vector_state(self._x)
        e = np.linalg.norm(ei).astype(float)

        cos_theta = np.dot(ei, self._x[:3]) / (e * np.linalg.norm(self._x[:3]).astype(float))
        ra = ol.get_orbit_apoapsis(self._x, e)
        rp = ol.get_orbit_periapsis(self._x, e)
        rc = ol.R_E + 1500

        T = (k1 * (rc - ra)) if cos_theta > 0.9 else (k2 * (rc - rp)) if cos_theta < -0.9 else 0

        ae = (T * self._x[3:] / np.linalg.norm(self._x[3:]).astype(float)) / self.m

        self._x = su.step_RK4(dt, t, self._x, su.two_body, ae=ae)
        self.ri = self._x[:3]  # Get position vector

        # Calculate earth's rotation from time step
        self.theta_E += dt * ol.w_E
        temp = ol.polar2xyz(1, self.theta_E / 2) # Normalized XY from q_E
        self.q_E = su.Quaternion([temp[0], 0, 0, temp[1]])

        # Log orbit data
        self.pos_plot = np.vstack((self.pos_plot, np.concatenate(([t], self.ri))))

        lon, lat = ol.ground_track(self.ri, self.theta_E) # Ground track
        self.ground_track_plot = np.vstack((self.ground_track_plot, np.concatenate(([t], [lon, lat]))))

        self.orbit_energy_plot = np.vstack((self.orbit_energy_plot, np.concatenate(([t], [ol.get_orbit_energy_state(self.euler_x, self.m), ol.get_orbit_energy_state(self.leapfrog_x, self.m), ol.get_orbit_energy_state(self.verlet_x, self.m), ol.get_orbit_energy_state(self.RK4_x, self.m)]))))


    def get(self):
        return [
            ['satellite', self.ri, self.q],
            ['body_frame', self.ri, self.q],
            ['earth', np.zeros(3), self.q_E],
            ['ECEF frame', np.zeros(3), self.q_E],
            ['ECI frame', np.zeros(3), su.Quaternion()]]


    def post_process(self, t, dt):
        # Plot orbit of satellite
        file = su.log_pos("assignment4_position", self.pos_plot)
        self.pos_plot = None # Clear the data after its saved
        pl.line_plot(file)

        file = su.log_pos("assignment4_ground_track", self.ground_track_plot)
        self.ground_track_plot = None  # Clear the data after its saved
        pl.ground_tracking(file, "3DModels/earth_8k.jpg")

        file = su.log_pos("assignment4_energy", self.orbit_energy_plot)
        self.orbit_energy_plot = None  # Clear the data after its saved
        pl.line_plot(file, labels=["Euler", "Leapfrog", "Verlet", "RK4"])


def main():
  #scenario = sim.BaseScenario()
  #scenario = ScenarioAssignment1()
  #sim.create_and_start_simulation(sim_config,scenario)

  #sim_config = {'t_0': 0, 't_e': 20000, 't_step': 10, 'speed_factor': 100, 'anim_dt': 0.04, 'scale_factor': 1000, 'visualise': True}
  sim_config = {'t_0': 0, 't_e': 53000, 't_step': 100, 'speed_factor': 1, 'anim_dt': 0.04, 'scale_factor': 1000,'visualise': True}
  scenario = ScenarioAssignment1()
  sim.create_and_start_simulation(sim_config,scenario)



if __name__ == "__main__":
    main()
