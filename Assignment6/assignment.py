import math

import numpy as np
import simutils as su
import sat_lib as sl
import orbit_lib as ol
import simulator as sim

import plotter as pl

# Extends upon the Base Scenario template from simulator
class ScenarioAssignment1(sim.BaseScenario):
    def __init__(self, file_path):
        self.theta_E = None
        self.sat = None
        self.q  = None
        self.q_E = None
        self.tau = None

        # Load TLE file and get all elements assigned to it
        epoch, e, rev, Me, omega, i, w = ol.orbit_params_from_tle_params(file_path, debug=True)

        ### Convert arguments to values ###
        JD = ol.epoch_to_julian_date(epoch)
        self.theta_G0 = ol.sidereal_angle(JD)

        # Julian Date information
        print(f"Julian Date: {JD}")
        print(f"Sidereal Angle: {self.theta_G0} [{ol.rad2deg(self.theta_G0):.2f}]")

        print(ol.julian_date_to_iso(JD))

        # Orbit params
        T = ol.orbital_period_from_revs_per_day(rev)
        n = 2 * math.pi / T

        self.r0, self.v0 = ol.state_from_tle_params(e, n, Me, omega, i, w)  # Satellite position

        self.ground_track_plot = None

    def init(self, t):
        q0 = su.Quaternion([1, 0, 0, 0])
        w0 = np.array([0, 0, 0])
        J = np.array([
            [ 0.00146519,  0.00001703, -0.00000633],
            [ 0.00001703,  0.00151512, -0.00001598],
            [-0.00000633, -0.00001598,  0.00146333],
        ])

        self.sat = sl.Satellite(q0, w0, J, self.r0, self.v0, substeps=50)

        self.q_E = su.Quaternion()

        # Earth rotation variables
        self.theta_E = self.theta_G0  # Offset to the rotation
        temp = ol.polar2xyz(1, self.theta_E / 2)  # Normalized XY from q_E
        self.q_E = su.Quaternion([temp[0], 0, 0, temp[1]])

        # Not needed just for fun
        ri, _, _, _ = self.sat.get_state()
        lon, lat, _ = ol.geodetic_from_xyz(ri)
        lon = ol.angle_wrap_radians(lon + math.pi) - math.pi
        # lon, lat = ol.ground_track(self.ri, self.theta_E)  # Ground track
        self.ground_track_plot = np.concatenate(([t], [lon, lat]))

    def update(self, t, dt):
        self.sat.update(t, dt)

        # Calculate earth's rotation from time step
        self.theta_E += dt * ol.w_E
        temp = ol.polar2xyz(1, self.theta_E / 2)  # Normalized XY from q_E
        self.q_E = su.Quaternion([temp[0], 0, 0, temp[1]])

        # Not needed just for fun
        ri, _, _, _ = self.sat.get_state()
        lon, lat, _ = ol.geodetic_from_xyz(ri)
        lon = ol.angle_wrap_radians(lon + math.pi) - math.pi

        #lon, lat = ol.ground_track(self.ri, self.theta_E)  # Ground track
        self.ground_track_plot = np.vstack((self.ground_track_plot, np.concatenate(([t], [lon, lat]))))

    def get(self):
        ri, _, q, _ = self.sat.get_state()

        return [
            ['satellite', ri, q],
            ['body_frame', ri, q],
            ['earth', np.zeros(3), self.q_E],
            ['ECEF frame', np.zeros(3), self.q_E],
            ['ECI frame', np.zeros(3), su.Quaternion()]
        ]

    def post_process(self, t, dt):
        # Not needed just for fun
        file = su.log_pos("assignment2_ground_track", self.ground_track_plot)
        self.ground_track_plot = None  # Clear the data after its saved
        pl.ground_tracking(file, "3DModels/earth_8k.jpg")


def main():
  file_path = "Assignment5/HINCUBE.txt"
  scenario = ScenarioAssignment1(file_path)

  sim_config = {'t_0': 0, 't_e': 5731, 't_step': 2, 'speed_factor': 100, 'anim_dt': 0.04, 'scale_factor': 1000,'visualise': True}
  sim.create_and_start_simulation(sim_config,scenario)

if __name__ == "__main__":
    main()
