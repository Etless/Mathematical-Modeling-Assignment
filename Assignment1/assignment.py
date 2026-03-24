import numpy as np
import simutils as su
import sat_lib as sl
import orbit_lib as ol
import simulator as sim
import math

import plotter as pl

def PolarToPoint(r, angle):
    return math.cos(angle)*r, math.sin(angle)*r, 0

# Orbit period
sat_r = ol.R_E + 400
T = 2 * math.pi * math.sqrt(((sat_r * 1000) ** 3)/(3.986004418 * 10**14))

# Extends upon the Base Scenario template from simulator
class ScenarioAssignment1(sim.BaseScenario):
    # Overriding
    def init(self, t):
        self.r = sat_r
        self.angle = 0
        angle_G = 0 # Not specified
        #self.q_E = su.Quaternion([math.cos(angle_G/2), 0, 0, math.sin(angle_G/2)])
        self.q_E = su.Quaternion()

        w_ie = 7.2921*10**(-5)
        self.omega_q = su.Quaternion([0, 0, w_ie])

        self.pos_plot = np.array([t, self.r * np.cos(self.angle), self.r * np.sin(self.angle), 0])
        self.r_i = np.array(PolarToPoint(self.r, self.angle))

    # Overriding
    def update(self, t, dt):
        self.angle += 2*math.pi/T * dt
        #self.q_E = 0.5 * self.q_E @ self.omega_q
        self.r_i = np.array(PolarToPoint(self.r, self.angle))
        self.pos_plot = np.vstack((self.pos_plot, np.array([t, self.r_i[0], self.r_i[1], self.r_i[2]])))

    # Overriding
    def get(self):
        return [
            ['satellite', self.r_i, su.Quaternion()],
            ['body_frame', self.r_i, su.Quaternion()],
            ['earth', np.zeros(3), self.q_E],
            ['ECI frame', np.zeros(3), su.Quaternion()]]
        """return [
            ['satellite', r_i, self.q],
            ['body_frame', r_i, self.q],
            ['earth', np.zeros(3), self.q_E],
            ['ECI frame',np.zeros(3), su.Quaternion()]]"""

    # Overriding
    def post_process(self, t, dt):
        file = su.log_pos("assignment1_position", self.pos_plot)#, path="C:/Users/Askar/PycharmProjects/Mathematical-Modeling-Assignment/Assignment1/data/")
        pl.main(["null", "lineplot", file])

def main():
  sim_config = {'t_0':0,'t_e':T,'t_step':1,'speed_factor':100,'anim_dt':0.04,'scale_factor':1000,'visualise':True}
  #scenario = sim.BaseScenario()
  scenario = ScenarioAssignment1()
  sim.create_and_start_simulation(sim_config,scenario)

if __name__ == "__main__":
    main()
