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
        self.sat = None
        self.body = None
        self.vi = None
        self.ri = None
        self.q  = None
        self.q_E = None
        self.tau = None

    def init(self, t):
        self.ri = np.zeros(3)
        """q0 = su.Quaternion([1, 0, 0, 0])
        w0 = np.array([0, 0, 5])
        J = np.array([
            [2,   1,   0],
            [1,  10, 0.1],
            [0, 0.1, 2.5],
        ])
        self.tau = np.array([0, 0, 0])
        self.body = sl.RigidBody(q0, w0, J)"""

        q0 = su.Quaternion([1, 0, 0, 0])
        w0 = np.array([0, 0, 0])
        J = np.array([
            [0.5,   0,   0],
            [  0, 0.5,   0],
            [  0,   0, 0.5],
        ])
        qd = su.Quaternion([0.5, 0.5, 0.5, 0.5])
        wd = np.array([0.2, -0.1, 0.05])

        self.sat = sl.Satellite(q0, w0, J, qd, wd)

        self.q_E = su.Quaternion()


    def update(self, t, dt):
        #self.body.update(t, dt, self.tau)
        #self.q = self.body.q

        self.sat.update(t, dt)


    def get(self):
        return [
            ['satellite', *self.sat.get_state()],
            ['body_frame', *self.sat.get_state()],
            #['earth', np.zeros(3), self.q_E],
            #['ECEF frame', np.zeros(3), self.q_E],
            #['ECI frame', np.zeros(3), su.Quaternion()]
        ]


    def post_process(self, t, dt):
        pass


def main():
  #scenario = sim.BaseScenario()
  #scenario = ScenarioAssignment1()
  #sim.create_and_start_simulation(sim_config,scenario)

  #sim_config = {'t_0': 0, 't_e': 20000, 't_step': 10, 'speed_factor': 100, 'anim_dt': 0.04, 'scale_factor': 1000, 'visualise': True}
  sim_config = {'t_0': 0, 't_e': 100, 't_step': 0.01, 'speed_factor': 1, 'anim_dt': 0.04, 'scale_factor': 150,'visualise': True}
  scenario = ScenarioAssignment1()
  sim.create_and_start_simulation(sim_config,scenario)



if __name__ == "__main__":
    main()
