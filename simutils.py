import datetime as dt
from typing import Callable

import math
import numpy as np
import orbit_lib as ol
from vispy.scene import MatrixTransform as Mat4
from vispy.util.quaternion import Quaternion as Quat

class Error(Exception):
    pass

class InvalidConstruction(Error):
    def __init__(self,message):
        self.message=message

class Quaternion:
    def __init__(self, arg1 = None, arg2 = None):
        if arg1 is None and arg2 is None:
            self.q = np.array([1,0,0,0])
        elif arg1 is not None and arg2 is None:
            if type(arg1) is Quaternion:
                self.q = np.array(arg1.q)
            elif len(arg1) == 4:
                self.q = np.array(arg1)
            elif len(arg1) == 3:
                self.q = np.array([0,*arg1])
            else:
                raise InvalidConstruction("Wrong initialization, expects one of:\narg1=None,arg2=None\narg1=Quaternion,arg2=None\narg1=list[4],arg2=None\narg1=list[3],arg2=None\narg1=float,arg2=list[3]\n")
        elif arg1 is not None and arg2 is not None:
            if len(arg2) == 3:
                mag = np.sqrt(arg2[0]**2.0+arg2[1]**2.0+arg2[2]**2.0)
                self.q = np.array([np.cos(arg1/2.0),*(np.sin(arg1/2.0)/mag*np.array(arg2))])
            else:
                raise InvalidConstruction("Wrong initialization, expects one of:\narg1=None,arg2=None\narg1=Quaternion,arg2=None\narg1=list[4],arg2=None\narg1=list[3],arg2=None\narg1=float,arg2=list[3]\n")
        else:
            raise InvalidConstruction("Wrong initialization, expects one of:\narg1=None,arg2=None\narg1=Quaternion,arg2=None\narg1=list[4],arg2=None\narg1=list[3],arg2=None\narg1=float,arg2=list[3]\n")
    
    def __len__(self):
        return len(self.q)
        
    def __repr__(self):
        return "Quaternion: [{}]".format(",".join([str(x) for x in self.q]))
    
    def __getitem__(self,index):
        if type(index) == slice:
            if index.stop < index.start:
                raise IndexError("starting index should be smaller than ending index")
            elif index.start in range(0,len(self)+1) and index.stop in range(0,len(self)+1):
                return np.array([self[i] for i in range(index.start,index.stop+1)])
            else:
                raise IndexError("Indexes out of bounds")
        else:
            if index > 3:
                raise IndexError("Index out of bounds")
            else:
                return self.q[index]

    def __add__(self,other):
        return Quaternion(self.q+other.q)
    
    def __sub__(self,other):
        return Quaternion(self.q-other.q)
    
    def __mul__(self,other):
        return Quaternion(self.q*other)
            
    def __rmul__(self,other):
        return self*other
    
    def __truediv__(self,other):
        return 1/other*self

    def __matmul__(self,other):
        return Quaternion([self[0]*other[0]-np.dot(self[1:3],other[1:3]), *(self[0]*other[1:3]+other[0]*self[1:3]+np.cross(self[1:3],other[1:3]))])
    
    def inverted(self):
        mag = self.magnitude()
        if mag < 1e-9:
            raise IndexError("Magnitude is zero")
        return 1.0/mag**2.0*self.conjugated()
    
    def conjugated(self):
        return Quaternion([self[0],*(-self[1:3])])

    def normalized(self):
        mag = self.magnitude()
        return Quaternion(self.q/mag)
    
    def invert(self):
        mag = self.magnitude()
        if mag < 1e-9:
            raise IndexError("Magnitude is zero")
        self.q /= mag**2.0
    
    def conjugate(self):
        self.q = np.array([self[0],*(-self[1:3])])
    
    def normalize(self):
        mag = self.magnitude()
        if mag < 1e-9:
            raise IndexError("Magnitude is zero")
        self.q = self.q/mag
        
    def magnitude(self):
        return np.linalg.norm(self.q)

    def rotate(self, u):
        v = self@Quaternion(u)@self.conjugated()
        return v[1:3]

def read_TLE_file(file_name,satellite_name=''):
  def validate_entry(Name,line1,line2):
    if not Name[0].isalpha():
      return False
    if not line1[0].startswith("1") or not len(line1) == 9:
      return False
    if not line2[0].startswith("2") or not len(line2) == 8:
      return False
    return True

  tle_data = []
  with open(file_name) as f:
    file_contents = f.readlines()
  if len(file_contents) < 3:
    print("Error reading file\nRequired format is:\nAAAAAAAAAAAAAAAAAAAAAAAA\n1 NNNNNU NNNNNAAA NNNNN.NNNNNNNN +.NNNNNNNN +NNNNN-N +NNNNN-N N NNNNN\n2 NNNNN NNN.NNNN NNN.NNNN NNNNNNN NNN.NNNN NNN.NNNN NN.NNNNNNNNNNNNNN\nfor each entry")
    return tle_data

  for i in range(0,len(file_contents),3):
    if satellite_name in file_contents[i]:
      Name = file_contents[i].strip()
      line1 = file_contents[i+1].strip().split()
      line2 = file_contents[i+2].strip().split()
      if validate_entry(Name,line1,line2):
        epoch = float(line1[3])
        e = float("0."+line2[4])
        rev = float(line2[7])
        Me = float(line2[6])
        i = float(line2[2])
        O = float(line2[3])
        w = float(line2[5])
        tle_data.append((Name,epoch,e,rev,Me,i,O,w))
      else:
        print("Error reading entry:\n{}{}{}".format(file_contents[i],file_contents[i+1],file_contents[i+2]))
        break
  return tle_data

def read_obj(fname):
    verts = []
    vcols = []
    faces = []
    with open(fname,'r') as f:
        for line in f:
            if line.startswith('v '):
                d = [float(x) for x in line.split(' ')[1:]]
                verts.append(d[0:3])
                if len(d) > 3:
                    vcols.append(d[3:])
            elif line.startswith('f '):
                faces.append([int(x.split('/')[0])-1 for x in line.split(' ')[1:]])
            else:
                pass
    return np.array(verts),np.array(vcols),np.array(faces)

def rotscaleloc_to_vispy(pos=None,quat=None,Rot=None,Eul=None,scale=None):
    if quat is not None:
        q = Quat(w=quat[0],x=quat[1],y=quat[2],z=quat[3])
        H = Mat4(q.conjugate().get_matrix())
    elif Rot is not None:
        p = np.array([[0,0,0]]).T
        HT = np.vstack(((np.hstack((Rot,p)),np.array([[0,0,0,1]]))))
        H = Mat4(HT.T)
    elif Eul is not None:
        q = Quat.create_from_euler_angles(Eul[2],Eul[1],Eul[0])
        H = Mat4(q.conjugate().get_matrix())
    else:
        H = Mat4()
    if scale is not None:
        H.scale((scale,scale,scale))
    if pos is not None:
        H.translate(pos)
    return H

def H_to_Rp(H):
    return H.matrix[:3,:3].T,H.matrix[-1][:3]

def log_pos(name,pos,path='data/'):
    #file_name = path + name + '_' + dt.datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + '.txt'
    file_name = path + name + '.txt'

    print("logged: "+file_name)

    open(file_name, 'a').close()
    np.savetxt(file_name,pos)

    return file_name


###################################
# Assignment 3 | Algorithms       #
###################################

# f(t, x)
def two_body(t: float, x: np.ndarray, ae: np.ndarray=None,u: float=ol.mu) -> np.ndarray:
    """
    Compute the time derivative of the state vector for the classical two-body problem.

    The state vector x contains both the position and velocity vectors,
    formatted as: [rx, ry, rz, vx, vy, vz].

    :param t: Current time [s]
    :param x: State vector containing both position vector and velocity vector [km | km/s]
    :param ae: External acceleration vector (default: 0) [km/s**2]
    :param u: Standard gravitational parameter (default: Earth's μ) [km**3/s**2]
    :return: State vector with the time derivative of x [km/s | km/s**2]
    """
    ri = x[:3] # Get position vector
    vi = x[3:] # Get velocity vector

    r = np.linalg.norm(ri).astype(float) # Get distance
    ai = -u/r**3 * ri + (np.zeros(3) if ae is None else ae) # Get acceleration vector + external acceleration
    return np.concatenate([vi, ai])

# Numeric solver functions
def step_euler(h: float, t_curr: float, x_curr: np.ndarray, f: Callable[[float, np.ndarray], np.ndarray]) -> np.ndarray:
    """
    Performs one step of the explicit Euler method for a first-order ODE.

    The state vector x contains both the position and velocity vectors,
    formatted as: [rx, ry, rz, vx, vy, vz].

    :param h: Time step size [s]
    :param t_curr: Current time [s]
    :param x_curr: State vector containing position [km] and velocity [km/s]
    :param f: Function f(t, x) returning dx/dt
    :return: State vector at time t + h [km | km/s]
    """
    return x_curr + h * f(t_curr, x_curr)

def step_leapfrog(h: float, t_curr: float, x_curr: np.ndarray, f: Callable[[float, np.ndarray], np.ndarray]) -> np.ndarray:
    """
    Performs one step of the Leapfrog method for a first-order ODE.

    The state vector x contains both the position and velocity vectors,
    formatted as: [rx, ry, rz, vx, vy, vz].

    :param h: Time step size [s]
    :param t_curr: Current time [s]
    :param x_curr: State vector containing position [km] and velocity [km/s]
    :param f: Function f(t, x) returning dx/dt
    :return: State vector at time t + h [km | km/s]
    """
    # Current values
    dx = f(t_curr, x_curr) # [vx, vy, vz, ax, ay, az]
    v_half = dx[:3] + 0.5 * dx[3:] * h

    # New values
    r_new  = x_curr[:3] + v_half * h
    v_new = 2 * v_half - x_curr[3:] # TODO: Ask about recomputing acceleration

    return np.concatenate([r_new, v_new])

def step_verlet(h: float, t_curr: float, x_curr: np.ndarray, x_prev: np.ndarray, f: Callable[[float, np.ndarray], np.ndarray]) -> np.ndarray:
    """
    Performs one step of the Verlet method for a first-order ODE.

    The state vector x contains both the position and velocity vectors,
    formatted as: [rx, ry, rz, vx, vy, vz].

    Velocity is reconstructed for post-processing purposes.

    :param h: Time step size [s]
    :param t_curr: Current time [s]
    :param x_curr: Current state vector containing position [km] and velocity [km/s]
    :param x_prev: Previous state vector containing position [km] and velocity [km/s]
    :param f: Function f(t, x) returning dx/dt
    :return: State vector at time t + h [km | km/s]
    """
    # Could use r_next = r + v * h + 0.5 * a * h**2
    # but due to it being identical to leapfrog its used instead.
    if x_prev is None: # First step
        return step_leapfrog(h, t_curr, x_curr, f) # This also updates velocity

    # Extract positions
    r_curr = x_curr[:3]
    r_prev = x_prev[:3]

    # Get acceleration from f
    a_curr = f(t_curr, x_curr)[3:]

    # Verlet position update
    r_next = 2 * r_curr - r_prev + a_curr * h ** 2

    # AI Slop!!! No idea of its accuracy but velocity is optional in verlet
    v_next = (r_next - r_prev) / (2 * h)
    return np.concatenate([r_next, v_next])

def step_RK4(h: float, t_curr: float, x_curr: np.ndarray, f: Callable[..., np.ndarray], ae: np.ndarray=None) -> np.ndarray:
    """
    Performs one step of the Runge-Kutta method for a first-order ODE.

    The state vector x contains both the position and velocity vectors,
    formatted as: [rx, ry, rz, vx, vy, vz].

    :param h: Time step size [s]
    :param t_curr: Current time [s]
    :param x_curr: Current state vector containing position [km] and velocity [km/s]
    :param f: Function f(t, x) returning dx/dt
    :param ae: External acceleration vector (default: 0) [km/s**2]
    :return: State vector at time t + h [km | km/s]
    """
    t1 = t_curr
    t2 = t3 = t_curr + 0.5 * h
    t4 = t_curr + h

    x1 = x_curr
    x2 = x_curr + 0.5 * h * f(t1, x1, ae=ae)
    x3 = x_curr + 0.5 * h * f(t2, x2, ae=ae)
    x4 = x_curr + h * f(t3, x3, ae=ae)

    f1 = f(t1, x1, ae=ae)
    f2 = f(t2, x2, ae=ae)
    f3 = f(t3, x3, ae=ae)
    f4 = f(t4, x4, ae=ae)

    return x_curr + h / 6 * (f1 + 2 * f2 + 2 * f3 + f4)


###################################
# Assignment 4 | Algorithms       #
###################################

def quaternion_to_dcm(q: Quaternion):
    """
    Converts a unit quaternion to a 3x3 rotation matrix (DCM).

    The quaternion should be normalized to produce a valid rotation matrix.

    :param q: Unit quaternion representing the rotation
    :return: 3x3 rotation matrix
    """
    # Unpack variables
    w, x, y, z = q

    # Temp vars
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz     = w * x, w * y, w * z
    xy, xz, yz     = x * y, x * z, y * z

    return np.array([
        [ww + xx - yy - zz, 2 * (xy + wz), 2 * (xz - wy)],
        [2 * (xy - wz), ww - xx + yy - zz, 2 * (yz + wx)],
        [2 * (xz + wy), 2 * (yz - wx), ww - xx - yy + zz]
    ])

def axis_angle_to_dcm(u: np.ndarray, theta: float):
    """
    Converts axis-angle representation to a 3x3 rotation matrix (DCM).

    :param u: Unit vector representing the rotation axis
    :param theta: Rotation angle [rad]
    :return: 3x3 rotation matrix
    """
    norm = np.linalg.norm(u)
    if norm == 0: # Handles invalid values
        raise ValueError("Rotation axis 'u' must be non-zero!")

    u = u / norm # Ensures that u is a unit vector

    I = np.eye(3)
    su = np.array([
        [    0, -u[2],  u[1]],
        [ u[2],    0 , -u[0]],
        [-u[1],  u[0],     0]
    ])

    return I + math.sin(theta) * su + (1 - math.cos(theta)) * su @ su

def dcm_to_quaternion(R: np.ndarray): # Shepperd’s algorithm
    """

    :param R:
    :return:
    """
    q = np.zeros(4)
    trR = np.linalg.trace(R)

    if trR > 0:
        q[0] = 0.5 * math.sqrt(1 + trR)
        _q = 4 * q[0]  # Temp variable for calculations

        q[1] = (R[1, 2] - R[2, 1]) / _q
        q[2] = (R[2, 0] - R[0, 2]) / _q
        q[3] = (R[0, 1] - R[1, 0]) / _q

    else:
        D = R.diagonal()
        i, j, k = np.roll(np.arange(3), -np.argmax(D))
        q[i + 1] = 0.5 * math.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        _q = 4 * q[i + 1]  # Temp variable for calculations

        q[j + 1] = (R[i, j] + R[j, i]) / _q
        q[k + 1] = (R[i, k] + R[k, i]) / _q
        q[0]     = (R[j, k] - R[k, j]) / _q

    return Quaternion(np.sign(q[0]) * q).conjugated()

def euler_to_quaternion(roll: float, pitch: float, yaw: float): # Function not needed! The quaternion_from_roll_pitch_yaw_sequence from orbit library instead
    """
    Return the quaternion for a roll-pitch-yaw (RPY) sequence.

    Rotations are applied in the following order:
    - Roll  (rotation about x-axis)
    - Pitch (rotation about y-axis)
    - Yaw   (rotation about z-axis)

    :param roll: Roll angle [radians]
    :param pitch: Pitch angle [radians]
    :param yaw: Yaw angle [radians]
    :return: Quaternion representing the rotation
    """
    return ol.quaternion_from_roll_pitch_yaw_sequence(roll, pitch, yaw)

def quaternion_to_euler(q: Quaternion):

    # Unpack variables
    w, x, y, z = q

    # Temp vars
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    roll  = math.atan2(2 * (wx + yz), ww + zz - xx - yy)
    pitch = math.asin(2 * (wy - xz))
    yaw   = math.atan2(2 * (wz + xy), ww + xx - yy - zz)
    return roll, pitch, yaw

def dcm_to_euler(R: np.ndarray):

    roll  = math.atan2(R[2, 1], R[2, 2])
    pitch = math.asin(-R[2, 0])
    yaw   = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def euler_to_dcm(roll: float, pitch: float, yaw: float): # Function not needed! The rotation_matrix_from_roll_pitch_yaw_sequence from orbit library instead
    return ol.rotation_matrix_from_roll_pitch_yaw_sequence(roll, pitch, yaw)