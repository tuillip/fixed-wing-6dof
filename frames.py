import numpy as np

"""
frames.py -- coordinate frame transformations for the 6-DOF simulator.
 
Reference: Beard & McLain, "Small Unmanned Aircraft: Theory and Practice", Ch. 2.
 
FRAME REFERENCE TABLE
---------------------
  i   inertial / NED     x north, y east, z down.  Fixed to the earth.
  v   vehicle            axes parallel to inertial, origin at the aircraft CG.
  v1  after yaw   (psi)    rotation about the vehicle z-axis
  v2  after pitch (theta)  rotation about the v1 y-axis
  b   body            after roll  (phi)    rotation about the v2 x-axis
  s   stability       body rotated by angle of attack alpha about the body y-axis
  w   wind            stability rotated by sideslip beta about the stability z-axis
 
  Vehicle -> body is a 3-2-1 Euler sequence: yaw, then pitch, then roll.
 
  Every rotation matrix here is ORTHOGONAL, so R_inverse == R.T
  You never need to write an inverse -- transpose the forward one.
 
CONVENTION WARNING
------------------
  Beard rotates the FRAME, not the vector. That decides where the minus
  signs go, and it is the single most common place to get this wrong.
  Copy the matrices from the book rather than reconstructing them from memory.
 
  ONE SOURCE OF TRUTH: the body frame is canonical. Never store the same
  physical vector in two frames -- convert on demand with these functions.
"""

def rot_x(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
    
def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    
def rot_z(psi):
    c,s = np.cos(psi), np.sin(psi)
    return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

def vehicle_to_body(phi, theta, psi):
    return rot_x(phi) @ (rot_y(theta) @ rot_z(psi))

def body_to_stability(alpha):
    return rot_y(-1*alpha)

def stability_to_wind(beta):
    return rot_z(beta)

def body_to_wind(alpha, beta):
    return rot_z(beta) @ rot_y(-1*alpha)

def airdata_from_body_velocity(u, v, w):
    return (np.sqrt(u**2+v**2+w**2), np.arctan2(w/u), np.arctan2(v/(np.sqrt(u**2+w**2))))

def body_velocity_from_airdata(Va, alpha, beta):
    return (Va*np.cos(alpha)*np.cos(beta), Va*np.sin(beta), Va*np.sin(alpha)*np.cos(beta))

TOL = 1e-10
rng = np.random.default_rng(0)

def test_elementary_rotations():
    a = 0.7   # any nonzero angle

    # 1. A rotation about an axis leaves that axis alone.
    assert np.allclose(rot_x(a) @ [1, 0, 0], [1, 0, 0])
    assert np.allclose(rot_y(a) @ [0, 1, 0], [0, 1, 0])
    assert np.allclose(rot_z(a) @ [0, 0, 1], [0, 0, 1])

    # 2. Rotation matrices are orthogonal: R @ R.T = I.
    assert np.allclose(rot_x(a) @ rot_x(a).T, np.eye(3))
    assert np.allclose(rot_y(a) @ rot_y(a).T, np.eye(3))
    assert np.allclose(rot_z(a) @ rot_z(a).T, np.eye(3))

    # 3. Quarter turn about z: a vector along +x reads as -y in the new frame.
    assert np.allclose(rot_z(np.pi/2) @ [1, 0, 0], [0, -1, 0])

    print("elementary rotations OK")

def test_vehicle_to_body():
    # testing code
    return 0

def test_wind_frames():
    # testing code
    return 0

def test_airdata_roundtrip():
    # testing code
    return 0

if __name__ == "__main__":
    test_elementary_rotations()
