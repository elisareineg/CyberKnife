import numpy as np

# reusable utility functions

# rot_z(theta)
def rot_z(deg):
    # 4x4 rotation about z 
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0, 0],
                     [s,  c, 0, 0],
                     [0,  0, 1, 0],
                     [0,  0, 0, 1]], dtype=float)

# normalize(v)
def normalize(v):
    # Return v scaled to unit length
    v = np.asarray(v, dtype=float)
    length = np.linalg.norm(v)

    if np.isclose(length, 0.0):
        raise ValueError("Cannot normalize a zero-length vector.")

    return v / length

# scale(s)
def scale(s):
    # Return a 4x4 homogeneous scaling matrix (uniform scale s on x, y, z)
    return np.diag([s, s, s, 1.0]).astype(float)

# translate(d)
def translate(d):
    # Return a 4x4 homogeneous translation matrix
    dx, dy, dz = d

    return np.array([
        [1.0, 0.0, 0.0, dx],
        [0.0, 1.0, 0.0, dy],
        [0.0, 0.0, 1.0, dz],
        [0.0, 0.0, 0.0, 1.0]
    ])

# frame_to_home(O, e1, e2, e3)
def frame_to_home(O, e1, e2, e3):
    # Build frame transform using axis directions as columns and origin as translation
    F = np.eye(4)

    F[:3, 0] = e1
    F[:3, 1] = e2
    F[:3, 2] = e3
    F[:3, 3] = O

    return F

# apply(F, p)
def apply(F, p):
    # Apply a 4x4 homogeneous transformation to a 3D point
    p_h = np.append(p, 1.0)
    result_h = F @ p_h

    return result_h[:3]

# line_from_points(A, B)

# intersect_line_plane(P, v, A, n)

# intersect_line_sphere(P, v, C, R)

# point_line_distance(A, P, v)

# symbolic_intersection(P1, v1, P2, v2)

# attenuate(I0, k, lengths, densities)

# author: akshay, elisa