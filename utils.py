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
    pass

# translate(d)


# frame_to_home(O, e1, e2, e3)

# apply(F, p)

# line_from_points(A, B)

# intersect_line_plane(P, v, A, n)

# intersect_line_sphere(P, v, C, R)

# point_line_distance(A, P, v)

# symbolic_intersection(P1, v1, P2, v2)

# attenuate(I0, k, lengths, densities)