# shared core: transforms, projector, reconstruction

import numpy as np

from utils import rot_z

# CyberKnife geometry constants

SAD = 1000.0       # source-to-axis distance [mm]
SDD = 2000.0       # source-to-detector distance [mm]

DETECTOR_SIZE = 200.0   # 200 mm x 200 mm
PIXEL_SIZE = 0.1        # mm / pixel
IMAGE_SIZE = int(DETECTOR_SIZE / PIXEL_SIZE)  # 2000 pixels


# Frame transforms

# Detector geometry
#
# Shared by the projectors; to be expressed as CK <-> Detector frame transforms.

def make_detector_geometry(angle_deg):
    """
    Construct geometry for one CyberKnife detector view.

    Home position:
        source = (0, +1000, 0)
        detector centre = (0, -1000, 0)

    The whole source-detector pair is rotated about CK +z.
    """

    R = rot_z(angle_deg)[:3, :3]

    source_home = np.array([0.0, +SAD, 0.0])
    detector_home = np.array([0.0, -(SDD - SAD), 0.0])

    source = R @ source_home
    center = R @ detector_home

    # Detector axes
    # +u is always CK +z
    u_axis = np.array([0.0, 0.0, 1.0])

    # At home, detector +v is CK +x
    v_axis = R @ np.array([1.0, 0.0, 0.0])

    # +w points toward the X-ray source
    w_axis = R @ np.array([0.0, 1.0, 0.0])

    return {
        "source": source,
        "center": center,
        "u": u_axis,
        "v": v_axis,
        "w": w_axis
    }


DETECTOR_A = make_detector_geometry(+45.0)
DETECTOR_B = make_detector_geometry(-45.0)


# Forward projector
    # needs line_from_points(A, B) and intersect_line_plane(P, v, A, n)
    # also needs detector geometry view (frame to home from Part 1)



# author: Serhat