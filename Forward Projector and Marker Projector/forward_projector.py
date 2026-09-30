import numpy as np

# CyberKnife geometry constants

SAD = 1000.0       # source-to-axis distance [mm]
SDD = 2000.0       # source-to-detector distance [mm]

DETECTOR_SIZE = 200.0   # 200 mm x 200 mm
PIXEL_SIZE = 0.1        # mm / pixel
IMAGE_SIZE = int(DETECTOR_SIZE / PIXEL_SIZE)  # 2000 pixels


# Temporary detector geometry
#
# To be replaced with the frame-transform module produced.

def rotation_z(angle_deg):
    """3x3 rotation matrix about CK +z."""
    theta = np.deg2rad(angle_deg)

    return np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0]
    ])


def make_detector_geometry(angle_deg):
    """
    Construct temporary geometry for one CyberKnife detector view.

    Home position:
        source = (0, +1000, 0)
        detector centre = (0, -1000, 0)

    The whole source-detector pair is rotated about CK +z.
    """

    R = rotation_z(angle_deg)

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


# PART 3: FORWARD PROJECTOR

def project_point_to_detector(point_ck, detector):
    """
    Project a 3-D point in CK coordinates onto an X-ray detector.

    Parameters
    ----------
    point_ck : array-like, shape (3,)
        Point in CK coordinates [mm].

    detector : dict
        Detector geometry containing:
            source
            center
            u
            v
            w

    Returns
    -------
    detector_point : np.ndarray, shape (3,)
        Coordinates (u, v, w) in the detector frame.
        For a correctly projected point, w should be approximately zero.

    hit_ck : np.ndarray, shape (3,)
        Intersection point expressed in CK coordinates.
    """

    P = np.asarray(point_ck, dtype=float)

    S = detector["source"]
    C = detector["center"]
    u = detector["u"]
    v = detector["v"]
    w = detector["w"]

    # Ray from X-ray source through the CK point
    ray = P - S

    # Detector plane:
    #
    #     (X - C) dot w = 0
    #
    # Ray:
    #
    #     X(t) = S + t(P - S)
    #
    # Solve for t.
    denominator = np.dot(ray, w)

    if np.isclose(denominator, 0.0):
        raise ValueError("Projection ray is parallel to detector plane.")

    t = np.dot(C - S, w) / denominator

    hit_ck = S + t * ray

    # Convert CK hit point into detector coordinates
    relative = hit_ck - C

    detector_point = np.array([
        np.dot(relative, u),
        np.dot(relative, v),
        np.dot(relative, w)
    ])

    return detector_point, hit_ck


def forward_project(point_ck):
    """
    Project one CK point onto both CyberKnife detectors.

    Returns detector-frame coordinates for views A and B.
    """

    point_A, _ = project_point_to_detector(point_ck, DETECTOR_A)
    point_B, _ = project_point_to_detector(point_ck, DETECTOR_B)

    return {
        "A": point_A,
        "B": point_B
    }

# Detector, then, image conversion

def detector_to_image(detector_point):
    """
    Convert detector coordinates (u,v,w) in mm to image coordinates.

    Temporary convention:
        detector u = image vertical coordinate
        detector v = image horizontal coordinate

    Image centre = (1000, 1000)
    """

    u, v, _ = detector_point

    centre_pixel = IMAGE_SIZE / 2.0

    image_u = u / PIXEL_SIZE + centre_pixel
    image_v = v / PIXEL_SIZE + centre_pixel

    return np.array([image_u, image_v])


# Forward projector tests

if __name__ == "__main__":

    test_points = {
        "CK origin": np.array([0.0, 0.0, 0.0]),

        "50 mm along +z": np.array([0.0, 0.0, 50.0]),

        "off-axis point": np.array([10.0, 0.0, 0.0])
    }

    print("\nFORWARD PROJECTOR TESTS\n")

    for name, point in test_points.items():

        projections = forward_project(point)

        print(name)
        print("CK:", point)
        print("Detector A:", projections["A"])
        print("Detector B:", projections["B"])
        print()