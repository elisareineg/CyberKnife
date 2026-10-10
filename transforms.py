import numpy as np

from geometry import (
    DETECTOR_A,
    DETECTOR_B,
    PIXEL_SIZE,
    IMAGE_SIZE
)

from utils import frame_to_home, apply

# PART 1: FRAME TRANSFORMS


# MD <-> CK
#
#   - MD origin and CK origin coincide
#   - MD basis vectors and CK basis vectors are parallel
#
# Therefore the transformation is simply the identity matrix.
#

MD_TO_CK = np.eye(4)
CK_TO_MD = np.eye(4)


def md_to_ck(point_md):
    """
    Transform a point from Model (MD) coordinates
    to CyberKnife (CK) coordinates.
    """
    return apply(MD_TO_CK, point_md)


def ck_to_md(point_ck):
    """
    Transform a point from CyberKnife (CK) coordinates
    to Model (MD) coordinates.
    """
    return apply(CK_TO_MD, point_ck)


# Detector <-> CK"

def detector_to_ck_matrix(detector):
    """
    Construct the homogeneous transformation matrix
    from a detector frame to the CK frame.

    Detector axes:
        x-like axis = u
        y-like axis = v
        z-like axis = w

    Detector origin:
        detector centre
    """

    return frame_to_home(
        detector["center"],
        detector["u"],
        detector["v"],
        detector["w"]
    )


def ck_to_detector_matrix(detector):
    """
    Construct the inverse transform:
    CK frame -> detector frame.
    """

    return np.linalg.inv(detector_to_ck_matrix(detector))


# Precompute matrices for the two CyberKnife detectors.

DETECTOR_A_TO_CK = detector_to_ck_matrix(DETECTOR_A)
CK_TO_DETECTOR_A = ck_to_detector_matrix(DETECTOR_A)

DETECTOR_B_TO_CK = detector_to_ck_matrix(DETECTOR_B)
CK_TO_DETECTOR_B = ck_to_detector_matrix(DETECTOR_B)


def get_detector(view):
    """
    Return detector geometry for view 'A' or 'B'.
    """

    view = view.upper()

    if view == "A":
        return DETECTOR_A

    if view == "B":
        return DETECTOR_B

    raise ValueError("Detector view must be 'A' or 'B'.")


def ck_to_detector(point_ck, view):
    """
    Transform a CK-frame 3D point into detector-frame
    coordinates (u, v, w).

    Important:
    This is a coordinate transformation, NOT an X-ray
    projection.
    """

    detector = get_detector(view)

    F = ck_to_detector_matrix(detector)

    return apply(F, point_ck)


def detector_to_ck(point_detector, view):
    """
    Transform detector-frame coordinates (u, v, w)
    back into CK coordinates.
    """

    detector = get_detector(view)

    F = detector_to_ck_matrix(detector)

    return apply(F, point_detector)


# Detector <-> Image
#
# Detector size = 200 mm
# Pixel size    = 0.1 mm
#
# Therefore:
#
#       1 mm = 10 pixels
#
# The detector centre corresponds to the image centre:
#
#       (u, v) = (0, 0) mm
#
# becomes
#
#       (1000, 1000) pixels
#

def detector_to_image(point_detector):
    """
    Convert detector coordinates into image coordinates.

    Parameters
    ----------
    point_detector : array-like
        Detector coordinates (u, v, w).

    Returns
    -------
    np.ndarray
        Image coordinates (u_pixel, v_pixel).
    """

    point_detector = np.asarray(point_detector, dtype=float)

    u = point_detector[0]
    v = point_detector[1]

    centre = IMAGE_SIZE / 2.0

    image_u = u / PIXEL_SIZE + centre
    image_v = v / PIXEL_SIZE + centre

    return np.array([image_u, image_v])


def image_to_detector(point_image, w=0.0):
    """
    Convert image coordinates back into detector coordinates.

    The image represents the detector plane, so w normally equals 0.
    """

    point_image = np.asarray(point_image, dtype=float)

    image_u = point_image[0]
    image_v = point_image[1]

    centre = IMAGE_SIZE / 2.0

    u = (image_u - centre) * PIXEL_SIZE
    v = (image_v - centre) * PIXEL_SIZE

    return np.array([u, v, w])


# FRAME TRANSFORM TESTS

def check(name, actual, expected, tolerance=1e-8):

    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)

    passed = np.allclose(
        actual,
        expected,
        atol=tolerance
    )

    print(name)
    print("  predicted:", expected)
    print("  actual:   ", actual)
    print("  result:   ", "PASS" if passed else "FAIL")
    print()

    return passed


if __name__ == "__main__":

    print("\nPART 1 - FRAME TRANSFORM TESTS\n")

    # MD <-> CK
    #
    # Because MD and CK are aligned, every point should keep
    # exactly the same coordinates.

    print("MD <-> CK\n")

    md_points = [
        np.array([0.0, 0.0, 0.0]),
        np.array([10.0, -20.0, 30.0]),
        np.array([-19.0, 28.0, 11.0])
    ]

    for i, point in enumerate(md_points, start=1):

        check(
            f"MD -> CK point {i}",
            md_to_ck(point),
            point
        )

        check(
            f"CK -> MD point {i}",
            ck_to_md(point),
            point
        )

    # CK <-> Detector A/B
    #
    # Three easy-to-predict test points:
    #
    # 1. Detector centre must become (0,0,0).
    #
    # 2. X-ray source is SDD = 2000 mm along detector +w,
    #    therefore source must become (0,0,2000).
    #
    # 3. C + 10u + 20v must become (10,20,0).

    for view in ["A", "B"]:

        detector = get_detector(view)

        print(f"CK <-> Detector {view}\n")

        C = detector["center"]
        S = detector["source"]
        u = detector["u"]
        v = detector["v"]

        test_ck = C + 10.0 * u + 20.0 * v

        check(
            f"Detector {view} centre",
            ck_to_detector(C, view),
            [0.0, 0.0, 0.0]
        )

        check(
            f"Detector {view} source",
            ck_to_detector(S, view),
            [0.0, 0.0, 2000.0]
        )

        check(
            f"Detector {view} u/v point",
            ck_to_detector(test_ck, view),
            [10.0, 20.0, 0.0]
        )

        # Reverse transformations

        check(
            f"Detector {view} -> CK origin",
            detector_to_ck([0.0, 0.0, 0.0], view),
            C
        )

        check(
            f"Detector {view} -> CK source",
            detector_to_ck([0.0, 0.0, 2000.0], view),
            S
        )

        check(
            f"Detector {view} -> CK u/v point",
            detector_to_ck([10.0, 20.0, 0.0], view),
            test_ck
        )

    # Detector <-> Image
    #
    # Pixel size = 0.1 mm.
    #
    # (0,0) detector -> (1000,1000) image
    # (10,-20) mm    -> (1100,800)
    # (-50,50) mm    -> (500,1500)

    print("Detector <-> Image\n")

    detector_image_tests = [
        (
            np.array([0.0, 0.0, 0.0]),
            np.array([1000.0, 1000.0])
        ),
        (
            np.array([10.0, -20.0, 0.0]),
            np.array([1100.0, 800.0])
        ),
        (
            np.array([-50.0, 50.0, 0.0]),
            np.array([500.0, 1500.0])
        )
    ]

    for i, (detector_point, image_point) in enumerate(
        detector_image_tests,
        start=1
    ):

        check(
            f"Detector -> Image point {i}",
            detector_to_image(detector_point),
            image_point
        )

        check(
            f"Image -> Detector point {i}",
            image_to_detector(image_point),
            detector_point
        )

        
        
# author: Serhat