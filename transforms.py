import numpy as np

from geometry import (
    DETECTOR_A,
    DETECTOR_B,
    PIXEL_SIZE,
    IMAGE_SIZE
)

from utils import frame_to_home

# PART 1: FRAME TRANSFORMS


# MD <-> CK
#
#   - MD origin and CK origin coincide
#   - MD basis vectors and CK basis vectors are parallel
#
# Therefore the transformation is simply the identity matrix.
#

# Identity transformations
MD_TO_CK = np.eye(4)
CK_TO_MD = np.eye(4)


def md_to_ck(point_md):
    """
    Convert MD coordinates to CK coordinates.

    Since the frames are identical, preserve the coordinates
    and return an independent floating-point array.
    """
    return np.asarray(point_md, dtype=float).copy()

def ck_to_md(point_ck):
    """
    Convert CK coordinates to MD coordinates.

    The frames share an origin and orientation, so no coordinate
    changes are required.
    """
    return np.asarray(point_ck, dtype=float).copy()


# Detector <-> CK

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
    """Construct the inverse rigid transformation from CK to detector."""

    transform = detector_to_ck_matrix(detector)

    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    inverse = np.eye(4)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation

    return inverse


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
    Express a CK-frame point in the selected detector's coordinates.

    Uses projections onto the detector's orthonormal basis vectors.
    This changes coordinate frames, it does not perform X-ray projection.
    """
    detector = get_detector(view)

    point = np.asarray(point_ck, dtype=float)
    displacement = point - detector["center"]

    return np.array([
        np.dot(displacement, detector["u"]),
        np.dot(displacement, detector["v"]),
        np.dot(displacement, detector["w"])
    ])


def detector_to_ck(point_detector, view):
    """
    Convert detector-frame coordinates (u, v, w) into CK coordinates.

    Express the detector-frame displacement in the CK basis,
    then translate it to the detector centre.
    """
    detector = get_detector(view)

    point = np.asarray(point_detector, dtype=float)

    return (
        detector["center"]
        + point[0] * detector["u"]
        + point[1] * detector["v"]
        + point[2] * detector["w"]
    )


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
    Convert detector-frame coordinates (mm) into image pixels.

    The detector's (u, v) origin maps to the image centre.
    The perpendicular w coordinate is not included.
    """
    point = np.asarray(point_detector, dtype=float)

    image_center = IMAGE_SIZE / 2.0

    return point[:2] / PIXEL_SIZE + image_center


def image_to_detector(point_image, w=0.0):
    """
    Convert image pixel coordinates into detector-frame coordinates.

    The optional w coordinate specifies the perpendicular distance
    from the detector plane, defaulting to zero.
    """
    point = np.asarray(point_image, dtype=float)

    image_center = IMAGE_SIZE / 2.0

    uv = (point[:2] - image_center) * PIXEL_SIZE

    return np.array([uv[0], uv[1], w], dtype=float)
        
# authors: Akshay, Serhat