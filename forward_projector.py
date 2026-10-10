import numpy as np

from geometry import (
    DETECTOR_A,
    DETECTOR_B,
    PIXEL_SIZE,
    IMAGE_SIZE
)


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
        Coordinates [u, v, w] in the detector frame.
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
    # (X - C) dot w = 0
    #
    # Ray:
    # X(t) = S + t(P - S)

    denominator = np.dot(ray, w)

    if np.isclose(denominator, 0.0):
        raise ValueError(
            "Projection ray is parallel to detector plane."
        )

    # Solve for intersection parameter t
    t = np.dot(C - S, w) / denominator

    # Intersection point in CK coordinates
    hit_ck = S + t * ray

    # Convert CK intersection point to detector-frame coordinates
    relative = hit_ck - C

    detector_point = np.array([
        np.dot(relative, u),
        np.dot(relative, v),
        np.dot(relative, w)
    ])

    return detector_point, hit_ck


def forward_project(point_ck):
    """
    Project one CK-frame point onto both CyberKnife detectors.

    Returns detector-frame coordinates for detector A and detector B.
    """

    point_A, _ = project_point_to_detector(
        point_ck,
        DETECTOR_A
    )

    point_B, _ = project_point_to_detector(
        point_ck,
        DETECTOR_B
    )

    return {
        "A": point_A,
        "B": point_B
    }


def detector_to_image(detector_point):
    """
    Convert detector coordinates [u, v, w] in mm
    to image coordinates [u, v] in pixels.

    The detector centre corresponds to the image centre.

    IMAGE_SIZE = 2000 pixels
    PIXEL_SIZE = 0.1 mm

    Therefore detector (0, 0) corresponds to
    image pixel (1000, 1000).
    """

    u, v, _ = detector_point

    centre_pixel = IMAGE_SIZE / 2.0

    image_u = u / PIXEL_SIZE + centre_pixel
    image_v = v / PIXEL_SIZE + centre_pixel

    return np.array([
        image_u,
        image_v
    ])


# Forward projector tests

if __name__ == "__main__":

    test_points = {
        "CK origin": np.array([
            0.0,
            0.0,
            0.0
        ]),

        "50 mm along +z": np.array([
            0.0,
            0.0,
            50.0
        ]),

        "Off-axis point": np.array([
            10.0,
            0.0,
            0.0
        ])
    }

    print("\nFORWARD PROJECTOR TESTS\n")

    for name, point in test_points.items():

        projections = forward_project(point)

        image_A = detector_to_image(
            projections["A"]
        )

        image_B = detector_to_image(
            projections["B"]
        )

        print(name)

        print("  CK")
        print("   ", point)

        print("  Detector A [u,v,w] mm")
        print("   ", projections["A"])

        print("  Image A [u,v] pixels")
        print("   ", image_A)

        print("  Detector B [u,v,w] mm")
        print("   ", projections["B"])

        print("  Image B [u,v] pixels")
        print("   ", image_B)

        print()

        
# author: Serhat