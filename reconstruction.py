import os
import json

import numpy as np

from geometry import (
    DETECTOR_A,
    DETECTOR_B
)

from transforms import detector_to_ck

from marker_transforms import MARKERS_CK

from marker_projector import project_markers


# PART 7: MARKER RECONSTRUCTION
#
# A marker observed on Detector A defines one 3D ray:
#
#     Source A -> detector point A
#
# The same marker observed on Detector B defines another:
#
#     Source B -> detector point B
#
# With exact detector positions, the two lines intersect at the
# original 3D marker position.
#
# With localization error, the two lines may become skew and may
# no longer intersect exactly. In that case, the reconstructed point
# is taken as the midpoint between the closest points on the two lines.
#
# REM is the shortest distance between the two back-projection lines.


def normalize(vector):
    """
    Return a unit-length version of a vector.
    """

    vector = np.asarray(
        vector,
        dtype=float
    )

    length = np.linalg.norm(
        vector
    )

    if np.isclose(
        length,
        0.0
    ):
        raise ValueError(
            "Cannot normalize a zero-length vector."
        )

    return vector / length


def detector_point_to_ck_ray(
    detector_point,
    detector,
    view
):
    """
    Convert a point in detector coordinates into a back-projection
    ray in the CK frame.

    Parameters
    ----------
    detector_point : array-like
        Detector coordinates [u, v, w].

        For a point on the detector plane:
            w = 0

    detector : dict
        Detector geometry.

    view : str
        "A" or "B".

    Returns
    -------
    source : ndarray
        X-ray source position in CK coordinates.

    direction : ndarray
        Unit direction vector pointing from the source through the
        detector point.
    """

    detector_point = np.asarray(
        detector_point,
        dtype=float
    )

    point_ck = detector_to_ck(
        detector_point,
        view
    )

    source = np.asarray(
        detector["source"],
        dtype=float
    )

    direction = normalize(
        point_ck - source
    )

    return source, direction


def closest_points_between_lines(
    P1,
    v1,
    P2,
    v2
):
    """
    Compute the closest points between two 3D lines.

    Line 1:
        L1(t) = P1 + t*v1

    Line 2:
        L2(s) = P2 + s*v2

    For exact marker projections, the lines intersect and the two
    closest points should be the same.

    With image localization error, they may be skew. The closest
    pair of points then gives both the reconstructed position and REM.
    """

    P1 = np.asarray(
        P1,
        dtype=float
    )

    P2 = np.asarray(
        P2,
        dtype=float
    )

    v1 = np.asarray(
        v1,
        dtype=float
    )

    v2 = np.asarray(
        v2,
        dtype=float
    )

    w0 = P1 - P2

    a = np.dot(
        v1,
        v1
    )

    b = np.dot(
        v1,
        v2
    )

    c = np.dot(
        v2,
        v2
    )

    d = np.dot(
        v1,
        w0
    )

    e = np.dot(
        v2,
        w0
    )

    denominator = (
        a * c
        - b * b
    )

    if np.isclose(
        denominator,
        0.0
    ):
        raise ValueError(
            "Back-projection lines are parallel or nearly parallel."
        )

    t = (
        b * e
        - c * d
    ) / denominator

    s = (
        a * e
        - b * d
    ) / denominator

    point_1 = (
        P1
        + t * v1
    )

    point_2 = (
        P2
        + s * v2
    )

    return (
        point_1,
        point_2
    )


def reconstruct_point(
    detector_point_A,
    detector_point_B
):
    """
    Reconstruct one CK-frame 3D point from observations on
    Detector A and Detector B.

    Returns
    -------
    reconstructed : ndarray
        Reconstructed CK point.

    rem : float
        Residual Error Metric.

        REM is the shortest distance between the two back-projection
        lines.

    closest_A : ndarray
        Closest point on Detector A's back-projection line.

    closest_B : ndarray
        Closest point on Detector B's back-projection line.
    """

    source_A, direction_A = detector_point_to_ck_ray(
        detector_point_A,
        DETECTOR_A,
        "A"
    )

    source_B, direction_B = detector_point_to_ck_ray(
        detector_point_B,
        DETECTOR_B,
        "B"
    )

    closest_A, closest_B = closest_points_between_lines(
        source_A,
        direction_A,
        source_B,
        direction_B
    )

    reconstructed = (
        closest_A
        + closest_B
    ) / 2.0

    rem = np.linalg.norm(
        closest_A
        - closest_B
    )

    return (
        reconstructed,
        rem,
        closest_A,
        closest_B
    )


def reconstruct_markers():
    """
    Test reconstruction using the exact detector projections of
    M1, M2 and M3.

    The detector locations are produced by the forward projector,
    so reconstructing them should recover the original CK marker
    locations with nearly zero numerical error.
    """

    projections = project_markers(
        MARKERS_CK
    )

    results = {}

    for marker_name in MARKERS_CK:

        true_ck = np.asarray(
            MARKERS_CK[
                marker_name
            ],
            dtype=float
        )

        detector_A = projections[
            marker_name
        ][
            "Detector_A"
        ]

        detector_B = projections[
            marker_name
        ][
            "Detector_B"
        ]

        reconstructed, rem, closest_A, closest_B = reconstruct_point(
            detector_A,
            detector_B
        )

        reconstruction_error = np.linalg.norm(
            reconstructed
            - true_ck
        )

        results[
            marker_name
        ] = {
            "true_ck": true_ck,
            "detector_A": detector_A,
            "detector_B": detector_B,
            "reconstructed_ck": reconstructed,
            "reconstruction_error_mm":
                reconstruction_error,
            "REM_mm": rem,
            "closest_A": closest_A,
            "closest_B": closest_B
        }

    return results


def save_results(
    results,
    filename="data/marker_reconstruction.json"
):
    """
    Save reconstruction results for later assignment sections.
    """

    os.makedirs(
        os.path.dirname(
            filename
        ),
        exist_ok=True
    )

    output = {}

    for marker_name, values in results.items():

        output[
            marker_name
        ] = {}

        for key, value in values.items():

            if isinstance(
                value,
                np.ndarray
            ):

                output[
                    marker_name
                ][
                    key
                ] = value.tolist()

            else:

                output[
                    marker_name
                ][
                    key
                ] = float(
                    value
                )

    with open(
        filename,
        "w"
    ) as f:

        json.dump(
            output,
            f,
            indent=4
        )


def print_results(
    results
):
    """
    Print true and reconstructed marker positions along with
    reconstruction error and REM.
    """

    print(
        "\nPART 7 - MARKER RECONSTRUCTION\n"
    )

    for marker_name, result in results.items():

        print(
            marker_name
        )

        print(
            "  True CK:",
            np.round(
                result[
                    "true_ck"
                ],
                10
            )
        )

        print(
            "  Reconstructed CK:",
            np.round(
                result[
                    "reconstructed_ck"
                ],
                10
            )
        )

        print(
            "  Reconstruction error:",
            f"{result['reconstruction_error_mm']:.12e}",
            "mm"
        )

        print(
            "  REM:",
            f"{result['REM_mm']:.12e}",
            "mm"
        )

        print()


if __name__ == "__main__":

    results = reconstruct_markers()

    print_results(
        results
    )

    save_results(
        results
    )

    print(
        "Saved data/marker_reconstruction.json"
    )

    
# author: Serhat