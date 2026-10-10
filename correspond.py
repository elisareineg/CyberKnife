import itertools
import json
import os

import numpy as np

from scipy.optimize import linear_sum_assignment

from geometry import (
    DETECTOR_A,
    DETECTOR_B
)

from transforms import detector_to_ck

from forward_projector import project_point_to_detector

from marker_transforms import MARKERS_CK

from marker_projector import project_markers

from reconstruction import reconstruct_point


# PART 8: MARKER CORRESPONDENCES
#
# The three markers are identical, so when they are localized in
# Detector A and Detector B we do not automatically know which point
# in A corresponds to which point in B.
#
# This module solves that correspondence problem in two different ways.
#
# Method 1:
#     Try every possible pairing and choose the permutation with the
#     smallest total reconstruction residual error (REM).
#
# Method 2:
#     Use epipolar geometry. A point in Detector A defines an
#     epipolar line in Detector B. The correct corresponding point in
#     B should lie very close to this line.


def observation_on_detector(detector_point):
    """
    Convert an observed detector position to [u, v, 0].

    Small numerical w values from the forward projector are discarded
    because observations lie on the detector plane.
    """

    detector_point = np.asarray(
        detector_point,
        dtype=float
    )

    return np.array([
        detector_point[0],
        detector_point[1],
        0.0
    ])


def correspondence_by_rem(
    points_A,
    points_B
):
    """
    Method 1: resolve correspondence by exhaustive REM search.

    For three markers there are only:

        3! = 6

    possible one-to-one correspondences.

    For each permutation, every A/B pair is reconstructed in 3D.
    The REM values of the three reconstructions are added together.

    The permutation with the smallest total REM is selected.

    Returns
    -------
    best_permutation : tuple
        For each point i in A, best_permutation[i] gives the index
        of its corresponding point in B.

    best_score : float
        Sum of the three REM values.

    pair_results : list
        Reconstruction details for the selected correspondence.
    """

    if len(points_A) != len(points_B):
        raise ValueError(
            "The two detector views must contain the same "
            "number of marker observations."
        )

    number_of_markers = len(
        points_A
    )

    best_permutation = None
    best_score = np.inf
    best_pair_results = None

    for permutation in itertools.permutations(
        range(number_of_markers)
    ):

        total_rem = 0.0
        current_results = []

        for index_A, index_B in enumerate(
            permutation
        ):

            point_A = observation_on_detector(
                points_A[index_A]
            )

            point_B = observation_on_detector(
                points_B[index_B]
            )

            reconstructed, rem, closest_A, closest_B = (
                reconstruct_point(
                    point_A,
                    point_B
                )
            )

            total_rem += rem

            current_results.append(
                {
                    "index_A": index_A,
                    "index_B": index_B,
                    "reconstructed": reconstructed,
                    "REM": rem
                }
            )

        if total_rem < best_score:

            best_score = total_rem
            best_permutation = permutation
            best_pair_results = current_results

    return (
        best_permutation,
        best_score,
        best_pair_results
    )


def epipolar_line_in_B(
    detector_point_A
):
    """
    Construct the epipolar line in Detector B produced by one
    observation in Detector A.

    The Detector A observation defines a back-projection ray:

        Source A -> observed point A

    Any 3D point on this ray must project somewhere onto one common
    line in Detector B. That line is the epipolar line.

    Two different 3D points on the A back-projection ray are chosen.
    Both are projected onto Detector B. Their two B projections define
    the epipolar line.
    """

    detector_point_A = observation_on_detector(
        detector_point_A
    )

    point_A_ck = detector_to_ck(
        detector_point_A,
        "A"
    )

    source_A = np.asarray(
        DETECTOR_A["source"],
        dtype=float
    )

    direction = (
        point_A_ck
        - source_A
    )

    direction = (
        direction
        / np.linalg.norm(direction)
    )

    # Select two points along the same A back-projection ray.
    # These lie around the treatment region rather than directly at
    # either endpoint of the ray.
    point_1_ck = (
        source_A
        + 900.0 * direction
    )

    point_2_ck = (
        source_A
        + 1100.0 * direction
    )

    projection_1, _ = project_point_to_detector(
        point_1_ck,
        DETECTOR_B
    )

    projection_2, _ = project_point_to_detector(
        point_2_ck,
        DETECTOR_B
    )

    return (
        projection_1[:2],
        projection_2[:2]
    )


def point_to_line_distance_2d(
    point,
    line_point_1,
    line_point_2
):
    """
    Calculate perpendicular distance from a 2D point to a 2D line.

    The line is defined by two points.
    """

    point = np.asarray(
        point,
        dtype=float
    )

    line_point_1 = np.asarray(
        line_point_1,
        dtype=float
    )

    line_point_2 = np.asarray(
        line_point_2,
        dtype=float
    )

    direction = (
        line_point_2
        - line_point_1
    )

    length = np.linalg.norm(
        direction
    )

    if np.isclose(
        length,
        0.0
    ):
        raise ValueError(
            "Cannot define an epipolar line from two identical points."
        )

    relative = (
        point
        - line_point_1
    )

    cross_value = (
        direction[0] * relative[1]
        - direction[1] * relative[0]
    )

    return abs(
        cross_value
    ) / length


def epipolar_cost_matrix(
    points_A,
    points_B
):
    """
    Build a matrix of epipolar distances.

    Entry [i, j] is the perpendicular distance from marker j in
    Detector B to the epipolar line generated by marker i in
    Detector A.

    Correct correspondences should have distances close to zero.
    """

    number_A = len(
        points_A
    )

    number_B = len(
        points_B
    )

    cost = np.zeros(
        (
            number_A,
            number_B
        ),
        dtype=float
    )

    for index_A, point_A in enumerate(
        points_A
    ):

        line_1, line_2 = epipolar_line_in_B(
            point_A
        )

        for index_B, point_B in enumerate(
            points_B
        ):

            point_B = observation_on_detector(
                point_B
            )

            cost[
                index_A,
                index_B
            ] = point_to_line_distance_2d(
                point_B[:2],
                line_1,
                line_2
            )

    return cost


def correspondence_by_epipolar(
    points_A,
    points_B
):
    """
    Method 2: resolve correspondence using epipolar geometry.

    A cost matrix is created from point-to-epipolar-line distances.

    The Hungarian assignment algorithm then chooses the one-to-one
    correspondence with the smallest total epipolar distance.

    Returns
    -------
    permutation : tuple
        For each A index, gives its matched B index.

    total_cost : float
        Total epipolar distance for the selected assignment.

    cost_matrix : ndarray
        Full pairwise epipolar-distance matrix.
    """

    if len(points_A) != len(points_B):
        raise ValueError(
            "The two detector views must contain the same "
            "number of marker observations."
        )

    cost_matrix = epipolar_cost_matrix(
        points_A,
        points_B
    )

    rows, columns = linear_sum_assignment(
        cost_matrix
    )

    permutation = [
        None
    ] * len(points_A)

    for row, column in zip(
        rows,
        columns
    ):

        permutation[
            row
        ] = int(
            column
        )

    total_cost = float(
        cost_matrix[
            rows,
            columns
        ].sum()
    )

    return (
        tuple(permutation),
        total_cost,
        cost_matrix
    )


def get_exact_projected_markers():
    """
    Generate the exact Detector A and Detector B marker observations
    from the forward projector.

    The returned order is:

        M1, M2, M3
    """

    projections = project_markers(
        MARKERS_CK
    )

    names = [
        "M1",
        "M2",
        "M3"
    ]

    points_A = [
        projections[name][
            "Detector_A"
        ]
        for name in names
    ]

    points_B = [
        projections[name][
            "Detector_B"
        ]
        for name in names
    ]

    return (
        names,
        points_A,
        points_B
    )


def describe_mapping(
    names_A,
    names_B,
    permutation
):
    """
    Convert an index permutation into readable marker pairings.
    """

    pairings = []

    for index_A, index_B in enumerate(
        permutation
    ):

        pairings.append(
            (
                names_A[index_A],
                names_B[index_B]
            )
        )

    return pairings


def run_test(
    test_name,
    names_A,
    points_A,
    names_B,
    points_B,
    expected_permutation
):
    """
    Run both correspondence methods on one test case.
    """

    print()
    print(test_name)
    print()

    rem_permutation, rem_score, rem_details = (
        correspondence_by_rem(
            points_A,
            points_B
        )
    )

    print("Method 1: minimum total REM")

    print(
        "  predicted permutation:",
        expected_permutation
    )

    print(
        "  computed permutation:",
        rem_permutation
    )

    print(
        "  pairings:",
        describe_mapping(
            names_A,
            names_B,
            rem_permutation
        )
    )

    print(
        "  total REM:",
        f"{rem_score:.12e}",
        "mm"
    )

    print(
        "  result:",
        (
            "PASS"
            if tuple(rem_permutation)
            == tuple(expected_permutation)
            else "FAIL"
        )
    )

    print()

    epipolar_permutation, epipolar_score, cost_matrix = (
        correspondence_by_epipolar(
            points_A,
            points_B
        )
    )

    print("Method 2: epipolar geometry")

    print(
        "  predicted permutation:",
        expected_permutation
    )

    print(
        "  computed permutation:",
        epipolar_permutation
    )

    print(
        "  pairings:",
        describe_mapping(
            names_A,
            names_B,
            epipolar_permutation
        )
    )

    print(
        "  total epipolar distance:",
        f"{epipolar_score:.12e}",
        "mm"
    )

    print(
        "  cost matrix [mm]:"
    )

    print(
        np.round(
            cost_matrix,
            6
        )
    )

    print(
        "  result:",
        (
            "PASS"
            if tuple(epipolar_permutation)
            == tuple(expected_permutation)
            else "FAIL"
        )
    )

    return {
        "method_1": {
            "permutation": list(
                rem_permutation
            ),
            "total_REM_mm": float(
                rem_score
            )
        },

        "method_2": {
            "permutation": list(
                epipolar_permutation
            ),
            "total_epipolar_distance_mm":
                float(
                    epipolar_score
                ),
            "cost_matrix_mm":
                cost_matrix.tolist()
        }
    }


def save_test_results(
    results,
    filename="data/marker_correspondence.json"
):
    """
    Save correspondence test results.
    """

    os.makedirs(
        os.path.dirname(
            filename
        ),
        exist_ok=True
    )

    with open(
        filename,
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )


if __name__ == "__main__":

    print(
        "\nPART 8 - MARKER CORRESPONDENCES\n"
    )

    names, points_A, points_B = (
        get_exact_projected_markers()
    )

    all_results = {}

    # TEST 1
    #
    # Normal ordering:
    #
    # Detector A:
    #     M1, M2, M3
    #
    # Detector B:
    #     M1, M2, M3
    #
    # Therefore:
    #
    #     A0 -> B0
    #     A1 -> B1
    #     A2 -> B2

    expected_normal = (
        0,
        1,
        2
    )

    all_results[
        "normal"
    ] = run_test(
        "TEST 1 - NORMAL MARKER ORDER",
        names,
        points_A,
        names,
        points_B,
        expected_normal
    )

    # TEST 2
    #
    # Swap M1 and M2 in Detector B.
    #
    # Detector A:
    #     M1, M2, M3
    #
    # Detector B input:
    #     M2, M1, M3
    #
    # Correct correspondence must therefore be:
    #
    #     A0 (M1) -> B1
    #     A1 (M2) -> B0
    #     A2 (M3) -> B2

    swapped_order = [
        1,
        0,
        2
    ]

    swapped_points_B = [
        points_B[index]
        for index in swapped_order
    ]

    swapped_names_B = [
        names[index]
        for index in swapped_order
    ]

    expected_swapped = (
        1,
        0,
        2
    )

    all_results[
        "swapped"
    ] = run_test(
        "TEST 2 - M1 AND M2 SWAPPED IN DETECTOR B",
        names,
        points_A,
        swapped_names_B,
        swapped_points_B,
        expected_swapped
    )

    save_test_results(
        all_results
    )

    print()
    print(
        "Saved data/marker_correspondence.json"
    )

    
# author: Serhat