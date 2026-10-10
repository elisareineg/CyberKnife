import itertools
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from marker_transforms import MARKERS_CK
from marker_projector import project_markers

from reconstruction import reconstruct_point

from correspond import epipolar_cost_matrix


# PART 9: MARKER AMBIGUITY
#
# A correspondence is ambiguous when more than one one-to-one
# assignment of Detector A markers to Detector B markers is
# geometrically consistent.
#
# We detect ambiguity using two different approaches:
#
# Method 1:
#     Compare the total REM of every possible correspondence.
#
# Method 2:
#     Compare the total epipolar-line distance of every possible
#     correspondence.
#
# If more than one permutation produces essentially the same
# near-minimum score, there is no unique correspondence.


# Exact projections are being used in this experiment, so a very
# small tolerance is suitable. With real noisy image measurements,
# this tolerance would need to reflect the expected localization error.
AMBIGUITY_TOLERANCE = 1e-9


# The original markers have different z coordinates.
#
# To deliberately create an ambiguity, keep their x and y positions
# but place all three markers in the CK z = 0 plane.
#
# Both CyberKnife X-ray sources are also in this plane. Therefore all
# marker rays belong to the same epipolar plane.
AMBIGUOUS_MARKERS_CK = {
    "M1": np.array([-19.0, 28.0, 0.0]),
    "M2": np.array([43.0, 14.0, 0.0]),
    "M3": np.array([2.0, 48.0, 0.0])
}


def observation_on_detector(point):
    """
    Convert a projected detector point to an exact detector-plane
    observation [u, v, 0].
    """

    point = np.asarray(
        point,
        dtype=float
    )

    return np.array([
        point[0],
        point[1],
        0.0
    ])


def get_projected_marker_points(markers_ck):
    """
    Project a supplied marker configuration onto both detectors.

    The returned order is:
        M1, M2, M3
    """

    projections = project_markers(
        markers_ck
    )

    names = [
        "M1",
        "M2",
        "M3"
    ]

    points_A = [
        observation_on_detector(
            projections[name]["Detector_A"]
        )
        for name in names
    ]

    points_B = [
        observation_on_detector(
            projections[name]["Detector_B"]
        )
        for name in names
    ]

    return (
        names,
        points_A,
        points_B
    )


def all_rem_permutation_scores(
    points_A,
    points_B
):
    """
    Method 1.

    Try all 3! = 6 marker correspondences.

    For every permutation, reconstruct all three marker pairs and
    add their REM values.

    A normal configuration should have one clearly best permutation.

    An ambiguous configuration may have multiple permutations with
    essentially the same minimum REM.
    """

    scores = []

    for permutation in itertools.permutations(
        range(len(points_B))
    ):

        total_rem = 0.0
        pair_rems = []

        for index_A, index_B in enumerate(
            permutation
        ):

            _, rem, _, _ = reconstruct_point(
                points_A[index_A],
                points_B[index_B]
            )

            total_rem += rem
            pair_rems.append(
                float(rem)
            )

        scores.append(
            {
                "permutation": tuple(
                    permutation
                ),
                "score": float(
                    total_rem
                ),
                "pair_REMs": pair_rems
            }
        )

    scores.sort(
        key=lambda result: result["score"]
    )

    return scores


def all_epipolar_permutation_scores(
    points_A,
    points_B
):
    """
    Method 2.

    Build the epipolar-distance cost matrix.

    Then calculate the total epipolar distance for all six possible
    one-to-one assignments.

    A unique configuration should have one clearly best assignment.

    If several permutations have the same near-zero epipolar cost,
    the correspondence is ambiguous.
    """

    cost_matrix = epipolar_cost_matrix(
        points_A,
        points_B
    )

    scores = []

    for permutation in itertools.permutations(
        range(len(points_B))
    ):

        total_cost = 0.0
        pair_costs = []

        for index_A, index_B in enumerate(
            permutation
        ):

            cost = cost_matrix[
                index_A,
                index_B
            ]

            total_cost += cost

            pair_costs.append(
                float(cost)
            )

        scores.append(
            {
                "permutation": tuple(
                    permutation
                ),
                "score": float(
                    total_cost
                ),
                "pair_costs": pair_costs
            }
        )

    scores.sort(
        key=lambda result: result["score"]
    )

    return (
        scores,
        cost_matrix
    )


def detect_ambiguity(
    scores,
    tolerance=AMBIGUITY_TOLERANCE
):
    """
    Decide whether a set of correspondence scores is ambiguous.

    The best score is found first.

    Any other permutation whose score differs from the best score by
    no more than the tolerance is considered an equally plausible
    correspondence.

    If more than one permutation is plausible, the correspondence is
    classified as ambiguous.
    """

    if len(scores) == 0:

        raise ValueError(
            "No correspondence scores were supplied."
        )

    best_score = scores[0]["score"]

    plausible = [
        result
        for result in scores
        if (
            result["score"]
            - best_score
        ) <= tolerance
    ]

    ambiguous = (
        len(plausible) > 1
    )

    return (
        ambiguous,
        plausible
    )


def print_marker_projections(
    title,
    names,
    points_A,
    points_B
):
    """
    Print projected detector coordinates.
    """

    print()
    print(title)
    print()

    print(
        f"{'Marker':<10}"
        f"{'Detector A (u,v)':<28}"
        f"{'Detector B (u,v)':<28}"
    )

    for name, point_A, point_B in zip(
        names,
        points_A,
        points_B
    ):

        text_A = (
            f"({point_A[0]:.6f}, "
            f"{point_A[1]:.6f})"
        )

        text_B = (
            f"({point_B[0]:.6f}, "
            f"{point_B[1]:.6f})"
        )

        print(
            f"{name:<10}"
            f"{text_A:<28}"
            f"{text_B:<28}"
        )


def print_score_table(
    method_name,
    scores,
    ambiguous,
    plausible
):
    """
    Display all six correspondence scores and the ambiguity result.
    """

    print()
    print(method_name)
    print()

    print(
        f"{'Permutation':<18}"
        f"{'Score':<24}"
    )

    for result in scores:

        print(
            f"{str(result['permutation']):<18}"
            f"{result['score']:.12e}"
        )

    print()

    print(
        "Number of equally plausible permutations:",
        len(plausible)
    )

    print(
        "Ambiguity detected:",
        "YES" if ambiguous else "NO"
    )

    print(
        "Plausible permutations:"
    )

    for result in plausible:

        print(
            " ",
            result["permutation"],
            "score =",
            f"{result['score']:.12e}"
        )


def run_configuration_test(
    name,
    markers_ck,
    expected_ambiguous
):
    """
    Project a marker configuration and test ambiguity using both
    computational methods.
    """

    print()
    print(name)
    print()

    names, points_A, points_B = (
        get_projected_marker_points(
            markers_ck
        )
    )

    print_marker_projections(
        "PROJECTED MARKER LOCATIONS",
        names,
        points_A,
        points_B
    )

    rem_scores = all_rem_permutation_scores(
        points_A,
        points_B
    )

    rem_ambiguous, rem_plausible = (
        detect_ambiguity(
            rem_scores
        )
    )

    print_score_table(
        "METHOD 1: REM PERMUTATION TEST",
        rem_scores,
        rem_ambiguous,
        rem_plausible
    )

    print(
        "Expected:",
        (
            "AMBIGUOUS"
            if expected_ambiguous
            else "NOT AMBIGUOUS"
        )
    )

    print(
        "Result:",
        (
            "PASS"
            if rem_ambiguous
            == expected_ambiguous
            else "FAIL"
        )
    )

    epipolar_scores, cost_matrix = (
        all_epipolar_permutation_scores(
            points_A,
            points_B
        )
    )

    epipolar_ambiguous, epipolar_plausible = (
        detect_ambiguity(
            epipolar_scores
        )
    )

    print()
    print(
        "Epipolar cost matrix [mm]:"
    )

    print(
        np.round(
            cost_matrix,
            9
        )
    )

    print_score_table(
        "METHOD 2: EPIPOLAR GEOMETRY TEST",
        epipolar_scores,
        epipolar_ambiguous,
        epipolar_plausible
    )

    print(
        "Expected:",
        (
            "AMBIGUOUS"
            if expected_ambiguous
            else "NOT AMBIGUOUS"
        )
    )

    print(
        "Result:",
        (
            "PASS"
            if epipolar_ambiguous
            == expected_ambiguous
            else "FAIL"
        )
    )

    return {
        "REM_method": {
            "ambiguous": bool(
                rem_ambiguous
            ),
            "plausible_permutations": [
                list(
                    result["permutation"]
                )
                for result in rem_plausible
            ],
            "scores": [
                {
                    "permutation": list(
                        result["permutation"]
                    ),
                    "score": float(
                        result["score"]
                    )
                }
                for result in rem_scores
            ]
        },

        "epipolar_method": {
            "ambiguous": bool(
                epipolar_ambiguous
            ),
            "plausible_permutations": [
                list(
                    result["permutation"]
                )
                for result in epipolar_plausible
            ],
            "scores": [
                {
                    "permutation": list(
                        result["permutation"]
                    ),
                    "score": float(
                        result["score"]
                    )
                }
                for result in epipolar_scores
            ],
            "cost_matrix_mm":
                cost_matrix.tolist()
        }
    }


def plot_ambiguous_projections(
    markers_ck,
    filename_A="data/ambiguity_detector_A.png",
    filename_B="data/ambiguity_detector_B.png"
):
    """
    Plot the engineered ambiguous projections.

    Since all engineered markers have CK z = 0, all detector u
    coordinates should also be zero.
    """

    names, points_A, points_B = (
        get_projected_marker_points(
            markers_ck
        )
    )

    os.makedirs(
        "data",
        exist_ok=True
    )

    plt.figure(
        figsize=(7, 5)
    )

    for name, point in zip(
        names,
        points_A
    ):

        plt.scatter(
            point[1],
            point[0]
        )

        plt.text(
            point[1] + 1,
            point[0] + 1,
            name
        )

    plt.xlabel(
        "Detector v [mm]"
    )

    plt.ylabel(
        "Detector u [mm]"
    )

    plt.title(
        "Ambiguous Marker Projection - Detector A"
    )

    plt.xlim(
        -100,
        100
    )

    plt.ylim(
        -100,
        100
    )

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        filename_A,
        dpi=200
    )

    plt.close()

    plt.figure(
        figsize=(7, 5)
    )

    for name, point in zip(
        names,
        points_B
    ):

        plt.scatter(
            point[1],
            point[0]
        )

        plt.text(
            point[1] + 1,
            point[0] + 1,
            name
        )

    plt.xlabel(
        "Detector v [mm]"
    )

    plt.ylabel(
        "Detector u [mm]"
    )

    plt.title(
        "Ambiguous Marker Projection - Detector B"
    )

    plt.xlim(
        -100,
        100
    )

    plt.ylim(
        -100,
        100
    )

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        filename_B,
        dpi=200
    )

    plt.close()

    print()
    print(
        "Saved",
        filename_A
    )

    print(
        "Saved",
        filename_B
    )


def save_results(
    results,
    filename="data/marker_ambiguity.json"
):
    """
    Save ambiguity test results.
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
        "\nPART 9 - MARKER AMBIGUITY\n"
    )

    print(
        "First test the original marker geometry."
    )

    print(
        "It should have one unique correspondence."
    )

    normal_results = run_configuration_test(
        "TEST 1 - ORIGINAL MARKERS",
        MARKERS_CK,
        expected_ambiguous=False
    )

    print()
    print(
        "Now engineer an ambiguity by placing all three "
        "markers in the CK z = 0 plane."
    )

    print(
        "Both X-ray sources also lie in this plane, so all "
        "marker rays share the same epipolar plane."
    )

    ambiguous_results = run_configuration_test(
        "TEST 2 - ENGINEERED AMBIGUOUS MARKERS",
        AMBIGUOUS_MARKERS_CK,
        expected_ambiguous=True
    )

    plot_ambiguous_projections(
        AMBIGUOUS_MARKERS_CK
    )

    all_results = {
        "normal": normal_results,
        "engineered_ambiguity":
            ambiguous_results
    }

    save_results(
        all_results
    )

    print()
    print(
        "Saved data/marker_ambiguity.json"
    )

# author: Serhat