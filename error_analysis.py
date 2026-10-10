import os
import json

import numpy as np
import matplotlib.pyplot as plt

from geometry import (
    DETECTOR_A,
    DETECTOR_B
)

from marker_transforms import MARKERS_CK
from marker_projector import project_markers


# PART 10: MARKER RECONSTRUCTION ERROR ANALYSIS
#
# Marker Localization Error (MLE):
#     The magnitude of the 2D error applied to a marker location
#     in each detector image.
#
# Marker Reconstruction Error (MRE):
#     The 3D distance between the reconstructed marker position
#     and the known true CK marker position.
#
# Residual Error Metric (REM):
#     The shortest distance between the two back-projection lines.
#
# For each MLE, the error direction in Detector A is independently
# varied from 0 to 359 degrees, and the error direction in Detector B
# is independently varied from 0 to 359 degrees.
#
# This gives:
#
#     360 * 360 = 129,600
#
# direction combinations per marker per MLE.


MAX_MRE = 5.0

# We test far enough to find the transition from safe to unsafe.
# If no unsafe MLE is found by 10 mm, increase this value.
MAX_MLE_TO_TEST = 10

MLE_VALUES = np.arange(
    0,
    MAX_MLE_TO_TEST + 1,
    dtype=int
)

ANGLES_DEG = np.arange(
    360,
    dtype=float
)

ANGLES_RAD = np.radians(
    ANGLES_DEG
)

ERROR_DIRECTIONS = np.column_stack(
    (
        np.cos(ANGLES_RAD),
        np.sin(ANGLES_RAD)
    )
)


def detector_rays_for_mle(
    detector,
    true_detector_point,
    mle
):
    """
    Generate the 360 perturbed back-projection rays for one detector.

    The true detector point is shifted by exactly MLE millimetres.

    Direction theta produces:

        delta_u = MLE * cos(theta)
        delta_v = MLE * sin(theta)

    Therefore every perturbation has exactly the requested magnitude:

        sqrt(delta_u^2 + delta_v^2) = MLE

    Returns
    -------
    directions : ndarray, shape (360, 3)
        Unit back-projection directions in CK coordinates.
    """

    true_detector_point = np.asarray(
        true_detector_point,
        dtype=float
    )

    offsets = (
        mle
        * ERROR_DIRECTIONS
    )

    u = (
        true_detector_point[0]
        + offsets[:, 0]
    )

    v = (
        true_detector_point[1]
        + offsets[:, 1]
    )

    center = np.asarray(
        detector["center"],
        dtype=float
    )

    u_axis = np.asarray(
        detector["u"],
        dtype=float
    )

    v_axis = np.asarray(
        detector["v"],
        dtype=float
    )

    source = np.asarray(
        detector["source"],
        dtype=float
    )

    detector_points_ck = (
        center[None, :]
        + u[:, None] * u_axis[None, :]
        + v[:, None] * v_axis[None, :]
    )

    directions = (
        detector_points_ck
        - source[None, :]
    )

    directions /= np.linalg.norm(
        directions,
        axis=1,
        keepdims=True
    )

    return directions


def reconstruct_all_direction_pairs(
    true_ck,
    true_detector_A,
    true_detector_B,
    mle
):
    """
    Reconstruct all 360 x 360 perturbation combinations for one
    marker and one MLE value.

    This is a vectorized version of the reconstruction method from
    Part 7.

    Each perturbed Detector A position creates one back-projection
    line and each perturbed Detector B position creates another.

    For every pair of lines:
        1. find the closest point on line A,
        2. find the closest point on line B,
        3. reconstruct using their midpoint,
        4. calculate MRE,
        5. calculate REM.

    Returns
    -------
    mre : ndarray, shape (360, 360)
        Marker reconstruction error for every direction pair.

    rem : ndarray, shape (360, 360)
        Residual error metric for every direction pair.
    """

    true_ck = np.asarray(
        true_ck,
        dtype=float
    )

    directions_A = detector_rays_for_mle(
        DETECTOR_A,
        true_detector_A,
        mle
    )

    directions_B = detector_rays_for_mle(
        DETECTOR_B,
        true_detector_B,
        mle
    )

    source_A = np.asarray(
        DETECTOR_A["source"],
        dtype=float
    )

    source_B = np.asarray(
        DETECTOR_B["source"],
        dtype=float
    )

    # The two lines are:
    #
    #   LA(t) = source_A + t * direction_A
    #   LB(s) = source_B + s * direction_B
    #
    # Both direction vectors are normalized.
    #
    # The closest-points equations are solved simultaneously for
    # every one of the 360 x 360 direction combinations.

    w0 = (
        source_A
        - source_B
    )

    # b[i,j] = direction_A[i] dot direction_B[j]
    b = (
        directions_A
        @ directions_B.T
    )

    # d[i] = direction_A[i] dot w0
    d = (
        directions_A
        @ w0
    )

    # e[j] = direction_B[j] dot w0
    e = (
        directions_B
        @ w0
    )

    # Since the direction vectors are normalized:
    #
    #   a = direction_A dot direction_A = 1
    #   c = direction_B dot direction_B = 1

    denominator = (
        1.0
        - b ** 2
    )

    if np.any(
        np.isclose(
            denominator,
            0.0
        )
    ):
        raise RuntimeError(
            "A pair of reconstruction rays became parallel "
            "or nearly parallel."
        )

    t = (
        b * e[None, :]
        - d[:, None]
    ) / denominator

    s = (
        e[None, :]
        - b * d[:, None]
    ) / denominator

    closest_A = (
        source_A[None, None, :]
        + t[:, :, None]
        * directions_A[:, None, :]
    )

    closest_B = (
        source_B[None, None, :]
        + s[:, :, None]
        * directions_B[None, :, :]
    )

    reconstructed = (
        closest_A
        + closest_B
    ) / 2.0

    rem = np.linalg.norm(
        closest_A
        - closest_B,
        axis=2
    )

    mre = np.linalg.norm(
        reconstructed
        - true_ck[None, None, :],
        axis=2
    )

    return (
        mre,
        rem
    )


def angle_pair_from_index(index):
    """
    Convert a 2D matrix index into the corresponding Detector A
    and Detector B error directions in degrees.
    """

    index_A, index_B = index

    return (
        int(ANGLES_DEG[index_A]),
        int(ANGLES_DEG[index_B])
    )


def analyze_simulation():
    """
    Run the complete exhaustive MLE simulation.

    For each MLE:
        3 markers
        x 360 Detector A directions
        x 360 Detector B directions

    = 388,800 reconstructions per MLE.

    For MLE = 0 through 10, this gives over four million
    reconstructions.
    """

    projections = project_markers(
        MARKERS_CK
    )

    marker_summaries = []
    mle_summaries = []

    # These arrays contain all computed MRE and REM values.
    # float32 is sufficient for the statistical analysis and keeps
    # memory usage reasonable.
    all_mre = []
    all_rem = []

    # A much smaller sample is kept for the REM-vs-MRE scatter plot.
    scatter_mre = []
    scatter_rem = []

    print(
        "\nPART 10 - MARKER RECONSTRUCTION ERROR ANALYSIS\n"
    )

    print(
        "Maximum acceptable MRE:",
        MAX_MRE,
        "mm"
    )

    print(
        "Directions per detector:",
        len(
            ANGLES_DEG
        )
    )

    print(
        "Direction combinations per marker per MLE:",
        len(ANGLES_DEG) ** 2
    )

    print()

    for mle in MLE_VALUES:

        print(
            f"MLE = {mle} mm"
        )

        mle_max_mre = 0.0
        mle_total_mre = 0.0
        mle_total_rem = 0.0
        mle_count = 0

        worst_marker = None
        worst_angles = None

        for marker_name, true_ck in MARKERS_CK.items():

            true_detector_A = projections[
                marker_name
            ][
                "Detector_A"
            ]

            true_detector_B = projections[
                marker_name
            ][
                "Detector_B"
            ]

            mre, rem = reconstruct_all_direction_pairs(
                true_ck,
                true_detector_A,
                true_detector_B,
                mle
            )

            max_index = np.unravel_index(
                np.argmax(mre),
                mre.shape
            )

            max_mre = float(
                mre[max_index]
            )

            min_mre = float(
                mre.min()
            )

            mean_mre = float(
                mre.mean()
            )

            max_rem = float(
                rem.max()
            )

            min_rem = float(
                rem.min()
            )

            mean_rem = float(
                rem.mean()
            )

            angle_A, angle_B = (
                angle_pair_from_index(
                    max_index
                )
            )

            marker_summaries.append(
                {
                    "MLE_mm":
                        int(mle),

                    "marker":
                        marker_name,

                    "minimum_MRE_mm":
                        min_mre,

                    "mean_MRE_mm":
                        mean_mre,

                    "maximum_MRE_mm":
                        max_mre,

                    "minimum_REM_mm":
                        min_rem,

                    "mean_REM_mm":
                        mean_rem,

                    "maximum_REM_mm":
                        max_rem,

                    "worst_angle_A_deg":
                        angle_A,

                    "worst_angle_B_deg":
                        angle_B
                }
            )

            print(
                f"  {marker_name}: "
                f"max MRE = {max_mre:.4f} mm, "
                f"mean MRE = {mean_mre:.4f} mm, "
                f"max REM = {max_rem:.4f} mm"
            )

            if max_mre > mle_max_mre:

                mle_max_mre = max_mre

                worst_marker = marker_name

                worst_angles = (
                    angle_A,
                    angle_B
                )

            mle_total_mre += float(
                mre.sum()
            )

            mle_total_rem += float(
                rem.sum()
            )

            mle_count += mre.size

            all_mre.append(
                mre.astype(
                    np.float32
                ).ravel()
            )

            all_rem.append(
                rem.astype(
                    np.float32
                ).ravel()
            )

            # Sample every tenth direction in each detector.
            # The exhaustive values are still used for all numerical
            # analysis. This sample is only for making the scatter
            # plot readable and reasonably small.
            scatter_mre.append(
                mre[
                    ::10,
                    ::10
                ].ravel()
            )

            scatter_rem.append(
                rem[
                    ::10,
                    ::10
                ].ravel()
            )

        mean_mre_for_mle = (
            mle_total_mre
            / mle_count
        )

        mean_rem_for_mle = (
            mle_total_rem
            / mle_count
        )

        safe = (
            mle_max_mre
            < MAX_MRE
        )

        mle_summaries.append(
            {
                "MLE_mm":
                    int(mle),

                "mean_MRE_mm":
                    float(
                        mean_mre_for_mle
                    ),

                "worst_MRE_mm":
                    float(
                        mle_max_mre
                    ),

                "mean_REM_mm":
                    float(
                        mean_rem_for_mle
                    ),

                "worst_marker":
                    worst_marker,

                "worst_angle_A_deg":
                    int(
                        worst_angles[0]
                    ),

                "worst_angle_B_deg":
                    int(
                        worst_angles[1]
                    ),

                "guaranteed_below_5_mm":
                    bool(
                        safe
                    )
            }
        )

        print(
            "  Worst over all markers:",
            f"{mle_max_mre:.4f} mm"
        )

        print(
            "  Safe for every tested direction:",
            "YES" if safe else "NO"
        )

        print()

    all_mre = np.concatenate(
        all_mre
    )

    all_rem = np.concatenate(
        all_rem
    )

    scatter_mre = np.concatenate(
        scatter_mre
    )

    scatter_rem = np.concatenate(
        scatter_rem
    )

    return (
        marker_summaries,
        mle_summaries,
        all_mre,
        all_rem,
        scatter_mre,
        scatter_rem
    )


def determine_max_mle(
    mle_summaries
):
    """
    Determine the largest tested MLE for which every marker and
    every tested direction combination has MRE < 5 mm.
    """

    safe_values = [
        result["MLE_mm"]
        for result in mle_summaries
        if result[
            "guaranteed_below_5_mm"
        ]
    ]

    if len(
        safe_values
    ) == 0:

        return None

    return max(
        safe_values
    )


def analyze_rem_warning(
    all_mre,
    all_rem
):
    """
    Evaluate whether REM can reliably warn about an unsafe MRE.

    An unsafe reconstruction is defined as:

        MRE >= 5 mm

    If a useful REM threshold exists, dangerous reconstructions
    should generally have larger REM values than safe ones.

    A perfect threshold would require:

        minimum REM among dangerous cases
            >
        maximum REM among safe cases

    If these ranges overlap, REM cannot perfectly separate safe and
    unsafe reconstructions.
    """

    dangerous = (
        all_mre
        >= MAX_MRE
    )

    safe = ~dangerous

    if not np.any(
        dangerous
    ):

        return {
            "dangerous_cases_found":
                False
        }

    correlation = float(
        np.corrcoef(
            all_mre,
            all_rem
        )[0, 1]
    )

    min_rem_dangerous = float(
        all_rem[
            dangerous
        ].min()
    )

    max_rem_dangerous = float(
        all_rem[
            dangerous
        ].max()
    )

    min_rem_safe = float(
        all_rem[
            safe
        ].min()
    )

    max_rem_safe = float(
        all_rem[
            safe
        ].max()
    )

    perfect_threshold_exists = (
        min_rem_dangerous
        > max_rem_safe
    )

    threshold_tests = []

    # These candidate thresholds are not assumed to be correct.
    # They show how many dangerous reconstructions would be missed
    # if REM >= threshold were used as a warning rule.
    for threshold in [
        0.01,
        0.1,
        0.5,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0
    ]:

        false_negatives = int(
            np.count_nonzero(
                dangerous
                & (
                    all_rem
                    < threshold
                )
            )
        )

        false_positives = int(
            np.count_nonzero(
                safe
                & (
                    all_rem
                    >= threshold
                )
            )
        )

        dangerous_count = int(
            np.count_nonzero(
                dangerous
            )
        )

        false_negative_fraction = (
            false_negatives
            / dangerous_count
        )

        threshold_tests.append(
            {
                "REM_threshold_mm":
                    threshold,

                "dangerous_cases_missed":
                    false_negatives,

                "dangerous_fraction_missed":
                    float(
                        false_negative_fraction
                    ),

                "safe_cases_flagged":
                    false_positives
            }
        )

    return {
        "dangerous_cases_found":
            True,

        "correlation_MRE_REM":
            correlation,

        "minimum_REM_dangerous_mm":
            min_rem_dangerous,

        "maximum_REM_dangerous_mm":
            max_rem_dangerous,

        "minimum_REM_safe_mm":
            min_rem_safe,

        "maximum_REM_safe_mm":
            max_rem_safe,

        "perfect_REM_threshold_exists":
            bool(
                perfect_threshold_exists
            ),

        "threshold_tests":
            threshold_tests
    }


def make_mle_mre_plot(
    mle_summaries,
    filename="data/MLE_vs_MRE.png"
):
    """
    Plot average and worst-case MRE as a function of MLE.

    The 5 mm clinical limit is shown as a horizontal line.
    """

    mle = np.array(
        [
            result["MLE_mm"]
            for result in mle_summaries
        ]
    )

    mean_mre = np.array(
        [
            result["mean_MRE_mm"]
            for result in mle_summaries
        ]
    )

    worst_mre = np.array(
        [
            result["worst_MRE_mm"]
            for result in mle_summaries
        ]
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        mle,
        mean_mre,
        marker="o",
        label="Mean MRE"
    )

    plt.plot(
        mle,
        worst_mre,
        marker="o",
        label="Worst-case MRE"
    )

    plt.axhline(
        MAX_MRE,
        linestyle="--",
        label="Maximum acceptable MRE = 5 mm"
    )

    plt.xlabel(
        "Marker Localization Error, MLE [mm]"
    )

    plt.ylabel(
        "Marker Reconstruction Error, MRE [mm]"
    )

    plt.title(
        "MLE vs MRE"
    )

    plt.xticks(
        mle
    )

    plt.grid()

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=200
    )

    plt.close()

    print(
        "Saved",
        filename
    )


def make_rem_mre_plot(
    scatter_mre,
    scatter_rem,
    filename="data/REM_vs_MRE.png"
):
    """
    Plot REM against MRE.

    A sample of the exhaustive simulation is used for visualization
    so the plot is readable. All exhaustive values are still used in
    the numerical REM analysis.
    """

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        scatter_rem,
        scatter_mre,
        s=5,
        alpha=0.25
    )

    plt.axhline(
        MAX_MRE,
        linestyle="--",
        label="Maximum acceptable MRE = 5 mm"
    )

    plt.xlabel(
        "Residual Error Metric, REM [mm]"
    )

    plt.ylabel(
        "Marker Reconstruction Error, MRE [mm]"
    )

    plt.title(
        "REM vs MRE"
    )

    plt.grid()

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=200
    )

    plt.close()

    print(
        "Saved",
        filename
    )


def save_results(
    marker_summaries,
    mle_summaries,
    max_mle,
    rem_analysis,
    filename="data/error_analysis_summary.json"
):
    """
    Save the important numerical results.

    The millions of individual direction combinations are not written
    to JSON because that would create an unnecessarily large file.
    The simulation still computes every combination exhaustively.
    """

    os.makedirs(
        "data",
        exist_ok=True
    )

    results = {
        "maximum_acceptable_MRE_mm":
            MAX_MRE,

        "directions_per_detector":
            360,

        "direction_combinations_per_marker_per_MLE":
            129600,

        "tested_MLE_values_mm":
            [
                int(value)
                for value in MLE_VALUES
            ],

        "maxMLE_mm":
            (
                None
                if max_mle is None
                else int(
                    max_mle
                )
            ),

        "MLE_summary":
            mle_summaries,

        "marker_summary":
            marker_summaries,

        "REM_analysis":
            rem_analysis
    }

    with open(
        filename,
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print(
        "Saved",
        filename
    )


def print_final_results(
    mle_summaries,
    max_mle,
    rem_analysis
):
    """
    Print the main answers requested by the assignment.
    """

    print()
    print(
        "FINAL RESULTS"
    )

    print()

    print(
        f"{'MLE':<8}"
        f"{'Mean MRE':<16}"
        f"{'Worst MRE':<16}"
        f"{'Safe for all directions':<24}"
    )

    for result in mle_summaries:

        print(
            f"{result['MLE_mm']:<8}"
            f"{result['mean_MRE_mm']:<16.4f}"
            f"{result['worst_MRE_mm']:<16.4f}"
            f"{str(result['guaranteed_below_5_mm']):<24}"
        )

    print()

    if max_mle is None:

        print(
            "No tested MLE guaranteed MRE < 5 mm."
        )

    else:

        print(
            "Largest tested MLE that guarantees "
            "MRE < 5 mm:",
            f"{max_mle} mm"
        )

    print()

    if not rem_analysis[
        "dangerous_cases_found"
    ]:

        print(
            "No MRE >= 5 mm occurred in the tested range."
        )

        print(
            "Increase MAX_MLE_TO_TEST before evaluating "
            "REM as a warning metric."
        )

        return

    print(
        "MRE/REM correlation:",
        f"{rem_analysis['correlation_MRE_REM']:.4f}"
    )

    print(
        "Minimum REM among unsafe reconstructions:",
        f"{rem_analysis['minimum_REM_dangerous_mm']:.6f}",
        "mm"
    )

    print(
        "Maximum REM among safe reconstructions:",
        f"{rem_analysis['maximum_REM_safe_mm']:.6f}",
        "mm"
    )

    print(
        "Perfect REM warning threshold exists:",
        (
            "YES"
            if rem_analysis[
                "perfect_REM_threshold_exists"
            ]
            else "NO"
        )
    )

    print()

    print(
        "Candidate REM threshold tests"
    )

    print(
        f"{'Threshold':<12}"
        f"{'Dangerous missed':<20}"
        f"{'Fraction missed':<20}"
        f"{'Safe flagged':<16}"
    )

    for result in rem_analysis[
        "threshold_tests"
    ]:

        print(
            f"{result['REM_threshold_mm']:<12.2f}"
            f"{result['dangerous_cases_missed']:<20}"
            f"{result['dangerous_fraction_missed']:<20.4f}"
            f"{result['safe_cases_flagged']:<16}"
        )


if __name__ == "__main__":

    os.makedirs(
        "data",
        exist_ok=True
    )

    (
        marker_summaries,
        mle_summaries,
        all_mre,
        all_rem,
        scatter_mre,
        scatter_rem
    ) = analyze_simulation()

    max_mle = determine_max_mle(
        mle_summaries
    )

    rem_analysis = analyze_rem_warning(
        all_mre,
        all_rem
    )

    make_mle_mre_plot(
        mle_summaries
    )

    make_rem_mre_plot(
        scatter_mre,
        scatter_rem
    )

    save_results(
        marker_summaries,
        mle_summaries,
        max_mle,
        rem_analysis
    )

    print_final_results(
        mle_summaries,
        max_mle,
        rem_analysis
    )

    
# author: Serhat