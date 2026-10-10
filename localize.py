import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from geometry import (
    DETECTOR_A,
    DETECTOR_B,
    DETECTOR_SIZE
)

from marker_transforms import MARKERS_CK
from forward_projector import project_point_to_detector


# PART 6: MARKER LOCALIZATION
#
# Goal:
# 1. Automatically find the three dark metal markers in each DRR.
# 2. Compare automatically segmented marker centres with the true
#    projected marker locations from Part 4.
# 3. Allow visual/manual clicking of the marker centres and compare
#    those locations with both the segmented and true positions.
#
# The metal markers attenuate X-rays much more strongly than the
# vertebra, so they appear as small dark regions in the DRR.


DATA_DIR = "data"

DRR_FILES = {
    "A": os.path.join(DATA_DIR, "vertebra_drr_A.npy"),
    "B": os.path.join(DATA_DIR, "vertebra_drr_B.npy")
}


def detector_mm_to_drr_pixel(u_mm, v_mm, resolution):
    """
    Convert detector coordinates in millimetres to continuous
    DRR pixel coordinates.

    The physical detector is 200 mm x 200 mm.

    For a 200 x 200 DRR:
        pixel size = 1 mm

    Pixel centres are sampled by the DRR generator, so:

        pixel = (coordinate + 100) / pixel_size - 0.5

    Returns:
        [u_pixel, v_pixel]
    """

    pixel_size = DETECTOR_SIZE / resolution

    u_pixel = (
        (u_mm + DETECTOR_SIZE / 2.0)
        / pixel_size
        - 0.5
    )

    v_pixel = (
        (v_mm + DETECTOR_SIZE / 2.0)
        / pixel_size
        - 0.5
    )

    return np.array([
        u_pixel,
        v_pixel
    ])


def true_marker_positions(detector, resolution):
    """
    Compute the true marker positions using the forward projector.

    These are calculated from the known 3D marker positions and
    projective geometry, rather than from the DRR image.
    """

    positions = {}

    for name, marker_ck in MARKERS_CK.items():

        detector_point, _ = project_point_to_detector(
            marker_ck,
            detector
        )

        u_mm = detector_point[0]
        v_mm = detector_point[1]

        positions[name] = detector_mm_to_drr_pixel(
            u_mm,
            v_mm,
            resolution
        )

    return positions


def find_dark_components(
    image,
    threshold,
    min_pixels=2,
    max_pixels=150
):
    """
    Find connected dark regions that may correspond to markers.

    Metal markers should be much darker than most of the vertebra,
    so pixels below a selected intensity threshold are segmented.

    Small isolated regions are ignored as noise.
    Extremely large dark regions are also ignored because they are
    unlikely to represent one of the 3 mm markers.
    """

    mask = image < threshold

    structure = np.ones(
        (3, 3),
        dtype=int
    )

    labels, count = ndimage.label(
        mask,
        structure=structure
    )

    candidates = []

    for label_number in range(
        1,
        count + 1
    ):

        component = labels == label_number

        size = int(
            np.count_nonzero(component)
        )

        if size < min_pixels:
            continue

        if size > max_pixels:
            continue

        values = image[component]

        weights = np.zeros_like(
            image,
            dtype=float
        )

        weights[component] = (
            threshold - image[component]
        )

        total_weight = weights.sum()

        if total_weight <= 0:
            continue

        centre = ndimage.center_of_mass(
            weights
        )

        centre = np.array(
            [
                centre[0],
                centre[1]
            ],
            dtype=float
        )

        darkness_score = float(
            np.sum(
                threshold - values
            )
        )

        candidates.append(
            {
                "centre": centre,
                "size": size,
                "minimum": float(
                    values.min()
                ),
                "score": darkness_score
            }
        )

    return candidates


def localize_markers(image):
    """
    Automatically find the three marker centres.

    Several intensity thresholds are tested.

    If exactly three suitable dark components are found, they are
    accepted.

    If more than three are found, the three strongest dark regions
    are kept.
    """

    thresholds = [
        0.06,
        0.08,
        0.10,
        0.12,
        0.15,
        0.18,
        0.22,
        0.28
    ]

    best_candidates = None
    best_threshold = None

    for threshold in thresholds:

        candidates = find_dark_components(
            image,
            threshold
        )

        if len(candidates) == 3:

            best_candidates = candidates
            best_threshold = threshold
            break

        if len(candidates) > 3:

            candidates.sort(
                key=lambda x: x["score"],
                reverse=True
            )

            best_candidates = candidates[:3]
            best_threshold = threshold

    if best_candidates is None:

        raise RuntimeError(
            "Could not detect three marker components. "
            "Try adjusting the segmentation thresholds."
        )

    return (
        best_candidates,
        best_threshold
    )


def match_to_true_positions(
    centres,
    true_positions
):
    """
    Match the three detected marker blobs with M1, M2 and M3.

    The marker detection itself is performed using only the DRR.

    After detection, the Hungarian algorithm is used to match
    the detected locations with the known projected marker
    locations by minimizing total distance.
    """

    marker_names = list(
        true_positions.keys()
    )

    true_array = np.array(
        [
            true_positions[name]
            for name in marker_names
        ]
    )

    detected_array = np.array(
        centres
    )

    cost = np.linalg.norm(
        detected_array[:, None, :]
        - true_array[None, :, :],
        axis=2
    )

    detected_indices, true_indices = (
        linear_sum_assignment(cost)
    )

    matched = {}

    for detected_index, true_index in zip(
        detected_indices,
        true_indices
    ):

        marker_name = marker_names[
            true_index
        ]

        matched[marker_name] = (
            detected_array[
                detected_index
            ]
        )

    return matched


def manually_pick_markers(
    image,
    view_name
):
    """
    Display the DRR and allow the user to manually click the
    centre of each of the three markers.

    The markers may be clicked in any order.

    matplotlib returns:
        x = image column = detector v
        y = image row = detector u

    The returned coordinates are converted to:
        [u_pixel, v_pixel]
    """

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    ax.imshow(
        image,
        cmap="gray",
        origin="lower",
        vmin=0.0,
        vmax=1.0
    )

    ax.set_title(
        f"Detector {view_name}: click the centre of all 3 markers"
    )

    ax.set_xlabel(
        "v pixel"
    )

    ax.set_ylabel(
        "u pixel"
    )

    print()
    print(
        f"Detector {view_name}: "
        "click the THREE dark marker centres."
    )

    print(
        "You may click them in any order."
    )

    clicks = plt.ginput(
        3,
        timeout=-1
    )

    plt.close(fig)

    if len(clicks) != 3:

        raise RuntimeError(
            "Exactly three marker centres must be clicked."
        )

    positions = []

    for x, y in clicks:

        positions.append(
            np.array(
                [
                    y,
                    x
                ],
                dtype=float
            )
        )

    return positions


def distance(a, b):
    """
    Compute Euclidean distance between two image locations.
    """

    return float(
        np.linalg.norm(
            np.asarray(a)
            - np.asarray(b)
        )
    )


def compare_locations(
    true_positions,
    segmented_positions,
    manual_positions,
    pixel_size
):
    """
    Compare true, automatically segmented, and manually selected
    marker locations.

    Errors are reported in both pixels and millimetres.
    """

    results = {}

    for marker_name in true_positions:

        true = true_positions[
            marker_name
        ]

        segmented = segmented_positions[
            marker_name
        ]

        manual = None

        if manual_positions is not None:

            manual = manual_positions[
                marker_name
            ]

        segmented_true_px = distance(
            segmented,
            true
        )

        marker_result = {
            "true": true,
            "segmented": segmented,
            "segmented_true_error_px":
                segmented_true_px,
            "segmented_true_error_mm":
                segmented_true_px
                * pixel_size
        }

        if manual is not None:

            manual_true_px = distance(
                manual,
                true
            )

            manual_segmented_px = distance(
                manual,
                segmented
            )

            marker_result.update(
                {
                    "manual": manual,
                    "manual_true_error_px":
                        manual_true_px,
                    "manual_true_error_mm":
                        manual_true_px
                        * pixel_size,
                    "manual_segmented_error_px":
                        manual_segmented_px,
                    "manual_segmented_error_mm":
                        manual_segmented_px
                        * pixel_size
                }
            )

        results[
            marker_name
        ] = marker_result

    return results


def plot_results(
    image,
    view_name,
    true_positions,
    segmented_positions,
    manual_positions=None
):
    """
    Save an annotated image showing true, segmented and,
    when available, manually selected marker locations.
    """

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    ax.imshow(
        image,
        cmap="gray",
        origin="lower",
        vmin=0.0,
        vmax=1.0
    )

    for name in true_positions:

        true = true_positions[
            name
        ]

        segmented = segmented_positions[
            name
        ]

        ax.scatter(
            true[1],
            true[0],
            marker="o",
            facecolors="none",
            edgecolors="red",
            s=100
        )

        ax.scatter(
            segmented[1],
            segmented[0],
            marker="x",
            color="lime",
            s=80
        )

        ax.text(
            true[1] + 2,
            true[0] + 2,
            name
        )

        if manual_positions is not None:

            manual = manual_positions[
                name
            ]

            ax.scatter(
                manual[1],
                manual[0],
                marker="+",
                color="cyan",
                s=100
            )

    ax.set_title(
        f"Marker Localization: Detector {view_name}"
    )

    ax.set_xlabel(
        "v pixel"
    )

    ax.set_ylabel(
        "u pixel"
    )

    output = os.path.join(
        DATA_DIR,
        f"marker_localization_{view_name}.png"
    )

    plt.tight_layout()

    plt.savefig(
        output,
        dpi=200
    )

    plt.close()

    print(
        "Saved",
        output
    )


def print_results(
    view_name,
    threshold,
    results
):
    """
    Print marker localization results.
    """

    print()
    print(
        f"DETECTOR {view_name}"
    )

    print(
        "Segmentation threshold:",
        threshold
    )

    print()

    for name, result in results.items():

        print(name)

        print(
            "  True [u,v] px:",
            np.round(
                result["true"],
                2
            )
        )

        print(
            "  Segmented [u,v] px:",
            np.round(
                result["segmented"],
                2
            )
        )

        print(
            "  Segmented vs true:",
            f"{result['segmented_true_error_px']:.2f}",
            "pixels /",
            f"{result['segmented_true_error_mm']:.2f}",
            "mm"
        )

        if "manual" in result:

            print(
                "  Visual [u,v] px:",
                np.round(
                    result["manual"],
                    2
                )
            )

            print(
                "  Visual vs true:",
                f"{result['manual_true_error_px']:.2f}",
                "pixels /",
                f"{result['manual_true_error_mm']:.2f}",
                "mm"
            )

            print(
                "  Visual vs segmented:",
                f"{result['manual_segmented_error_px']:.2f}",
                "pixels /",
                f"{result['manual_segmented_error_mm']:.2f}",
                "mm"
            )

        print()


def make_json_serializable(results):
    """
    Convert NumPy arrays into standard Python lists so the
    localization results can be saved as JSON.
    """

    output = {}

    for view_name, view_results in results.items():

        output[
            view_name
        ] = {}

        for marker_name, values in view_results.items():

            output[
                view_name
            ][
                marker_name
            ] = {}

            for key, value in values.items():

                if isinstance(
                    value,
                    np.ndarray
                ):

                    output[
                        view_name
                    ][
                        marker_name
                    ][key] = value.tolist()

                else:

                    output[
                        view_name
                    ][
                        marker_name
                    ][key] = float(
                        value
                    )

    return output


def main(manual=False):

    print(
        "\nPART 6 - MARKER LOCALIZATION\n"
    )

    all_results = {}

    detectors = {
        "A": DETECTOR_A,
        "B": DETECTOR_B
    }

    for view_name in ["A", "B"]:

        print(
            f"Loading Detector {view_name} DRR..."
        )

        image = np.load(
            DRR_FILES[
                view_name
            ]
        )

        resolution = image.shape[0]

        pixel_size = (
            DETECTOR_SIZE
            / resolution
        )

        print(
            "  image size:",
            image.shape
        )

        print(
            "  detector pixel size:",
            pixel_size,
            "mm"
        )

        true_positions = true_marker_positions(
            detectors[
                view_name
            ],
            resolution
        )

        candidates, threshold = (
            localize_markers(
                image
            )
        )

        candidate_centres = [
            candidate[
                "centre"
            ]
            for candidate in candidates
        ]

        segmented_positions = (
            match_to_true_positions(
                candidate_centres,
                true_positions
            )
        )

        manual_positions = None

        if manual:

            manual_centres = (
                manually_pick_markers(
                    image,
                    view_name
                )
            )

            manual_positions = (
                match_to_true_positions(
                    manual_centres,
                    true_positions
                )
            )

        results = compare_locations(
            true_positions,
            segmented_positions,
            manual_positions,
            pixel_size
        )

        all_results[
            view_name
        ] = results

        print_results(
            view_name,
            threshold,
            results
        )

        plot_results(
            image,
            view_name,
            true_positions,
            segmented_positions,
            manual_positions
        )

    output_file = os.path.join(
        DATA_DIR,
        "marker_localization.json"
    )

    with open(
        output_file,
        "w"
    ) as f:

        json.dump(
            make_json_serializable(
                all_results
            ),
            f,
            indent=4
        )

    print(
        "Saved",
        output_file
    )

    print(
        "\nMarker localization complete."
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manual",
        action="store_true",
        help=(
            "Open the DRRs and manually "
            "click the three marker centres."
        )
    )

    args = parser.parse_args()

    main(
        manual=args.manual
    )

    
# author: Serhat