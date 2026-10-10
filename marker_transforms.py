import json
import os

import numpy as np
import matplotlib.pyplot as plt

from transforms import (
    md_to_ck,
    ck_to_detector
)


# PART 2: MARKER TRANSFORMS

# Marker coordinates given by the assignment in the Model frame.

MARKERS_MD = {
    "M1": np.array([-19.0, 28.0, 11.0]),
    "M2": np.array([43.0, 14.0, 2.0]),
    "M3": np.array([2.0, 48.0, -6.0])
}


def transform_markers(markers_md=MARKERS_MD):
    """
    Transform all markers through the relevant coordinate frames.

    MD -> CK -> Detector A / Detector B

    Note:
    Detector coordinates here represent the actual 3-D marker
    location expressed in each detector coordinate system.

    They are NOT X-ray projected detector locations.
    Projection is handled later by the Marker Projector.
    """

    results = {}

    for name, marker_md in markers_md.items():

        marker_ck = md_to_ck(marker_md)

        marker_detector_A = ck_to_detector(
            marker_ck,
            "A"
        )

        marker_detector_B = ck_to_detector(
            marker_ck,
            "B"
        )

        results[name] = {
            "MD": marker_md,
            "CK": marker_ck,
            "Detector_A_3D": marker_detector_A,
            "Detector_B_3D": marker_detector_B
        }

    return results


# CK marker coordinates for repeated use by other modules.

MARKERS_CK = {
    name: md_to_ck(marker)
    for name, marker in MARKERS_MD.items()
}


def save_results(results, filename="data/marker_transforms.json"):
    """
    Save marker transformations for repeated later use.
    """

    os.makedirs(
        os.path.dirname(filename),
        exist_ok=True
    )

    serializable = {}

    for name, values in results.items():

        serializable[name] = {}

        for frame, coordinates in values.items():

            serializable[name][frame] = (
                np.asarray(coordinates).tolist()
            )

    with open(filename, "w") as f:
        json.dump(
            serializable,
            f,
            indent=4
        )


def plot_markers_ck(
    results,
    filename="data/markers_ck.png"
):
    """
    Graphically inspect the marker arrangement
    in the CyberKnife frame.
    """

    os.makedirs(
        os.path.dirname(filename),
        exist_ok=True
    )

    fig = plt.figure()

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    for name, values in results.items():

        x, y, z = values["CK"]

        ax.scatter(x, y, z)

        ax.text(
            x,
            y,
            z,
            name
        )

    ax.set_xlabel("CK x [mm]")
    ax.set_ylabel("CK y [mm]")
    ax.set_zlabel("CK z [mm]")

    ax.set_title(
        "CyberKnife Marker Positions"
    )

    plt.savefig(
        filename,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()


if __name__ == "__main__":

    results = transform_markers()

    print("\nPART 2 - MARKER TRANSFORMS\n")

    for name, values in results.items():

        print(name)

        print(
            "  MD:",
            values["MD"]
        )

        print(
            "  CK:",
            values["CK"]
        )

        print(
            "  Detector A 3D:",
            values["Detector_A_3D"]
        )

        print(
            "  Detector B 3D:",
            values["Detector_B_3D"]
        )

        print()

    save_results(results)

    print(
        "Saved marker transformations to "
        "data/marker_transforms.json"
    )

    plot_markers_ck(results)

    print(
        "Saved marker plot to "
        "data/markers_ck.png"
    )

    
# author: Serhat