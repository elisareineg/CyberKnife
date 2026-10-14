import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from transforms import md_to_ck, ck_to_detector


# PART 2: MARKER TRANSFORMS

# Marker coordinates given by the assignment in the Model frame.

MARKERS_MD = {
    "M1": np.array([-19.0, 28.0, 11.0]),
    "M2": np.array([43.0, 14.0, 2.0]),
    "M3": np.array([2.0, 48.0, -6.0])
}


def transform_markers(markers_md=None):
    """
    Transform marker positions from MD into CK and both detector frames.

    Detector coordinates represent 3D positions in each detector's
    local coordinate system, not projected image positions.
    """
     
    if markers_md is None:
        markers_md = MARKERS_MD

    results = {}

    for name, marker_md in markers_md.items():
        marker_md = np.asarray(marker_md, dtype=float)
        marker_ck = md_to_ck(marker_md)

        results[name] = {
            "MD": marker_md.copy(),
            "CK": marker_ck,
        }

        for view in ("A", "B"):
            results[name][f"Detector_{view}_3D"] = ck_to_detector(
                marker_ck, view
            )

    return results


# CK marker coordinates for repeated use by other modules.

MARKERS_CK = {
    name: md_to_ck(marker)
    for name, marker in MARKERS_MD.items()
}


def save_results(results, filename="data/marker_transforms.json"):
    """Save marker coordinates to a JSON file."""

    output_path = Path(filename)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        name: {
            frame: np.asarray(coordinates).tolist()
            for frame, coordinates in values.items()
        }
        for name, values in results.items()
    }

    with output_path.open("w") as file:
        json.dump(serializable, file, indent=4)


def plot_markers_ck(
    results,
    filename="data/markers_ck.png"
):
    """
    Graphically inspect the marker arrangement
    in the CyberKnife frame.
    """

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
    
    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)


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

    
# authors: Serhat, Akshay