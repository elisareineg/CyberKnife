import numpy as np

from geometry import DETECTOR_A, DETECTOR_B
from projector.forward_projector import (
    project_point_to_detector,
    detector_to_image
)


# PART 4: MARKER PROJECTOR

# Temporary marker coordinates
#
# MD and CK are currently aligned according to the assignment.
# To be replaced with marker coordinates returned by the Marker Transform module.
MARKERS_CK = {
    "M1": np.array([-19.0, 28.0, 11.0]),
    "M2": np.array([43.0, 14.0, 2.0]),
    "M3": np.array([2.0, 48.0, -6.0])
}


def project_markers(markers_ck):
    """
    Project all CK-frame markers onto detector A and detector B
    and convert their detector positions to image coordinates.
    """

    results = {}

    for marker_name, marker_ck in markers_ck.items():

        detector_A, _ = project_point_to_detector(
            marker_ck,
            DETECTOR_A
        )

        detector_B, _ = project_point_to_detector(
            marker_ck,
            DETECTOR_B
        )

        image_A = detector_to_image(detector_A)
        image_B = detector_to_image(detector_B)

        results[marker_name] = {
            "CK": marker_ck,
            "Detector_A": detector_A,
            "Detector_B": detector_B,
            "Image_A": image_A,
            "Image_B": image_B
        }

    return results


# Marker projector tests

if __name__ == "__main__":

    marker_results = project_markers(MARKERS_CK)

    print("\nMARKER PROJECTIONS\n")

    for name, result in marker_results.items():

        print(name)

        print("  CK")
        print("   ", result["CK"])

        print("  Detector A [u,v,w] mm")
        print("   ", result["Detector_A"])

        print("  Image A [u,v] pixels")
        print("   ", result["Image_A"])

        print("  Detector B [u,v,w] mm")
        print("   ", result["Detector_B"])

        print("  Image B [u,v] pixels")
        print("   ", result["Image_B"])

        print()