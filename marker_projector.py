from geometry import (
    DETECTOR_A,
    DETECTOR_B
)

from marker_transforms import MARKERS_CK

from forward_projector import (
    project_point_to_detector,
    detector_to_image
)


# PART 4: MARKER PROJECTOR

def project_markers(markers_ck):
    """
    Project all markers in the CK frame onto detector A and detector B.

    Each detector projection is also converted into the corresponding
    image coordinates.

    Parameters
    ----------
    markers_ck : dict
        Dictionary containing marker names and their 3-D CK coordinates.

        Example:
            {
                "M1": np.array([x, y, z]),
                "M2": np.array([x, y, z]),
                "M3": np.array([x, y, z])
            }

    Returns
    -------
    results : dict
        Dictionary containing the CK coordinates, detector coordinates,
        and image coordinates for each marker.
    """

    results = {}

    for marker_name, marker_ck in markers_ck.items():

        # Project marker onto detector A
        detector_A, hit_A_ck = project_point_to_detector(
            marker_ck,
            DETECTOR_A
        )

        # Project marker onto detector B
        detector_B, hit_B_ck = project_point_to_detector(
            marker_ck,
            DETECTOR_B
        )

        # Convert detector coordinates into image coordinates
        image_A = detector_to_image(detector_A)
        image_B = detector_to_image(detector_B)

        # Save all useful results for this marker
        results[marker_name] = {
            "CK": marker_ck,

            "Detector_A": detector_A,
            "Detector_B": detector_B,

            "Hit_A_CK": hit_A_ck,
            "Hit_B_CK": hit_B_ck,

            "Image_A": image_A,
            "Image_B": image_B
        }

    return results


# Marker projector tests

if __name__ == "__main__":

    marker_results = project_markers(MARKERS_CK)

    print("\nMARKER PROJECTOR TESTS\n")

    for marker_name, result in marker_results.items():

        print(marker_name)

        print("  CK")
        print("   ", result["CK"])

        print("  Detector A [u,v,w] mm")
        print("   ", result["Detector_A"])

        print("  Image A [u,v] pixels")
        print("   ", result["Image_A"])

        print("  Detector A hit in CK coordinates")
        print("   ", result["Hit_A_CK"])

        print("  Detector B [u,v,w] mm")
        print("   ", result["Detector_B"])

        print("  Image B [u,v] pixels")
        print("   ", result["Image_B"])

        print("  Detector B hit in CK coordinates")
        print("   ", result["Hit_B_CK"])

        print()

        
# author: Serhat