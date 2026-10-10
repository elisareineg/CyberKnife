import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def pause(message="Press Enter to continue..."):
    input(f"\n{message}")


def run_script(filename):
    print()
    print("=" * 70)
    print(f"Running {filename}")
    print("=" * 70)
    subprocess.run(
        [sys.executable, str(ROOT / filename)],
        cwd=ROOT,
        check=True
    )


def show_image(filename, title):
    path = DATA / filename

    if not path.exists():
        print(f"Image not found: {path}")
        return

    image = plt.imread(path)

    plt.figure(figsize=(8, 7))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def show_part10_summary():
    path = DATA / "error_analysis_summary.json"

    if not path.exists():
        print("Part 10 summary has not been generated yet.")
        print("Run: python error_analysis.py")
        return

    with open(path, "r") as file:
        results = json.load(file)

    rem = results["REM_analysis"]

    print()
    print("Part 10 main results")
    print("--------------------")
    print("Maximum acceptable MRE: 5.0 mm")
    print("Maximum guaranteed-safe tested MLE:",
          results["maxMLE_mm"], "mm")
    print("MRE/REM correlation:",
          round(rem["correlation_MRE_REM"], 4))
    print("Perfect REM warning threshold exists:",
          rem["perfect_REM_threshold_exists"])


def main():
    print()
    print("=" * 70)
    print("CYBERKNIFE PROJECT DEMONSTRATION")
    print("=" * 70)
    print()
    print("This script walks through the project in assignment order.")
    print("The expensive DRR and error-analysis results are shown from")
    print("their already-generated output files instead of being recomputed.")

    pause()

    print("\nPART 1: FRAME TRANSFORMS")
    print("Show that MD, CK, detector and image-frame transforms work.")
    run_script("transforms.py")

    pause()

    print("\nPART 2: MARKER TRANSFORMS")
    print("Transform the three markers into the required coordinate frames.")
    run_script("marker_transforms.py")
    show_image(
        "markers_ck.png",
        "Marker Positions in the CK Frame"
    )

    pause()

    print("\nPART 3: FORWARD PROJECTOR")
    print("Project known 3-D CK points onto both X-ray detectors.")
    run_script("forward_projector.py")

    pause()

    print("\nPART 4: MARKER PROJECTOR")
    print("Project M1, M2 and M3 onto Detector A and Detector B.")
    run_script("marker_projector.py")

    pause()

    print("\nPART 5: DIGITALLY RECONSTRUCTED RADIOGRAPHS")
    print("The DRRs are expensive to regenerate, so this demo displays")
    print("the images that were already produced by drr.py.")

    show_image(
        "sphere_drr_A.png",
        "Sphere DRR - Detector A"
    )

    show_image(
        "sphere_drr_B.png",
        "Sphere DRR - Detector B"
    )

    show_image(
        "vertebra_drr_A.png",
        "Vertebra DRR - Detector A"
    )

    show_image(
        "vertebra_drr_B.png",
        "Vertebra DRR - Detector B"
    )

    pause()

    print("\nPART 6: MARKER LOCALIZATION")
    print("First the automatic method localizes the dark marker regions.")
    print("Then you will manually click the centre of each marker in both")
    print("detector images so the visual localization can be compared.")
    print()
    print("When the image window opens:")
    print("  click the centre of the three markers")
    print("  close or finish the window when all three points are selected")
    print()
    input("Press Enter when you are ready to start manual localization...")

    run_python = [
        sys.executable,
        str(ROOT / "localize.py"),
        "--manual"
    ]

    subprocess.run(
        run_python,
        cwd=ROOT,
        check=True
    )

    show_image(
        "marker_localization_A.png",
        "Marker Localization - Detector A"
    )

    show_image(
        "marker_localization_B.png",
        "Marker Localization - Detector B"
    )

    pause()

    print("\nPART 7: MARKER RECONSTRUCTION")
    print("Back-project corresponding detector observations into CK space")
    print("and calculate reconstruction error and REM.")
    run_script("reconstruction.py")

    pause()

    print("\nPART 8: MARKER CORRESPONDENCES")
    print("Resolve the identities of the three identical markers using")
    print("both REM-based matching and epipolar geometry.")
    run_script("correspond.py")

    pause()

    print("\nPART 9: MARKER AMBIGUITY")
    print("Create an ambiguous marker arrangement and show that both")
    print("methods detect that the correspondence is not unique.")
    run_script("ambiguity.py")

    show_image(
        "ambiguity_detector_A.png",
        "Engineered Ambiguity - Detector A"
    )

    show_image(
        "ambiguity_detector_B.png",
        "Engineered Ambiguity - Detector B"
    )

    pause()

    print("\nPART 10: RECONSTRUCTION ERROR ANALYSIS")
    print("The exhaustive 360 x 360 simulation is not rerun during the demo.")
    print("Instead, show the results generated earlier by error_analysis.py.")

    show_part10_summary()

    show_image(
        "MLE_vs_MRE.png",
        "Marker Localization Error vs Marker Reconstruction Error"
    )

    show_image(
        "REM_vs_MRE.png",
        "Residual Error Metric vs Marker Reconstruction Error"
    )

    print()
    print("=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print()
    print("Main final result:")
    print("maxMLE = 7 mm for maxMRE = 5 mm.")
    print("REM is useful geometrically, but not reliable by itself as")
    print("a safety threshold for reconstruction accuracy.")


if __name__ == "__main__":
    main()

# author: Serhat