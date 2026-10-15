# CyberKnife Marker Reconstruction Project

This project implements a simplified CyberKnife imaging and marker-reconstruction pipeline. It covers coordinate-frame transformations, forward projection, digitally reconstructed radiographs, marker localization, 3-D reconstruction, marker correspondence, ambiguity detection, and reconstruction-error analysis.

## How to run

**For the clearest walkthrough of the whole project, run the demo flow rather than each Python file individually.**

From the repository root:

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the demo:

   ```bash
   python demo_flow.py
   ```
## Automated tests

Run the automated test suite from the repository root:

    python -m pytest -v

The tests cover coordinate-frame transformations, marker transformations,
detector geometry, round-trip consistency, JSON export, and visualization.

The commands work in Windows PowerShell and in macOS/Linux terminals. The demo pauses between parts so you can inspect the terminal output and images before continuing. The DRR images it needs are already included in `data/`, so it can be run straight away.

## Demo flow

The demo walks through the project in assignment order:

1. Frame transforms
2. Marker transforms
3. Forward projector
4. Marker projector
5. Digitally reconstructed radiographs
6. Marker localization
7. Marker reconstruction
8. Marker correspondences
9. Marker ambiguity
10. Marker reconstruction error analysis

During **Part 6**, the program will ask the user to manually click the centre of each marker in the Detector A and Detector B images. These manual selections are then compared with the automatic localization and true projected marker positions.

Some computationally expensive results, especially the DRR generation and exhaustive reconstruction-error simulation, are displayed from previously generated output files during the demo rather than recomputed every time.

## Requirements

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

The project uses:

- NumPy
- Matplotlib
- SciPy
- Trimesh
- Rtree
- Jupyter
- Pytest

## Repository structure

```text
repository/
│
├── demo_flow.py
├── transforms.py
├── marker_transforms.py
├── forward_projector.py
├── marker_projector.py
├── drr.py
├── localize.py
├── reconstruction.py
├── correspond.py
├── ambiguity.py
├── error_analysis.py
├── geometry.py
├── utils.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── LumbarVertebrae.-witrh-Markers.stl
│   ├── marker_transforms.json
│   ├── markers_ck.png
│   ├── sphere_drr_A.npy
│   ├── sphere_drr_B.npy
│   ├── sphere_drr_A.png
│   ├── sphere_drr_B.png
│   ├── vertebra_drr_A.npy
│   ├── vertebra_drr_B.npy
│   ├── vertebra_drr_A.png
│   ├── vertebra_drr_B.png
│   ├── marker_localization.json
│   ├── marker_localization_A.png
│   ├── marker_localization_B.png
│   ├── marker_reconstruction.json
│   ├── marker_correspondence.json
│   ├── marker_ambiguity.json
│   ├── ambiguity_detector_A.png
│   ├── ambiguity_detector_B.png
│   ├── error_analysis_summary.json
│   ├── MLE_vs_MRE.png
│   └── REM_vs_MRE.png
│
├── notebooks/
│   ├── module_tests.ipynb
│   └── module_tests.py
│
└── tests/
    └── test_util.py
    └── test_transforms.py
    └── test_marker_transforms.py
```

## Individual modules

The modules can also be run separately if needed.

```bash
python transforms.py
python marker_transforms.py
python forward_projector.py
python marker_projector.py
python drr.py
python localize.py
python localize.py --manual
python reconstruction.py
python correspond.py
python ambiguity.py
python error_analysis.py
```

However, for reviewing or demonstrating the complete project, the preferred command is:

```bash
python demo_flow.py
```
## Notebook

The full assignment testing and discussion are contained in:

```text
notebooks/module_tests.ipynb
```

The notebook includes the predictions, tests, numerical results, figures, and findings for Parts 1 through 10.

The notebook is saved with all of its outputs, so it can be read directly on GitHub without running anything. The main results are also summarised below, in case the notebook does not open.

### If running the code fails

- Install the dependencies first with `pip install -r requirements.txt`. `drr.py` needs `trimesh`.
- The DRR arrays (`data/sphere_drr_*.npy`, `data/vertebra_drr_*.npy`) are included in the repository. If they are missing, or after changing `drr.py`, run `python drr.py` to regenerate them.
- Run `localize.py` with `--manual` (or through the demo). Running it without `--manual` overwrites the saved manual localization results.

## Results summary

The same results as in `notebooks/module_tests.ipynb`, in condensed form.

| Part | What was tested | Result |
|---|---|---|
| 1. Frame transforms | MD ↔ CK, CK ↔ Detector A/B and Detector ↔ Image, each on points with easy predictions (e.g. detector centre → (0, 0, 0), X-ray source → (0, 0, 2000), detector (0, 0) mm → image (1000, 1000) px) | All 24 predictions matched |
| 2. Marker transforms | The three markers transformed MD → CK → Detector A/B and plotted in 3D | MD and CK coordinates are identical, as expected for aligned frames; saved to `data/marker_transforms.json` and `data/markers_ck.png` |
| 3. Forward projector | Three points with hand predictions, using magnification M = 2000 / (1000 − d) | All six projections matched; every hit lies on the detector plane (w ≈ 0) |
| 4. Marker projector | Rough hand estimates assuming M ≈ 2 | Every marker landed on the predicted side of both detectors, within a few mm of the estimate |
| 5. DRR | Homogeneous sphere (R = 50 mm) and the vertebra STL, on both detectors, with Beer–Lambert attenuation | Images match the predicted appearance; see `data/sphere_drr_*.png` and `data/vertebra_drr_*.png` |
| 6. Marker localization | Automatic segmentation vs manual clicks vs true projected positions | Automatic: 0.04–0.15 mm (mean 0.09 mm). Manual: 0.40–1.21 mm (mean 0.84 mm) |
| 7. Marker reconstruction | Exact detector positions back-projected to 3D | True positions recovered; error and REM about 10⁻¹³ mm |
| 8. Marker correspondences | Two methods (minimum total REM, epipolar distance), with markers in normal order and with M1/M2 swapped in Detector B | Both methods found the correct pairing in both tests |
| 9. Marker ambiguity | Two methods, on the original markers and on markers moved into the z = 0 plane (the plane containing both X-ray sources) | Original: one unique pairing. z = 0: all six pairings equally plausible, so both methods detect the ambiguity |
| 10. Error analysis | MLE 0–10 mm, 360 × 360 error directions, all three markers | maxMLE = 7 mm (worst MRE 4.96 mm; 5.66 mm at 8 mm). REM–MRE correlation 0.31; no REM threshold separates safe from unsafe cases |

## Limitations of the error analysis

- **Integer MLE steps.** Only whole millimetres were tested, so the true limit lies between 7 and 8 mm. The worst-case MRE grew almost exactly linearly (about 0.71 mm per mm of MLE), which puts the actual limit near 7.06 mm; 7 mm is the safe, rounded-down answer.
- **1° direction steps.** Directions between the tested angles were not checked. The error changes smoothly with direction, so the true worst case can only be marginally larger than the sampled one.
- **Equal error in both images.** Each test shifted the marker by exactly the same MLE on both detectors. Real localization errors differ between the two images and are random, so these results are a worst-case bound rather than a prediction of typical errors.
- **Ideal geometry.** The source and detector positions, the ±45° angles and the marker centres are assumed to be known exactly. Calibration errors, patient motion and image noise were not modelled, so in practice the reconstruction error would be larger than localization error alone predicts.
- **Three markers near the isocentre.** All tested markers lie within about 50 mm of the isocentre. Markers further away are magnified differently, so the error scaling would change slightly.

MLE is measured on the detector, where everything is magnified about 2×. The automatic localization in Part 6 was within 0.15 mm and manual clicking within 1.21 mm, both far below the 7 mm limit, so this set-up has a large safety margin. REM should be combined with other checks rather than used on its own as a safety alarm.

## AI attribution

Parts of this project were developed with help from Claude (Anthropic), an AI assistant. It was used to help draft and refactor code, design the forward and marker projector tests and their explanations. All AI-assisted code and text were reviewed by the team, and the code was tested with the predictions and checks shown in each part of the notebook.

## Main final result

The exhaustive marker localization error simulation found that the largest tested integer localization error that kept marker reconstruction error below the required 5 mm limit for every tested direction was:

```text
maxMLE = 7 mm
```

The residual error metric, REM, was useful for describing geometric disagreement between the two back-projection rays, but it was not a reliable standalone warning threshold for reconstruction accuracy.

Authors: Serhat, Akshay, Elisa
