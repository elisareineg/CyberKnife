# CyberKnife Marker Reconstruction Project

This project implements a simplified CyberKnife imaging and marker-reconstruction pipeline. It covers coordinate-frame transformations, forward projection, digitally reconstructed radiographs, marker localization, 3-D reconstruction, marker correspondence, ambiguity detection, and reconstruction-error analysis.

---------->
---------->
---------->
## (IMPORTANT PLEASE DO IT!) Recommended way to run the project (IMPORTANT PLEASE DO IT!)

For the clearest walkthrough of the entire project, **run the demo flow** instead of executing each Python file individually.

From the repository root, run:

```powershell
python .\demo_flow.py
```

The demo pauses between parts so you can inspect the terminal output and images before continuing.

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

```powershell
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
└── notebooks/
    └── module_tests.ipynb
```

## Individual modules

The modules can also be run separately if needed.

```powershell
python .\transforms.py
python .\marker_transforms.py
python .\forward_projector.py
python .\marker_projector.py
python .\drr.py
python .\localize.py
python .\localize.py --manual
python .\reconstruction.py
python .\correspond.py
python .\ambiguity.py
python .\error_analysis.py
```

However, for reviewing or demonstrating the complete project, the preferred command is:

```powershell
python .\demo_flow.py
```

## Notebook

The full assignment testing and discussion are contained in:

```text
notebooks/module_tests.ipynb
```

The notebook includes the predictions, tests, numerical results, figures, and findings for Parts 1 through 10.

## Main final result

The exhaustive marker localization error simulation found that the largest tested integer localization error that kept marker reconstruction error below the required 5 mm limit for every tested direction was:

```text
maxMLE = 7 mm
```

The residual error metric, REM, was useful for describing geometric disagreement between the two back-projection rays, but it was not a reliable standalone warning threshold for reconstruction accuracy.

author: Serhat
