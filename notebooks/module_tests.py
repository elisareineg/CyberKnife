# # CyberKnife — Module Tests
# One section per module in the assignment. Each section explains
# the test, predicts the result, runs the module and compares.
# All physical coordinates are in mm unless stated otherwise.

import sys

import os

import json

import numpy as np

import matplotlib.pyplot as plt

sys.path.insert(0, "..")

def fmt(uv):

    return "(%.2f, %.2f)" % tuple(

        np.round(uv, 2) + 0.0

    )

# ## Part 1 — Frame Transforms
# **Test required:** test each frame transform
# (MD ↔ CK, CK ↔ Detector A/B, Detector A/B ↔ Image A/B)
# on suitable points, explain the choice, predict the result,
# run the module and show that it matches.
# The MD and CK frames have the same origin and parallel axes in
# the assignment setup, so MD ↔ CK should be an identity transform.
# For each detector, three points with easy predictions are used:
# 1. The detector centre must have detector coordinates (0, 0, 0).
# 2. The X-ray source is 2000 mm from the detector along +w, so
# its detector coordinates must be (0, 0, 2000).
# 3. The point C + 10u + 20v must have coordinates (10, 20, 0).
# For Detector ↔ Image, the detector is 200 mm wide with 0.1 mm
# pixels, so the 2000 × 2000 image centre is (1000, 1000).
# Examples:
# detector (0, 0) mm → image (1000, 1000)
# detector (10, -20) mm → image (1100, 800)
# detector (-50, 50) mm → image (500, 1500)

from transforms import (

    md_to_ck,

    ck_to_md,

    ck_to_detector,

    detector_to_ck,

    detector_to_image,

    image_to_detector,

    get_detector

)

def check_transform(name, actual, expected, tolerance=1e-8):

    actual = np.asarray(actual, dtype=float)

    expected = np.asarray(expected, dtype=float)

    passed = np.allclose(

        actual,

        expected,

        atol=tolerance

    )

    print(name)

    print("  predicted:", expected)

    print("  actual:   ", actual)

    print("  result:   ", "PASS" if passed else "FAIL")

    print()

    return passed

print("PART 1 — FRAME TRANSFORM TESTS\n")

print("MD ↔ CK\n")

md_points = [

    np.array([0.0, 0.0, 0.0]),

    np.array([10.0, -20.0, 30.0]),

    np.array([-19.0, 28.0, 11.0])

]

for i, point in enumerate(md_points, start=1):

    check_transform(

        f"MD → CK point {i}",

        md_to_ck(point),

        point

    )

    check_transform(

        f"CK → MD point {i}",

        ck_to_md(point),

        point

    )

for view in ("A", "B"):

    detector = get_detector(view)

    C = detector["center"]

    S = detector["source"]

    u = detector["u"]

    v = detector["v"]

    test_point = C + 10.0 * u + 20.0 * v

    print(f"CK ↔ Detector {view}\n")

    check_transform(

        f"Detector {view} centre",

        ck_to_detector(C, view),

        [0.0, 0.0, 0.0]

    )

    check_transform(

        f"Detector {view} source",

        ck_to_detector(S, view),

        [0.0, 0.0, 2000.0]

    )

    check_transform(

        f"Detector {view} u/v point",

        ck_to_detector(test_point, view),

        [10.0, 20.0, 0.0]

    )

    check_transform(

        f"Detector {view} → CK origin",

        detector_to_ck(

            [0.0, 0.0, 0.0],

            view

        ),

        C

    )

    check_transform(

        f"Detector {view} → CK source",

        detector_to_ck(

            [0.0, 0.0, 2000.0],

            view

        ),

        S

    )

print("Detector ↔ Image\n")

detector_image_tests = [

    (

        np.array([0.0, 0.0, 0.0]),

        np.array([1000.0, 1000.0])

    ),

    (

        np.array([10.0, -20.0, 0.0]),

        np.array([1100.0, 800.0])

    ),

    (

        np.array([-50.0, 50.0, 0.0]),

        np.array([500.0, 1500.0])

    )

]

for i, (detector_point, image_point) in enumerate(

    detector_image_tests,

    start=1

):

    check_transform(

        f"Detector → Image point {i}",

        detector_to_image(detector_point),

        image_point

    )

    check_transform(

        f"Image → Detector point {i}",

        image_to_detector(image_point),

        detector_point

    )

# **Result.**
# All predicted and computed values matched. The extremely small
# values around 10^-14 that occasionally appear instead of zero are
# ordinary floating-point round-off errors.
# The tests show that the MD and CK frames are aligned as expected,
# both detector coordinate systems have the correct origins and axes,
# and the detector-to-image conversion correctly applies the
# 0.1 mm pixel spacing and image-centre offset.
# ## Part 2 — Marker Transforms
# **Test required:** transform the three markers through the relevant
# coordinate frames, save the results, and inspect them visually.
# The assignment gives the marker positions in the MD frame:
# M1 = (-19, 28, 11)
# M2 = (43, 14, 2)
# M3 = (2, 48, -6)
# Because MD and CK are aligned, the CK coordinates are expected to
# be identical to the MD coordinates.
# The same physical 3D marker positions are also expressed in the
# coordinate systems of Detector A and Detector B. These are not
# projected locations yet, so their detector w coordinates are not
# expected to be zero.

from marker_transforms import (

    MARKERS_MD,

    transform_markers

)

marker_transform_results = transform_markers()

print("PART 2 — MARKER TRANSFORMS\n")

for name, result in marker_transform_results.items():

    print(name)

    print("  MD:", result["MD"])

    print("  CK:", result["CK"])

    print(

        "  Detector A 3D:",

        result["Detector_A_3D"]

    )

    print(

        "  Detector B 3D:",

        result["Detector_B_3D"]

    )

    same_md_ck = np.allclose(

        result["MD"],

        result["CK"]

    )

    print(

        "  MD → CK prediction matched:",

        same_md_ck

    )

    print()

fig = plt.figure(figsize=(7, 6))

ax = fig.add_subplot(111, projection="3d")

for name, result in marker_transform_results.items():

    x, y, z = result["CK"]

    ax.scatter(

        x,

        y,

        z,

        s=60

    )

    ax.text(

        x,

        y,

        z,

        name

    )

ax.set_xlabel("CK x [mm]")

ax.set_ylabel("CK y [mm]")

ax.set_zlabel("CK z [mm]")

ax.set_title("Marker Positions in the CyberKnife Frame")

plt.show()

# **Result.**
# The MD and CK coordinates of all three markers were identical,
# which is the expected result because the two frames are aligned.
# The detector-frame values were different for Detector A and
# Detector B because the two detector coordinate systems are rotated
# differently around the patient.
# The 3D plot also showed that all three markers occupy distinct
# locations around the vertebra, which agrees with their expected
# anatomical placement. The numerical transformation results are
# saved separately by \`marker_transforms.py\` in
# \`data/marker_transforms.json\` for reuse by later modules.
# ## Part 3 — Forward Projector
# **Approach.** Each test point is chosen so that its projection can
# be predicted by hand and so that it checks one specific property
# of the projector.
# A point is magnified by:
# **M = SDD / (SAD − d) = 2000 / (1000 − d)**
# where *d* is how far the point lies from the isocentre towards that
# detector's X-ray source.
# Points level with the isocentre have d = 0 and are magnified
# exactly 2×. Points closer to the source are magnified more.
# Detector coordinates are:
# **u = M × z**
# and
# **v = M × sideways offset**
# along the detector's v axis.
# Detectors A and B sit at +45° and -45°.
# **Test points**
# 1. **P1 = (0, 0, 0)**, the isocentre.
# Both central rays pass through it.
# Predicted: A (0, 0), B (0, 0).
# 2. **P2 = (0, 0, 50)**.
# M = 2 for both detectors and +u is CK +z.
# Predicted: A (100, 0), B (100, 0).
# 3. **P3 = (7.07, 7.07, 50)**.
# It lies 10 mm along Detector A's v direction.
# Predicted: A (100, 20), B (101.01, 0).

from forward_projector import forward_project

tests = {

    "P1": (

        [0, 0, 0],

        {

            "A": (0, 0),

            "B": (0, 0)

        }

    ),

    "P2": (

        [0, 0, 50],

        {

            "A": (100, 0),

            "B": (100, 0)

        }

    ),

    "P3": (

        [

            10 / np.sqrt(2),

            10 / np.sqrt(2),

            50

        ],

        {

            "A": (100, 20),

            "B": (

                50 * 2000 / 990,

                0

            )

        }

    )

}

print(

    f"{'Point':<7}"

    f"{'Detector':<10}"

    f"{'Predicted (u, v)':<20}"

    f"{'Projected (u, v)':<20}"

    f"Match"

)

for name, (point, predicted) in tests.items():

    projected = forward_project(

        np.array(

            point,

            dtype=float

        )

    )

    for det in ("A", "B"):

        u, v, w = projected[det]

        match = (

            np.allclose(

                (u, v),

                predicted[det]

            )

            and abs(w) < 1e-9

        )

        print(

            f"{name:<7}"

            f"{det:<10}"

            f"{fmt(predicted[det]):<20}"

            f"{fmt((u, v)):<20}"

            f"{'yes' if match else 'NO'}"

        )

# **Result.**
# All six projections matched the predictions, and every projected
# point lay on the detector plane with w approximately zero.
# The projector correctly aligns both detector central rays with the
# isocentre, applies projective magnification, and gives the expected
# detector u and v directions.
# ## Part 4 — Marker Projector
# **Approach.**
# All three markers are within roughly 50 mm of the isocentre.
# For a rough hand calculation, depth can be ignored and the
# projection magnification approximated by M ≈ 2.
# Therefore:
# **u ≈ 2 × z**
# Detector A's v direction is approximately:
# (0.71, 0.71, 0)
# and Detector B's is approximately:
# (0.71, -0.71, 0)
# giving:
# Detector A sideways offset ≈ 0.71(x + y)
# Detector B sideways offset ≈ 0.71(x − y)
# **Hand estimates**
# | Marker | Estimate A | Estimate B |
# |---|---|---|
# | M1 = (-19, 28, 11) | (22, 12.7) | (22, -66.5) |
# | M2 = (43, 14, 2) | (4, 80.6) | (4, 41.0) |
# | M3 = (2, 48, -6) | (-12, 70.7) | (-12, -65.1) |

from marker_projector import (

    MARKERS_CK,

    project_markers

)

estimates = {

    "M1": {

        "A": (22, 12.7),

        "B": (22, -66.5)

    },

    "M2": {

        "A": (4, 80.6),

        "B": (4, 41.0)

    },

    "M3": {

        "A": (-12, 70.7),

        "B": (-12, -65.1)

    }

}

results = project_markers(

    MARKERS_CK

)

print(

    f"{'Marker':<8}"

    f"{'Detector':<10}"

    f"{'Estimate (u, v)':<18}"

    f"{'Projected (u, v)':<20}"

    f"{'Image (u, v) px':<22}"

    f"{'Diff':<8}"

    f"Close"

)

for name, est in estimates.items():

    for det in ("A", "B"):

        uv = results[

            name

        ][

            f"Detector_{det}"

        ][:2]

        image = results[

            name

        ][

            f"Image_{det}"

        ]

        diff = np.abs(

            uv - est[det]

        ).max()

        close = (

            diff < 4

            and np.all(

                np.sign(uv)

                == np.sign(est[det])

            )

        )

        print(

            f"{name:<8}"

            f"{det:<10}"

            f"{fmt(est[det]):<18}"

            f"{fmt(uv):<20}"

            f"{fmt(image):<22}"

            f"{diff:<8.2f}"

            f"{'yes' if close else 'NO'}"

        )

# **Result.**
# Every marker landed on the expected side of both detectors and
# stayed within a few millimetres of the rough hand estimate.
# The difference is expected because the hand calculation assumes a
# constant magnification of 2 and ignores each marker's depth relative
# to the two X-ray sources.
# ## Part 5 — Digitally Reconstructed Radiographs
# The DRR was developed in two stages.
# ### Phase 1: homogeneous sphere
# The vertebra was first replaced by a 50 mm radius homogeneous
# sphere centred at the CK origin. One ray was traced from the X-ray
# source through every detector sample.
# The length of each ray inside the sphere was calculated and
# Beer-Lambert attenuation was applied:
# **I = I0 exp(-μL)**
# The three metal markers were represented as 3 mm diameter spheres
# with a substantially larger attenuation coefficient.
# **Prediction before inspection**
# Because a sphere is rotationally symmetric, the outline should be
# circular and almost identical in views A and B. The centre should
# appear darker than the edges because central rays travel through
# more material. The three metal markers should appear as much darker
# small regions.
# ### Phase 2: vertebra STL
# The actual STL surface was then used. The largest connected mesh
# component was treated as the vertebra, while the three markers were
# still simulated separately so that metal and vertebra could use
# different attenuation values.
# **Prediction before inspection**
# Unlike the sphere, the vertebra should produce an irregular shape.
# Detector A and B should look different because the vertebra is not
# rotationally symmetric. Thicker regions should attenuate more, and
# the three markers should remain the darkest small structures.

sphere_A = np.load(

    "../data/sphere_drr_A.npy"

)

sphere_B = np.load(

    "../data/sphere_drr_B.npy"

)

vertebra_A = np.load(

    "../data/vertebra_drr_A.npy"

)

vertebra_B = np.load(

    "../data/vertebra_drr_B.npy"

)

print("Sphere Detector A")

print("  shape:", sphere_A.shape)

print("  min:", sphere_A.min())

print("  max:", sphere_A.max())

print("  mean:", sphere_A.mean())

print()

print("Sphere Detector B")

print("  shape:", sphere_B.shape)

print("  min:", sphere_B.min())

print("  max:", sphere_B.max())

print("  mean:", sphere_B.mean())

print()

print("Vertebra Detector A")

print("  shape:", vertebra_A.shape)

print("  min:", vertebra_A.min())

print("  max:", vertebra_A.max())

print("  mean:", vertebra_A.mean())

print()

print("Vertebra Detector B")

print("  shape:", vertebra_B.shape)

print("  min:", vertebra_B.min())

print("  max:", vertebra_B.max())

print("  mean:", vertebra_B.mean())

fig, ax = plt.subplots(figsize=(7, 7))

ax.imshow(

    sphere_A,

    cmap="gray",

    origin="lower",

    vmin=0,

    vmax=1

)

ax.set_title(

    "Sphere DRR — Detector A"

)

ax.set_xlabel("v pixel")

ax.set_ylabel("u pixel")

plt.show()

fig, ax = plt.subplots(figsize=(7, 7))

ax.imshow(

    sphere_B,

    cmap="gray",

    origin="lower",

    vmin=0,

    vmax=1

)

ax.set_title(

    "Sphere DRR — Detector B"

)

ax.set_xlabel("v pixel")

ax.set_ylabel("u pixel")

plt.show()

fig, ax = plt.subplots(figsize=(7, 7))

ax.imshow(

    vertebra_A,

    cmap="gray",

    origin="lower",

    vmin=0,

    vmax=1

)

ax.set_title(

    "Vertebra DRR — Detector A"

)

ax.set_xlabel("v pixel")

ax.set_ylabel("u pixel")

plt.show()

fig, ax = plt.subplots(figsize=(7, 7))

ax.imshow(

    vertebra_B,

    cmap="gray",

    origin="lower",

    vmin=0,

    vmax=1

)

ax.set_title(

    "Vertebra DRR — Detector B"

)

ax.set_xlabel("v pixel")

ax.set_ylabel("u pixel")

plt.show()

# **Findings.**
# The sphere DRRs matched the expected appearance. Both views showed
# a centred circular object because the sphere has the same shape
# from either viewing direction. The centre was darker than the outer
# edge because rays passing near the centre travel through more
# material. The three markers appeared as much darker spots because
# their simulated attenuation was much stronger.
# The vertebra DRRs were clearly different from the sphere images.
# The projected outline became irregular and the A and B views no
# longer looked identical. This is expected because the vertebra has
# processes, cavities and regions with very different thicknesses.
# Rays passing through thicker sections of the model were attenuated
# more strongly.
# The marker locations also changed between views A and B, as expected
# from the marker projection calculations in Part 4. Overall, the
# visual appearance agreed with the behaviour predicted from the
# projection geometry and attenuation model.
# These images are intentionally simplified DRRs rather than
# diagnostic-quality X-rays. The goal here is to preserve the useful
# projective geometry and attenuation behaviour needed for the later
# marker localization and reconstruction steps.
# ## Part 6 — Marker Localization
# The goal is to estimate the centre of each metal marker directly
# from the generated vertebra DRRs.
# ### Automatic localization
# Metal was assigned a much larger attenuation coefficient than the
# vertebra, so the markers appear as small, very dark regions.
# The localization algorithm:
# 1. thresholds the image to select very dark pixels,
# 2. groups connected dark pixels into components,
# 3. keeps components that have a reasonable marker-like size,
# 4. calculates an intensity-weighted centre for each marker region.
# The automatically detected positions are then matched to M1, M2 and
# M3 and compared with the true projected positions calculated using
# the forward projector.
# ### Visual localization
# The DRRs were also opened interactively and the centre of each
# marker was selected manually with the cursor.
# Since the vertebra DRRs were generated at 200 × 200 resolution over
# a 200 mm detector, one DRR pixel corresponds to approximately
# 1 mm on the detector.

with open(

    "../data/marker_localization.json",

    "r"

) as f:

    localization_results = json.load(f)

print(

    f"{'View':<7}"

    f"{'Marker':<8}"

    f"{'Seg. error mm':<16}"

    f"{'Visual error mm':<18}"

    f"{'Visual-Seg. mm':<16}"

)

segmented_errors = []

visual_errors = []

for view in ("A", "B"):

    for marker in ("M1", "M2", "M3"):

        result = localization_results[

            view

        ][

            marker

        ]

        seg_error = result[

            "segmented_true_error_mm"

        ]

        visual_error = result[

            "manual_true_error_mm"

        ]

        visual_seg_error = result[

            "manual_segmented_error_mm"

        ]

        segmented_errors.append(

            seg_error

        )

        visual_errors.append(

            visual_error

        )

        print(

            f"{view:<7}"

            f"{marker:<8}"

            f"{seg_error:<16.2f}"

            f"{visual_error:<18.2f}"

            f"{visual_seg_error:<16.2f}"

        )

print()

print(

    "Average segmented error:",

    f"{np.mean(segmented_errors):.2f} mm"

)

print(

    "Average visual error:",

    f"{np.mean(visual_errors):.2f} mm"

)

print(

    "Largest segmented error:",

    f"{np.max(segmented_errors):.2f} mm"

)

print(

    "Largest visual error:",

    f"{np.max(visual_errors):.2f} mm"

)

image_A = plt.imread(

    "../data/marker_localization_A.png"

)

plt.figure(figsize=(8, 8))

plt.imshow(image_A)

plt.axis("off")

plt.title(

    "Marker Localization — Detector A"

)

plt.show()

image_B = plt.imread(

    "../data/marker_localization_B.png"

)

plt.figure(figsize=(8, 8))

plt.imshow(image_B)

plt.axis("off")

plt.title(

    "Marker Localization — Detector B"

)

plt.show()

# ### Findings
# The automatic localization was very close to the true projected
# marker locations in both detector images. Across all six marker
# projections, the segmented error ranged from 0.04 mm to 0.15 mm.
# The average segmented error was about 0.09 mm.
# The manually selected locations were also close, but the differences
# were noticeably larger. The visual errors ranged from 0.42 mm to
# 1.46 mm and averaged about 0.90 mm.
# This difference is reasonable. The automatic method uses the
# intensity values from several pixels in the whole dark marker region
# to estimate its centre. When selecting the marker manually, I had
# to judge the centre by eye and place the mouse cursor there. Since
# the image resolution is 1 mm per pixel, being off by about one pixel
# already gives approximately 1 mm of localization error.
# M2 was particularly easy to select visually in Detector A, where
# the visual error was only 0.42 mm. M3 in Detector A had the largest
# visual error at 1.46 mm. This does not mean the physical marker
# moved. It mainly shows that manually choosing the exact centre of a
# small projected marker is less repeatable than calculating its centre
# from the image intensities.
# The localization errors were also not identical between Detector A
# and Detector B. The two images view the vertebra and markers from
# different directions, so each marker appears against a different
# surrounding part of the vertebra. This can slightly change how the
# dark marker region looks and where its apparent centre is judged.
# Overall, the visual, segmented and true locations agreed closely.
# The automatic localization was more consistent than manual clicking
# for these DRRs. The very small automatic errors are also helped by
# the fact that these are clean simulated images. Real X-ray images
# would contain additional noise, scatter, blur and other effects that
# could make marker localization less precise.
# ## Part 7 — Marker Reconstruction
# **Test required:** reconstruct a point in the CK frame from a pair
# of corresponding Detector A and Detector B observations. Also
# calculate the residual error metric, REM.
# The exact marker projections calculated earlier are used as the
# detector observations. Because these detector positions were
# generated from the known CK marker positions using the forward
# projector, reconstructing them should return the original CK
# coordinates.
# Each detector observation defines a back-projection line from its
# X-ray source through the observed detector point.
# For exact detector positions, the two back-projection lines should
# intersect at the true marker position.
# In practice, localization errors may make the lines slightly skew.
# The reconstruction method therefore finds the closest point on each
# back-projection line and uses the midpoint between them as the
# reconstructed 3D position.
# REM is defined as the shortest distance between these two
# back-projection lines. For the exact marker projections used in this
# test, REM should be zero or extremely close to zero because of
# floating-point numerical precision.

from reconstruction import reconstruct_markers

reconstruction_results = reconstruct_markers()

print(

    f"{'Marker':<8}"

    f"{'True CK':<25}"

    f"{'Reconstructed CK':<30}"

    f"{'3D error (mm)':<18}"

    f"{'REM (mm)':<18}"

)

for name, result in reconstruction_results.items():

    true_ck = result["true_ck"]

    reconstructed = result["reconstructed_ck"]

    error = result[

        "reconstruction_error_mm"

    ]

    rem = result[

        "REM_mm"

    ]

    print(

        f"{name:<8}"

        f"{str(np.round(true_ck, 6)):<25}"

        f"{str(np.round(reconstructed, 6)):<30}"

        f"{error:<18.3e}"

        f"{rem:<18.3e}"

    )

# ### Findings
# All three reconstructed marker positions matched their known CK
# coordinates.
# M1 reconstructed to (-19, 28, 11), M2 reconstructed to
# (43, 14, 2), and M3 reconstructed to (2, 48, -6), which are the
# original marker positions used by the forward projector.
# The reconstruction errors were either exactly zero or on the order
# of 10^-13 mm. The REM values were also on the order of 10^-13 mm or
# smaller. These values are effectively zero and come from normal
# floating-point rounding during the line calculations rather than
# from an actual reconstruction error.
# This result is what we expected because the detector positions used
# in this test were generated from the exact marker coordinates. Both
# back-projection rays therefore pass through the same 3D marker
# position.
# This test also provides a useful baseline for the later error
# analysis. Once localization error is added to the detector
# positions, the two back-projection lines will generally no longer
# intersect exactly. In that case, the reconstructed point will be
# taken from the midpoint of the closest points on the two rays, and
# REM will measure how far apart those rays are at their closest
# approach.
# ## Part 8 — Marker Correspondences
# The three implanted markers are identical, so after localization
# there is no visual label telling us which marker in Detector A
# corresponds to which marker in Detector B.
# A correct correspondence is required before two detector
# observations can be used to reconstruct a marker in 3D.
# Two different methods were implemented.
# ### Method 1: minimum total REM
# With three markers, there are only 3! = 6 possible one-to-one
# correspondences.
# For each possible assignment, the three A/B marker pairs are
# reconstructed using the Part 7 reconstruction module.
# The REM values of the three reconstructed pairs are added.
# The correspondence with the smallest total REM is selected.
# Correct pairs should have nearly zero REM because their two
# back-projection rays intersect at the same physical marker.
# Incorrect pairs normally produce skew rays and therefore a
# noticeably larger REM.
# ### Method 2: epipolar geometry
# A marker observation in Detector A defines a 3D back-projection
# ray. Together with the second X-ray source, this ray defines an
# epipolar plane.
# That plane intersects Detector B along an epipolar line.
# Therefore, the correct corresponding marker in Detector B should
# lie on or extremely close to that line.
# A cost matrix is built from the perpendicular distances of each
# Detector B marker to each Detector A marker's epipolar line.
# The Hungarian assignment algorithm then chooses the one-to-one
# correspondence with the smallest total epipolar distance.
# Two tests were performed:
# 1. Normal ordering:
# Detector A = [M1, M2, M3]
# Detector B = [M1, M2, M3]
# Expected permutation = (0, 1, 2)
# 2. Swapped ordering:
# Detector A = [M1, M2, M3]
# Detector B = [M2, M1, M3]
# Expected permutation = (1, 0, 2)

from correspond import (

    get_exact_projected_markers,

    correspondence_by_rem,

    correspondence_by_epipolar

)

names, points_A, points_B = (

    get_exact_projected_markers()

)

print("TEST 1 — NORMAL ORDER\n")

rem_perm_1, rem_score_1, _ = (

    correspondence_by_rem(

        points_A,

        points_B

    )

)

epi_perm_1, epi_score_1, epi_cost_1 = (

    correspondence_by_epipolar(

        points_A,

        points_B

    )

)

print(

    "REM method permutation:",

    rem_perm_1

)

print(

    "REM total:",

    f"{rem_score_1:.3e} mm"

)

print(

    "Epipolar permutation:",

    epi_perm_1

)

print(

    "Epipolar total:",

    f"{epi_score_1:.3e} mm"

)

print(

    "Epipolar cost matrix [mm]:"

)

print(

    np.round(

        epi_cost_1,

        6

    )

)

print()

swapped_points_B = [

    points_B[1],

    points_B[0],

    points_B[2]

]

print("TEST 2 — M1 AND M2 SWAPPED IN DETECTOR B\n")

rem_perm_2, rem_score_2, _ = (

    correspondence_by_rem(

        points_A,

        swapped_points_B

    )

)

epi_perm_2, epi_score_2, epi_cost_2 = (

    correspondence_by_epipolar(

        points_A,

        swapped_points_B

    )

)

print(

    "REM method permutation:",

    rem_perm_2

)

print(

    "REM total:",

    f"{rem_score_2:.3e} mm"

)

print(

    "Epipolar permutation:",

    epi_perm_2

)

print(

    "Epipolar total:",

    f"{epi_score_2:.3e} mm"

)

print(

    "Epipolar cost matrix [mm]:"

)

print(

    np.round(

        epi_cost_2,

        6

    )

)

# ### Findings
# Both correspondence methods correctly identified all three marker
# pairs in the normal test.
# The REM method returned the expected permutation (0, 1, 2).
# Its total REM was approximately 5.96 × 10^-13 mm, which is
# effectively zero. This is expected because the detector
# observations were generated from the exact same 3D marker
# positions, so the correct back-projection rays intersect.
# The epipolar method also returned (0, 1, 2). The correct entries in
# its cost matrix were essentially zero, while incorrect pairings had
# distances between roughly 16 mm and 35 mm. This gave a very clear
# separation between correct and incorrect correspondences.
# In the second test, M1 and M2 were deliberately swapped in the
# Detector B input. Both methods still recovered the correct physical
# correspondences and returned the expected permutation (1, 0, 2).
# This is important because the algorithms are not depending on the
# order in which the markers are supplied. They are using the
# geometry of the two X-ray views to determine which observations
# belong to the same physical marker.
# The two approaches reach the same result in different ways.
# The REM method directly asks which pairing produces the most
# geometrically consistent 3D reconstructions. The epipolar method
# avoids performing a full reconstruction for every possible pair
# and instead checks whether each Detector B point lies on the
# epipolar line predicted from Detector A.
# For these exact simulated marker locations, both methods give a
# very strong and unambiguous correspondence result.
# ## Part 9 — Marker Ambiguity
# Marker correspondence becomes ambiguous when more than one
# one-to-one assignment between Detector A and Detector B is
# geometrically consistent.
# Two different ambiguity-detection approaches were tested:
# 1. REM-based ambiguity detection
# 2. Epipolar-geometry ambiguity detection
# ### Normal marker geometry
# The original marker coordinates were:
# M1 = (-19, 28, 11)
# M2 = (43, 14, 2)
# M3 = (2, 48, -6)
# These markers do not all lie in the same epipolar plane, so the
# correct correspondence should be unique.
# ### Engineered ambiguous geometry
# To deliberately create ambiguity, the z coordinate of every marker
# was changed to zero:
# M1 = (-19, 28, 0)
# M2 = (43, 14, 0)
# M3 = (2, 48, 0)
# Both X-ray sources are also located in the CK z = 0 plane.
# Therefore all three markers and both X-ray sources lie in the same
# plane.
# This makes all three markers share the same epipolar plane.
# Because detector +u is aligned with CK +z, all three engineered
# markers also project to u = 0 on both detectors.
# In this geometry, incorrect marker pairings can still produce
# geometrically valid intersections, so the correspondence can no
# longer be determined uniquely.

from ambiguity import (

    AMBIGUOUS_MARKERS_CK,

    get_projected_marker_points,

    all_rem_permutation_scores,

    all_epipolar_permutation_scores,

    detect_ambiguity

)

from marker_transforms import MARKERS_CK

def print_ambiguity_test(

    label,

    markers

):

    print(label)

    print()

    names, points_A, points_B = (

        get_projected_marker_points(

            markers

        )

    )

    print("Projected marker positions:\n")

    for name, point_A, point_B in zip(

        names,

        points_A,

        points_B

    ):

        print(

            name,

            "A:",

            np.round(

                point_A[:2],

                6

            ),

            "B:",

            np.round(

                point_B[:2],

                6

            )

        )

    print()

    rem_scores = (

        all_rem_permutation_scores(

            points_A,

            points_B

        )

    )

    rem_ambiguous, rem_plausible = (

        detect_ambiguity(

            rem_scores

        )

    )

    print(

        "REM ambiguity detected:",

        rem_ambiguous

    )

    print(

        "Plausible REM permutations:"

    )

    for result in rem_plausible:

        print(

            result["permutation"],

            f"{result['score']:.3e}"

        )

    print()

    epipolar_scores, cost_matrix = (

        all_epipolar_permutation_scores(

            points_A,

            points_B

        )

    )

    epi_ambiguous, epi_plausible = (

        detect_ambiguity(

            epipolar_scores

        )

    )

    print(

        "Epipolar ambiguity detected:",

        epi_ambiguous

    )

    print(

        "Epipolar cost matrix [mm]:"

    )

    print(

        np.round(

            cost_matrix,

            6

        )

    )

    print(

        "Plausible epipolar permutations:"

    )

    for result in epi_plausible:

        print(

            result["permutation"],

            f"{result['score']:.3e}"

        )

    print()

print_ambiguity_test(

    "TEST 1 — ORIGINAL MARKER GEOMETRY",

    MARKERS_CK

)

print_ambiguity_test(

    "TEST 2 — ENGINEERED AMBIGUOUS GEOMETRY",

    AMBIGUOUS_MARKERS_CK

)

# ### Findings
# The original marker configuration was not ambiguous.
# For the REM-based method, only the correct permutation
# (0, 1, 2) had a near-zero score. Its total REM was about
# 5.96 × 10^-13 mm, while the incorrect permutations had much
# larger scores, beginning at about 16 mm.
# The epipolar method showed the same behaviour. Only the correct
# correspondence had essentially zero total epipolar distance.
# The incorrect pairings produced much larger distances.
# This means the original marker geometry provides enough geometric
# information to uniquely determine which marker in Detector A
# belongs to which marker in Detector B.
# The engineered configuration behaved very differently.
# After setting all three marker z coordinates to zero, every marker
# lay in the same plane as both X-ray sources.
# In the REM test, all six possible marker permutations had total
# REM values on the order of 10^-13 mm. These values are all
# effectively zero and differ only because of floating-point
# numerical precision. As a result, all six possible
# correspondences were classified as equally plausible.
# The epipolar test showed the ambiguity even more clearly.
# The complete epipolar cost matrix became zero:
# [[0, 0, 0],
# [0, 0, 0],
# [0, 0, 0]]
# This means every marker in Detector B lies on the epipolar line
# generated by every marker in Detector A. Epipolar geometry
# therefore gives no information that can distinguish one
# correspondence from another.
# This ambiguity was intentionally created by placing all of the
# markers in a common epipolar plane. In normal geometry, different
# markers usually produce different epipolar constraints. Here those
# constraints collapse onto the same line, so several physically
# different marker assignments satisfy the imaging geometry equally
# well.
# Both methods successfully detected the same ambiguity in different
# ways. The REM method detected multiple correspondence permutations
# with equally small reconstruction residuals, while the epipolar
# method detected multiple assignments with identical epipolar
# distances.
# This demonstrates why marker placement matters. Even if individual
# markers are localized accurately in both X-ray images, certain 3D
# arrangements can still make their identities impossible to resolve
# uniquely from the two views.
# ## Part 10 — Marker Reconstruction Error Analysis
# Marker localization in an X-ray image is not perfectly accurate.
# An error in the 2D marker location changes the back-projection ray
# and can therefore change the reconstructed 3D marker position.
# In this simulation, Marker Localization Error (MLE) values from
# 0 mm to 10 mm were tested.
# For each MLE, the marker position in Detector A was shifted by
# exactly that distance in directions from 0° to 359°. Detector B
# was independently shifted through the same 360 directions.
# This gives:
# 360 × 360 = 129,600
# direction combinations for every marker at every MLE.
# All three markers were tested, so each MLE required:
# 3 × 129,600 = 388,800
# reconstructions.
# For every reconstruction, two quantities were calculated:
# MRE:
# distance between the reconstructed 3D marker location and the
# known true CK marker location.
# REM:
# shortest distance between the two back-projection lines.
# The maximum clinically acceptable MRE was defined as 5.0 mm.
# The main questions were:
# 1. What is the largest MLE that guarantees MRE remains below
# 5 mm for every marker and every tested direction?
# 2. Does REM increase reliably enough with MRE to act as a warning
# metric for unsafe reconstructions?

import json

with open(

    "../data/error_analysis_summary.json",

    "r"

) as f:

    error_results = json.load(f)

mle_summary = error_results[

    "MLE_summary"

]

print(

    f"{'MLE':<8}"

    f"{'Mean MRE':<16}"

    f"{'Worst MRE':<16}"

    f"{'Safe for all directions':<24}"

)

for result in mle_summary:

    print(

        f"{result['MLE_mm']:<8}"

        f"{result['mean_MRE_mm']:<16.4f}"

        f"{result['worst_MRE_mm']:<16.4f}"

        f"{str(result['guaranteed_below_5_mm']):<24}"

    )

print()

print(

    "Maximum guaranteed-safe MLE:",

    error_results["maxMLE_mm"],

    "mm"

)

rem_results = error_results[

    "REM_analysis"

]

print()

print(

    "MRE/REM correlation:",

    rem_results[

        "correlation_MRE_REM"

    ]

)

print(

    "Minimum REM among unsafe cases:",

    rem_results[

        "minimum_REM_dangerous_mm"

    ],

    "mm"

)

print(

    "Maximum REM among safe cases:",

    rem_results[

        "maximum_REM_safe_mm"

    ],

    "mm"

)

print(

    "Perfect REM threshold exists:",

    rem_results[

        "perfect_REM_threshold_exists"

    ]

)

mle_mre_plot = plt.imread(

    "../data/MLE_vs_MRE.png"

)

plt.figure(figsize=(9, 7))

plt.imshow(

    mle_mre_plot

)

plt.axis("off")

plt.title(

    "Marker Localization Error vs Marker Reconstruction Error"

)

plt.show()

rem_mre_plot = plt.imread(

    "../data/REM_vs_MRE.png"

)

plt.figure(figsize=(9, 7))

plt.imshow(

    rem_mre_plot

)

plt.axis("off")

plt.title(

    "Residual Error Metric vs Marker Reconstruction Error"

)

plt.show()

# ### Findings
# The reconstruction error increased steadily as marker localization
# error increased. At MLE = 1 mm, the worst MRE over all markers and
# error directions was about 0.71 mm. At MLE = 7 mm, the worst MRE
# increased to 4.96 mm.
# MLE = 7 mm was still safe according to the 5 mm reconstruction
# error limit because every marker and every tested pair of error
# directions remained below 5 mm.
# At MLE = 8 mm, the worst reconstruction error increased to about
# 5.66 mm. Therefore 8 mm cannot guarantee a clinically acceptable
# reconstruction.
# Based on the tested integer MLE values, the maximum localization
# error that guarantees MRE < 5 mm is therefore:
# maxMLE = 7 mm
# The direction of the localization error was important. For the same
# MLE magnitude, some combinations of directions produced much larger
# reconstruction errors than others. This is why testing only one
# direction would not have been enough. The exhaustive 360 × 360
# direction search was needed to find the true worst-case behaviour.
# The REM results were more surprising. REM and MRE had a correlation
# of only about 0.31, so REM increased somewhat with reconstruction
# error overall, but the relationship was not strong enough to make
# REM a reliable warning measure by itself.
# Most importantly, some unsafe reconstructions with MRE >= 5 mm had
# REM values extremely close to zero. The minimum REM among unsafe
# cases was about 0.000002 mm. At the same time, some reconstructions
# that were still safe had REM values as large as about 9.90 mm.
# This means there is a large overlap between the REM values of safe
# and unsafe reconstructions. There is therefore no single REM
# threshold that can perfectly separate acceptable and unacceptable
# reconstructions.
# The reason is that REM measures how closely the two back-projection
# rays approach each other, not how close their intersection is to the
# true marker position. Two localization errors can move both rays in
# a compatible way so that they still intersect almost perfectly, but
# at the wrong 3D location. In that situation REM can be nearly zero
# even though MRE is clinically unacceptable.
# The threshold tests show the same problem. A very small REM
# threshold catches most dangerous cases, but it also incorrectly
# flags a very large number of safe reconstructions. Increasing the
# threshold reduces false alarms, but then an increasing fraction of
# dangerous reconstructions is missed.
# Overall, MLE has a clear effect on reconstruction accuracy, and the
# worst-case analysis gives a maximum tested safe value of 7 mm.
# REM provides some information about geometric inconsistency between
# the two rays, but it should not be used alone as a guarantee that
# the reconstructed marker position is accurate.

# author: Serhat
