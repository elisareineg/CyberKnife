import numpy as np

from transforms import (
    md_to_ck,
    ck_to_md,
    ck_to_detector,
    detector_to_ck,
    detector_to_image,
    image_to_detector,
    get_detector,
    ck_to_detector_matrix,
    detector_to_ck_matrix,
)

from utils import apply

def test_md_ck_identity():
    """Verify that aligned MD and CK frames preserve coordinates."""

    test_points = [
        [0.0, 0.0, 0.0],
        [25.0, -10.0, 40.0],
        [-19.0, 28.0, 11.0],
    ]

    for point in test_points:
        expected = np.asarray(point, dtype=float)

        actual_ck = md_to_ck(point)
        actual_md = ck_to_md(point)

        assert np.allclose(actual_ck, expected)
        assert np.allclose(actual_md, expected)

    print("MD <-> CK tests passed!")

def test_ck_to_detector_a():
    """Verify CK-to-detector coordinates using known geometric points."""

    detector = get_detector("A")

    center = detector["center"]
    u_axis = detector["u"]
    v_axis = detector["v"]

    assert np.allclose(
        ck_to_detector(center, "A"),
        [0.0, 0.0, 0.0]
    )

    test_point = center + 10.0 * u_axis + 20.0 * v_axis

    assert np.allclose(
        ck_to_detector(test_point, "A"),
        [10.0, 20.0, 0.0]
    )

def test_detector_transforms():
    """Test forward, reverse, and round-trip transforms for both detectors."""

    for view in ("A", "B"):
        detector = get_detector(view)

        center = detector["center"]
        source = detector["source"]
        u = detector["u"]
        v = detector["v"]

        # Detector centre is the local origin.
        assert np.allclose(
            ck_to_detector(center, view),
            [0.0, 0.0, 0.0]
        )

        # Source lies 2000 mm along the detector's +w axis.
        assert np.allclose(
            ck_to_detector(source, view),
            [0.0, 0.0, 2000.0]
        )

        # Known displacement along the detector's local axes.
        test_point = center + 10.0 * u + 20.0 * v

        assert np.allclose(
            ck_to_detector(test_point, view),
            [10.0, 20.0, 0.0]
        )

        # Reverse transformation: detector origin -> CK centre.
        assert np.allclose(
            detector_to_ck([0.0, 0.0, 0.0], view),
            center
        )

        # Reverse transformation: detector source -> CK source.
        assert np.allclose(
            detector_to_ck([0.0, 0.0, 2000.0], view),
            source
        )

        # Round-trip: CK -> Detector -> CK.
        recovered = detector_to_ck(
            ck_to_detector(test_point, view),
            view
        )

        assert np.allclose(recovered, test_point)

def test_detector_to_image():
    """Check predicted detector-to-image pixel coordinates."""

    cases = [
        ([0.0, 0.0, 0.0], [1000.0, 1000.0]),
        ([10.0, -20.0, 0.0], [1100.0, 800.0]),
        ([-50.0, 50.0, 0.0], [500.0, 1500.0]),
    ]

    for detector_point, expected_image in cases:
        actual = detector_to_image(detector_point)

        assert np.allclose(actual, expected_image)


def test_image_to_detector():
    """Check predicted image-to-detector coordinates."""

    cases = [
        ([1000.0, 1000.0], [0.0, 0.0, 0.0]),
        ([1100.0, 800.0], [10.0, -20.0, 0.0]),
        ([500.0, 1500.0], [-50.0, 50.0, 0.0]),
    ]

    for image_point, expected_detector in cases:
        actual = image_to_detector(image_point)

        assert np.allclose(actual, expected_detector)


def test_image_detector_round_trip():
    """Verify image-to-detector-to-image coordinate consistency."""

    image_point = np.array([1250.0, 750.0])

    detector_point = image_to_detector(image_point)
    recovered_image = detector_to_image(detector_point)

    assert np.allclose(recovered_image, image_point)

def test_detector_matrix_consistency():
    """Verify direct transforms agree with the public matrix interfaces."""
    for view in ("A", "B"):
        detector = get_detector(view)

        ck_point = np.array([15.0, -25.0, 40.0])
        detector_point = np.array([10.0, 20.0, 30.0])

        # CK -> Detector
        direct_result = ck_to_detector(ck_point, view)
        matrix_result = apply(
            ck_to_detector_matrix(detector),
            ck_point
        )

        assert np.allclose(direct_result, matrix_result)

        # Detector -> CK
        direct_result = detector_to_ck(detector_point, view)
        matrix_result = apply(
            detector_to_ck_matrix(detector),
            detector_point
        )

        assert np.allclose(direct_result, matrix_result)

def test_image_to_detector_with_w():
    """Verify that the optional detector depth coordinate is preserved."""

    image_point = [1100.0, 800.0]

    result = image_to_detector(image_point, w=25.0)

    expected = [10.0, -20.0, 25.0]

    assert np.allclose(result, expected)

def test_detector_centers():
    """Verify detector centres against the specified CK geometry."""

    expected_a = [
        1000.0 / np.sqrt(2),
        -1000.0 / np.sqrt(2),
        0.0,
    ]

    expected_b = [
        -1000.0 / np.sqrt(2),
        -1000.0 / np.sqrt(2),
        0.0,
    ]

    assert np.allclose(get_detector("A")["center"], expected_a)
    assert np.allclose(get_detector("B")["center"], expected_b)

def test_detector_to_ck_known_point():
    """Check detector-to-CK transforms using analytically predicted points."""

    diagonal = 1000.0 / np.sqrt(2)
    offset = 20.0 / np.sqrt(2)

    expected_points = {
        "A": [diagonal + offset, -diagonal + offset, 10.0],
        "B": [-diagonal + offset, -diagonal - offset, 10.0],
    }

    for view, expected in expected_points.items():
        actual = detector_to_ck([10.0, 20.0, 0.0], view)

        assert np.allclose(actual, expected, atol=1e-8)