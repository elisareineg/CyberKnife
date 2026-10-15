import json

import matplotlib.pyplot as plt
import numpy as np

from marker_transforms import (
    MARKERS_MD,
    MARKERS_CK,
    transform_markers,
    save_results,
    plot_markers_ck,
)
from transforms import detector_to_ck



def test_marker_coordinates():
    """Verify the assignment's marker coordinates are preserved."""

    expected_markers = {
        "M1": [-19.0, 28.0, 11.0],
        "M2": [43.0, 14.0, 2.0],
        "M3": [2.0, 48.0, -6.0],
    }

    for name, expected in expected_markers.items():
        assert np.allclose(MARKERS_MD[name], expected)
        assert np.allclose(MARKERS_CK[name], expected)


def test_marker_transform_structure():
    """Verify all markers are transformed into the expected frames."""

    results = transform_markers()

    assert set(results) == {"M1", "M2", "M3"}

    expected_frames = {
        "MD",
        "CK",
        "Detector_A_3D",
        "Detector_B_3D",
    }

    for name, coordinates in results.items():
        assert set(coordinates) == expected_frames

        for point in coordinates.values():
            assert np.asarray(point).shape == (3,)

def test_marker_detector_coordinates():
    """Verify detector coordinates against the specified 45-degree geometry."""

    results = transform_markers()
    root_two = np.sqrt(2.0)

    for name, marker in MARKERS_MD.items():
        x, y, z = marker

        expected_a = [
            z,
            (x + y) / root_two,
            1000.0 + (-x + y) / root_two,
        ]

        expected_b = [
            z,
            (x - y) / root_two,
            1000.0 + (x + y) / root_two,
        ]

        assert np.allclose(
            results[name]["Detector_A_3D"],
            expected_a,
            atol=1e-8,
        )

        assert np.allclose(
            results[name]["Detector_B_3D"],
            expected_b,
            atol=1e-8,
        )

def test_marker_detector_round_trip():
    """Verify marker positions can be recovered from both detector frames."""

    results = transform_markers()

    for name, coordinates in results.items():
        original_ck = coordinates["CK"]

        for view in ("A", "B"):
            detector_point = coordinates[f"Detector_{view}_3D"]

            recovered_ck = detector_to_ck(detector_point, view)

            assert np.allclose(
                recovered_ck,
                original_ck,
                atol=1e-8,
            )

def test_save_results(tmp_path):
    """Verify marker transformations are correctly saved as JSON."""

    results = transform_markers()
    output_file = tmp_path / "marker_transforms.json"

    save_results(results, str(output_file))

    assert output_file.exists()

    with open(output_file, "r") as f:
        saved_results = json.load(f)

    assert set(saved_results) == set(results)

    for name, coordinates in results.items():
        assert set(saved_results[name]) == set(coordinates)

        for frame, expected in coordinates.items():
            assert np.allclose(
                saved_results[name][frame],
                expected,
                atol=1e-8,
            )

def test_custom_marker_input():
    """Verify custom markers can be transformed without changing the input."""

    custom_markers = {
        "Test": np.array([10.0, 20.0, 30.0])
    }

    results = transform_markers(custom_markers)

    assert set(results) == {"Test"}

    assert np.allclose(
        results["Test"]["CK"],
        [10.0, 20.0, 30.0]
    )

    assert np.allclose(
        custom_markers["Test"],
        [10.0, 20.0, 30.0]
    )

    # The returned MD coordinates should not alias the input.
    assert not np.shares_memory(
        results["Test"]["MD"],
        custom_markers["Test"]
    )

def test_save_results_current_directory(tmp_path, monkeypatch):
    """Verify JSON export works without a directory in the filename."""

    monkeypatch.chdir(tmp_path)

    save_results(
        transform_markers(),
        "markers.json"
    )

    assert (tmp_path / "markers.json").exists()


def test_plot_markers_ck(tmp_path, monkeypatch):
    """Verify the marker visualization is saved successfully."""

    monkeypatch.setattr(plt, "show", lambda: None)

    output_file = tmp_path / "markers.png"

    plot_markers_ck(
        transform_markers(),
        str(output_file)
    )

    assert output_file.exists()
    assert output_file.stat().st_size > 0

# Author: Akshay