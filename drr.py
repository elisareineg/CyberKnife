import os

import numpy as np
import matplotlib.pyplot as plt
import trimesh

from geometry import (
    DETECTOR_A,
    DETECTOR_B,
    DETECTOR_SIZE
)

from marker_transforms import MARKERS_CK


# ============================================================
# PART 5: DIGITALLY RECONSTRUCTED RADIOGRAPHS
# ============================================================


# ------------------------------------------------------------
# Physical / simulation parameters
# ------------------------------------------------------------

# Assignment specifies a homogeneous sphere of radius 5.0 cm.
SPHERE_CENTER = np.array([0.0, 0.0, 0.0])
SPHERE_RADIUS = 50.0       # mm

# Assignment markers are 3 mm in diameter.
MARKER_RADIUS = 1.5        # mm

# Arbitrary source intensity.
# Intensities are normalized so that an unobstructed ray has
# intensity 1.0.
I0 = 1.0

# Effective attenuation coefficients [1/mm].
#
# These values are used for visualization rather than as a
# clinical calibration.
#
# The marker coefficient is intentionally much larger than the
# vertebra coefficient so that the metal markers appear clearly.
MU_VERTEBRA = 0.015
MU_MARKER = 1.0


# ============================================================
# RAY / SPHERE INTERSECTION
# ============================================================

def ray_sphere_path_length(source, directions, center, radius):
    """
    Compute the distance travelled by each X-ray through a sphere.

    Parameters
    ----------
    source : ndarray, shape (3,)
        X-ray source position in CK coordinates.

    directions : ndarray, shape (..., 3)
        Unit direction vectors of rays leaving the source.

    center : ndarray, shape (3,)
        Sphere center in CK coordinates.

    radius : float
        Sphere radius in mm.

    Returns
    -------
    lengths : ndarray, shape (...)
        Length of each ray segment inside the sphere in mm.

    Method
    ------
    A ray can be written as:

        X(t) = S + t*d

    where:

        S = source
        d = unit ray direction

    A sphere satisfies:

        |X - C|^2 = R^2

    Substituting the ray into the sphere equation gives a
    quadratic equation. If the discriminant is negative, the
    ray misses the sphere. Otherwise, the two roots represent
    the entry and exit positions.

    Because d is normalized, the difference between the two
    ray parameters is directly a distance in millimetres.
    """

    source = np.asarray(source, dtype=float)
    center = np.asarray(center, dtype=float)

    # Vector from sphere centre to the X-ray source.
    m = source - center

    # For a normalized direction:
    #
    # t^2 + 2(m dot d)t + (m dot m - R^2) = 0
    b = np.sum(directions * m, axis=-1)
    c = np.dot(m, m) - radius ** 2

    discriminant = b ** 2 - c

    # Rays with negative discriminant miss the sphere.
    hit = discriminant >= 0.0

    sqrt_disc = np.zeros_like(discriminant)
    sqrt_disc[hit] = np.sqrt(discriminant[hit])

    t1 = -b - sqrt_disc
    t2 = -b + sqrt_disc

    # Only intersections in front of the source are relevant.
    entry = np.maximum(t1, 0.0)
    exit_ = np.maximum(t2, 0.0)

    lengths = np.zeros_like(discriminant)

    valid = hit & (exit_ > entry)

    lengths[valid] = exit_[valid] - entry[valid]

    return lengths


# ============================================================
# DETECTOR RAY GENERATION
# ============================================================

def make_detector_rays(detector, resolution):
    """
    Generate one ray from the X-ray source to every sampled
    detector location.

    The physical detector is always 200 mm x 200 mm.

    A smaller computational resolution can be used during
    development. For example:

        resolution = 400

    produces a 400 x 400 DRR even though the full detector is
    defined as 2000 x 2000 pixels.
    """

    source = detector["source"]
    center = detector["center"]

    u_axis = detector["u"]
    v_axis = detector["v"]

    # Physical size represented by each simulated pixel.
    pixel_size = DETECTOR_SIZE / resolution

    # Sample the centre of each detector pixel.
    coordinates = (
        (np.arange(resolution) + 0.5) * pixel_size
        - DETECTOR_SIZE / 2.0
    )

    U, V = np.meshgrid(
        coordinates,
        coordinates,
        indexing="ij"
    )

    # Convert detector coordinates to CK coordinates.
    #
    # X = detector_center + u*u_axis + v*v_axis
    detector_points = (
        center
        + U[..., None] * u_axis
        + V[..., None] * v_axis
    )

    # Ray from source to detector point.
    rays = detector_points - source

    # Normalize the directions.
    #
    # This is important because the ray parameter will then
    # correspond directly to distance in millimetres.
    norms = np.linalg.norm(
        rays,
        axis=-1,
        keepdims=True
    )

    directions = rays / norms

    return directions


# ============================================================
# PHASE 1: HOMOGENEOUS SPHERE DRR
# ============================================================

def generate_sphere_drr(detector, resolution=400):
    """
    Generate one DRR of the homogeneous vertebra-equivalent
    sphere containing three spherical metal markers.

    Steps
    -----
    1. Generate source-to-detector rays.
    2. Compute path length through the large sphere.
    3. Compute path length through each metal marker.
    4. Add attenuation contributions.
    5. Apply the Beer-Lambert law.
    """

    source = detector["source"]

    directions = make_detector_rays(
        detector,
        resolution
    )

    # --------------------------------------------------------
    # Large homogeneous sphere
    # --------------------------------------------------------

    vertebra_length = ray_sphere_path_length(
        source,
        directions,
        SPHERE_CENTER,
        SPHERE_RADIUS
    )

    total_attenuation = (
        MU_VERTEBRA * vertebra_length
    )

    # --------------------------------------------------------
    # Metal markers
    # --------------------------------------------------------

    for marker_name, marker_center in MARKERS_CK.items():

        marker_length = ray_sphere_path_length(
            source,
            directions,
            marker_center,
            MARKER_RADIUS
        )

        total_attenuation += (
            MU_MARKER * marker_length
        )

    # --------------------------------------------------------
    # Beer-Lambert attenuation
    #
    # I = I0 * exp(-sum(mu_i * L_i))
    # --------------------------------------------------------

    image = I0 * np.exp(
        -total_attenuation
    )

    return image


def save_sphere_drr(
    image,
    view_name,
    output_directory="data"
):
    """
    Save the sphere DRR as both:

        .npy
            Raw floating-point intensity values.

        .png
            Visualization for qualitative inspection.
    """

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    npy_filename = os.path.join(
        output_directory,
        f"sphere_drr_{view_name}.npy"
    )

    png_filename = os.path.join(
        output_directory,
        f"sphere_drr_{view_name}.png"
    )

    np.save(
        npy_filename,
        image
    )

    plt.figure(figsize=(7, 7))

    # High transmission appears light.
    # Strong attenuation appears dark.
    plt.imshow(
        image,
        cmap="gray",
        origin="lower",
        vmin=0.0,
        vmax=1.0
    )

    plt.title(
        f"Sphere DRR - Detector {view_name}"
    )

    plt.xlabel("Detector v pixel")
    plt.ylabel("Detector u pixel")

    plt.colorbar(
        label="Normalized transmitted intensity"
    )

    plt.tight_layout()

    plt.savefig(
        png_filename,
        dpi=200
    )

    plt.close()

    print(f"Saved {npy_filename}")
    print(f"Saved {png_filename}")


# ============================================================
# QUALITATIVE IMAGE CHECK
# ============================================================

def describe_image(image, name):
    """
    Print simple statistics for qualitative checking.
    """

    print(name)

    print(
        "  shape:",
        image.shape
    )

    print(
        "  minimum intensity:",
        image.min()
    )

    print(
        "  maximum intensity:",
        image.max()
    )

    print(
        "  mean intensity:",
        image.mean()
    )

    print()


# ============================================================
# PHASE 2: ACTUAL VERTEBRA STL + THREE METAL MARKERS
# ============================================================

STL_FILENAME = "LumbarVertebrae.-witrh-Markers.stl"


def find_stl_file():
    """
    Locate the supplied lumbar vertebra STL.

    The function checks both:

        data/LumbarVertebrae.-witrh-Markers.stl

    and:

        LumbarVertebrae.-witrh-Markers.stl

    This allows the STL to be placed either in the data folder
    or directly in the repository root.
    """

    possible_paths = [
        os.path.join(
            "data",
            STL_FILENAME
        ),
        STL_FILENAME
    ]

    for path in possible_paths:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        "\nCould not find the vertebra STL.\n"
        "\nExpected one of:\n"
        f"  data/{STL_FILENAME}\n"
        f"  {STL_FILENAME}\n"
    )


def load_vertebra_mesh():
    """
    Load the lumbar vertebra STL.

    The supplied STL contains several connected components,
    including the vertebra and marker spheres.

    For attenuation purposes, the largest connected component
    is treated as the vertebra.

    The three metal markers are simulated separately using
    MARKERS_CK so that bone and metal can have different
    attenuation coefficients.
    """

    filename = find_stl_file()

    print(
        f"Loading STL: {filename}"
    )

    mesh = trimesh.load(
        filename,
        force="mesh"
    )

    # Split the STL into connected components.
    components = mesh.split(
        only_watertight=False
    )

    if len(components) == 0:
        raise ValueError(
            "No mesh components were found in the STL."
        )

    print(
        "Number of connected components:",
        len(components)
    )

    # The vertebra should be the largest connected component.
    vertebra = max(
        components,
        key=lambda component: len(component.faces)
    )

    print()
    print("Loaded vertebra STL")
    print(
        "  vertices:",
        len(vertebra.vertices)
    )

    print(
        "  faces:",
        len(vertebra.faces)
    )

    print(
        "  watertight:",
        vertebra.is_watertight
    )

    print("  bounds:")
    print(vertebra.bounds)

    print("  centroid:")
    print(vertebra.centroid)

    print()

    return vertebra


# ============================================================
# RAY / TRIANGLE-MESH INTERSECTION
# ============================================================

def mesh_ray_path_lengths(
    mesh,
    source,
    directions,
    batch_size=10000
):
    """
    Compute how much distance every X-ray travels inside the
    vertebra triangle mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Vertebra surface mesh.

    source : ndarray, shape (3,)
        X-ray source position in CK coordinates.

    directions : ndarray, shape (..., 3)
        Normalized X-ray directions.

    batch_size : int
        Number of rays processed at one time.

    Returns
    -------
    lengths : ndarray
        Total distance travelled through the vertebra for every
        ray, in millimetres.

    Method
    ------
    For a simple closed object, a ray normally enters the object
    once and exits once:

        source -> entry ===== object ===== exit -> detector

    Therefore:

        path length = exit - entry

    A complicated object can produce several pairs:

        entry1, exit1, entry2, exit2, ...

    In that case all interior segments are added together.

    trimesh performs the ray / triangle intersection test.
    """

    original_shape = directions.shape[:-1]

    directions_flat = directions.reshape(
        -1,
        3
    )

    number_of_rays = len(
        directions_flat
    )

    lengths_flat = np.zeros(
        number_of_rays,
        dtype=float
    )

    # --------------------------------------------------------
    # Process rays in batches.
    #
    # Sending every ray at once can require too much memory,
    # especially when the DRR resolution is increased.
    # --------------------------------------------------------

    for start in range(
        0,
        number_of_rays,
        batch_size
    ):

        end = min(
            start + batch_size,
            number_of_rays
        )

        batch_directions = directions_flat[
            start:end
        ]

        batch_origins = np.repeat(
            np.asarray(source, dtype=float)[None, :],
            len(batch_directions),
            axis=0
        )

        # Find all intersections between rays and mesh triangles.
        locations, ray_indices, _ = (
            mesh.ray.intersects_location(
                ray_origins=batch_origins,
                ray_directions=batch_directions,
                multiple_hits=True
            )
        )

        # Nothing in this batch intersected the vertebra.
        if len(locations) == 0:
            continue

        # Since ray directions are normalized, Euclidean distance
        # from the source is also the ray parameter t.
        distances = np.linalg.norm(
            locations
            - batch_origins[ray_indices],
            axis=1
        )

        # Sort first by ray index and then by distance from source.
        order = np.lexsort(
            (
                distances,
                ray_indices
            )
        )

        ray_indices = ray_indices[order]
        distances = distances[order]

        unique_rays = np.unique(
            ray_indices
        )

        # ----------------------------------------------------
        # Convert surface intersections into material lengths.
        # ----------------------------------------------------

        for ray_index in unique_rays:

            hit_distances = distances[
                ray_indices == ray_index
            ]

            hit_distances = np.sort(
                hit_distances
            )

            # A closed mesh should normally have an even number
            # of crossings.
            #
            # If numerical issues give an odd number, ignore the
            # final unmatched crossing.
            if len(hit_distances) % 2 != 0:
                hit_distances = hit_distances[:-1]

            path_length = 0.0

            for i in range(
                0,
                len(hit_distances),
                2
            ):

                entry = hit_distances[i]
                exit_ = hit_distances[i + 1]

                if exit_ > entry:
                    path_length += (
                        exit_ - entry
                    )

            lengths_flat[
                start + ray_index
            ] = path_length

        print(
            f"  Processed rays "
            f"{start + 1} to {end} "
            f"of {number_of_rays}"
        )

    return lengths_flat.reshape(
        original_shape
    )


# ============================================================
# VERTEBRA DRR
# ============================================================

def generate_vertebra_drr(
    detector,
    vertebra_mesh,
    resolution=200
):
    """
    Generate a DRR of the actual vertebra STL plus the three
    spherical metal markers.

    Steps
    -----
    1. Generate rays from the X-ray source to the detector.
    2. Intersect each ray with the vertebra mesh.
    3. Calculate total path length through the vertebra.
    4. Calculate path length through each spherical marker.
    5. Add attenuation contributions.
    6. Apply Beer-Lambert attenuation.
    """

    source = detector["source"]

    directions = make_detector_rays(
        detector,
        resolution
    )

    # --------------------------------------------------------
    # Vertebra attenuation
    # --------------------------------------------------------

    vertebra_length = mesh_ray_path_lengths(
        vertebra_mesh,
        source,
        directions
    )

    total_attenuation = (
        MU_VERTEBRA
        * vertebra_length
    )

    # --------------------------------------------------------
    # Metal markers
    #
    # We use the known assignment marker locations and radius
    # rather than treating the marker components in the STL as
    # bone.
    # --------------------------------------------------------

    for marker_name, marker_center in MARKERS_CK.items():

        marker_length = ray_sphere_path_length(
            source,
            directions,
            marker_center,
            MARKER_RADIUS
        )

        total_attenuation += (
            MU_MARKER
            * marker_length
        )

    # --------------------------------------------------------
    # Beer-Lambert law
    #
    # I = I0 * exp(-sum(mu_i * L_i))
    # --------------------------------------------------------

    image = I0 * np.exp(
        -total_attenuation
    )

    return image


def save_vertebra_drr(
    image,
    view_name,
    output_directory="data"
):
    """
    Save raw and visualization versions of a vertebra DRR.
    """

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    npy_filename = os.path.join(
        output_directory,
        f"vertebra_drr_{view_name}.npy"
    )

    png_filename = os.path.join(
        output_directory,
        f"vertebra_drr_{view_name}.png"
    )

    np.save(
        npy_filename,
        image
    )

    plt.figure(
        figsize=(7, 7)
    )

    plt.imshow(
        image,
        cmap="gray",
        origin="lower",
        vmin=0.0,
        vmax=1.0
    )

    plt.title(
        f"Vertebra DRR - Detector {view_name}"
    )

    plt.xlabel(
        "Detector v pixel"
    )

    plt.ylabel(
        "Detector u pixel"
    )

    plt.colorbar(
        label="Normalized transmitted intensity"
    )

    plt.tight_layout()

    plt.savefig(
        png_filename,
        dpi=200
    )

    plt.close()

    print(
        f"Saved {npy_filename}"
    )

    print(
        f"Saved {png_filename}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "\nPART 5 - DIGITALLY RECONSTRUCTED RADIOGRAPHS\n"
    )

    # ========================================================
    # PHASE 1: HOMOGENEOUS SPHERE
    # ========================================================

    SPHERE_RESOLUTION = 400

    print(
        "PHASE 1 - HOMOGENEOUS SPHERE\n"
    )

    print(
        f"Generating Detector A sphere DRR "
        f"({SPHERE_RESOLUTION} x {SPHERE_RESOLUTION})..."
    )

    sphere_A = generate_sphere_drr(
        DETECTOR_A,
        SPHERE_RESOLUTION
    )

    print(
        f"Generating Detector B sphere DRR "
        f"({SPHERE_RESOLUTION} x {SPHERE_RESOLUTION})..."
    )

    sphere_B = generate_sphere_drr(
        DETECTOR_B,
        SPHERE_RESOLUTION
    )

    print()

    describe_image(
        sphere_A,
        "Sphere Detector A"
    )

    describe_image(
        sphere_B,
        "Sphere Detector B"
    )

    save_sphere_drr(
        sphere_A,
        "A"
    )

    save_sphere_drr(
        sphere_B,
        "B"
    )

    # ========================================================
    # PHASE 2: ACTUAL VERTEBRA STL
    # ========================================================

    print(
        "\nPHASE 2 - ACTUAL VERTEBRA STL\n"
    )

    vertebra = load_vertebra_mesh()

    # Mesh intersection is significantly more expensive than
    # the analytic sphere intersection, so start at 200 x 200.
    VERTEBRA_RESOLUTION = 200

    print(
        f"Generating Detector A vertebra DRR "
        f"({VERTEBRA_RESOLUTION} x "
        f"{VERTEBRA_RESOLUTION})..."
    )

    vertebra_A = generate_vertebra_drr(
        DETECTOR_A,
        vertebra,
        VERTEBRA_RESOLUTION
    )

    print(
        f"Generating Detector B vertebra DRR "
        f"({VERTEBRA_RESOLUTION} x "
        f"{VERTEBRA_RESOLUTION})..."
    )

    vertebra_B = generate_vertebra_drr(
        DETECTOR_B,
        vertebra,
        VERTEBRA_RESOLUTION
    )

    print()

    describe_image(
        vertebra_A,
        "Vertebra Detector A"
    )

    describe_image(
        vertebra_B,
        "Vertebra Detector B"
    )

    save_vertebra_drr(
        vertebra_A,
        "A"
    )

    save_vertebra_drr(
        vertebra_B,
        "B"
    )

    print(
        "\nDRR generation complete."
    )

    
# author: Serhat