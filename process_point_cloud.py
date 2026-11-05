"""
PointCloudProcessing-CSF - automatic, batch version of the CanopyPC pipeline.

Same processing steps as CanopyPC, but nothing waits for the user: the
manual clean-up window is off by default and a whole folder of plots can be
processed in one run, with all row traits collected in a single CSV.

Pipeline
--------
1. Ground / canopy separation with the Cloth Simulation Filter (CSF).
2. Levelling: fit a plane to the ground points, rotate its normal onto +Z
   and shift the lowest ground point to z = 0.
3. Canopy cleaning: statistical outlier removal, small-cluster removal and
   removal of compact objects such as plot labels.
4. Optional manual clean-up in an Open3D editing window.
5. Row segmentation by K-means on the axis perpendicular to the planting
   direction, followed by per-row trait extraction.

Units follow the input cloud (metres for a georeferenced reconstruction).

Author: Liangchao Deng (邓良超)
License: MIT
"""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import numpy as np
import open3d as o3d
from matplotlib import colormaps
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull, QhullError
from sklearn.cluster import KMeans

try:
    import CSF  # provided by `pip install cloth-simulation-filter`
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "The CSF module is missing. Install it with `pip install cloth-simulation-filter`."
    ) from exc


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def get_rotation_matrix_from_vectors(vec1, vec2):
    """Return the 3x3 matrix that rotates ``vec1`` onto ``vec2`` (Rodrigues).

    The parallel and anti-parallel cases are handled explicitly; the plain
    formula is undefined when the two vectors point in opposite directions.
    """
    a = np.asarray(vec1, dtype=float)
    b = np.asarray(vec2, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)

    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # 180 degree turn about any axis orthogonal to a
        ortho = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, ortho)
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)

    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)


def projected_area_xy(points):
    """Area of the 2-D convex hull of the XY projection.

    For 2-D input ``scipy.spatial.ConvexHull.volume`` is the enclosed area.
    Earlier versions took the surface area of a flattened 3-D hull, which
    counts the top and bottom faces and therefore doubled the value.
    """
    xy = np.asarray(points)[:, :2]
    if xy.shape[0] < 3:
        return 0.0
    try:
        return float(ConvexHull(xy).volume)
    except QhullError:
        return 0.0


def points_in_polygon(xy, polygon_xy):
    """Boolean mask of ``xy`` points inside ``polygon_xy`` (vectorised)."""
    return MplPath(polygon_xy).contains_points(xy)


def create_lineset_from_mesh(mesh, color=(1, 0, 0)):
    """Wireframe of a triangle mesh, used to draw convex hulls."""
    line_set = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    line_set.paint_uniform_color(color)
    return line_set


def create_oobb_lineset(oobb, color=(0, 0, 1)):
    """Wireframe of an oriented bounding box."""
    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(oobb)
    line_set.paint_uniform_color(color)
    return line_set


def compute_mesh_volume(mesh):
    """Volume of a closed triangle mesh from the divergence theorem."""
    vertices = np.asarray(mesh.vertices)
    tri = np.asarray(mesh.triangles)
    p1, p2, p3 = vertices[tri[:, 0]], vertices[tri[:, 1]], vertices[tri[:, 2]]
    return abs(np.einsum("ij,ij->i", p1, np.cross(p2, p3)).sum()) / 6.0


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def remove_noise(pcd, nb_neighbors=20, std_ratio=2.0):
    """Statistical outlier removal."""
    filtered, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return filtered


def remove_cluster_noise(pcd, eps=0.05, min_cluster_size=100, min_points=10):
    """Drop DBSCAN clusters with fewer than ``min_cluster_size`` points."""
    labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    if labels.size == 0:
        return pcd
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    keep = unique[counts >= min_cluster_size]
    return pcd.select_by_index(np.flatnonzero(np.isin(labels, keep)).tolist())


def remove_landmark_sign(pcd, eps=0.1, min_points=20, area_threshold=0.2):
    """Remove compact clusters (plot labels, markers).

    Clusters whose convex-hull surface area is below ``area_threshold``
    (cloud units squared) are dropped.
    """
    labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    keep = []
    for label in np.unique(labels):
        if label < 0:
            continue
        idx = np.flatnonzero(labels == label)
        try:
            hull, _ = pcd.select_by_index(idx.tolist()).compute_convex_hull()
            area = hull.get_surface_area()
        except RuntimeError:
            area = 0.0  # too few or coplanar points: treat as clutter
        if area >= area_threshold:
            keep.append(idx)
    if not keep:
        return pcd.select_by_index([])
    return pcd.select_by_index(np.concatenate(keep).tolist())


def csf_soil_segmentation(pcd, cloth_resolution=0.5, slope_smooth=False,
                          rigidness=3, time_step=0.65, class_threshold=0.12,
                          iterations=500):
    """Split a cloud into ground and non-ground points with CSF.

    Parameter names follow Zhang et al. (2016), Remote Sensing 8(6):501.

    Returns
    -------
    (soil_cloud, canopy_cloud)
    """
    csf = CSF.CSF()
    csf.setPointCloud(np.asarray(pcd.points))
    csf.params.bSloopSmooth = slope_smooth
    csf.params.cloth_resolution = cloth_resolution
    csf.params.rigidness = rigidness
    csf.params.time_step = time_step
    csf.params.class_threshold = class_threshold
    csf.params.interations = iterations  # spelling used by the CSF binding

    ground, non_ground = CSF.VecInt(), CSF.VecInt()
    csf.do_filtering(ground, non_ground, exportCloth=False)
    return pcd.select_by_index(list(ground)), pcd.select_by_index(list(non_ground))


def manual_remove_points(pcd):
    """Interactive clean-up in an Open3D editing window.

    Shift + left click picks points, ``q`` closes the window.
    * 3 or more picks: every point inside the XY convex hull of the picks is removed.
    * 1-2 picks: only the picked points are removed.
    """
    print("Shift + click to pick points, press 'q' when done.")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Manual clean-up (Shift+click, q to finish)")
    vis.add_geometry(pcd)
    vis.run()
    picked = vis.get_picked_points()
    vis.destroy_window()

    if not picked:
        print("No points picked; cloud unchanged.")
        return pcd

    pts = np.asarray(pcd.points)
    if len(picked) >= 3:
        sel = pts[picked, :2]
        try:
            hull = ConvexHull(sel)
            mask = points_in_polygon(pts[:, :2], sel[hull.vertices])
            print(f"Removing {int(mask.sum())} points inside the picked region.")
            return pcd.select_by_index(np.flatnonzero(mask).tolist(), invert=True)
        except QhullError as err:
            print(f"Picked points are collinear ({err}); removing them one by one.")

    print(f"Removing {len(picked)} picked points.")
    return pcd.select_by_index(picked, invert=True)


# ---------------------------------------------------------------------------
# Row segmentation and traits
# ---------------------------------------------------------------------------

def estimate_row_direction(xy, bin_size=0.02, coarse_step=1.0, fine_step=0.1):
    """Estimate the planting direction (unit vector in the XY plane).

    Points are projected onto the axis perpendicular to each candidate
    direction and binned. When the candidate is parallel to the rows, the
    histogram shows sharp peaks (rows) and empty bins (inter-row gaps), so
    the variance of the bin counts is largest. A coarse 1 degree scan is
    refined to 0.1 degree. Unlike PCA, this does not depend on the plot
    shape, which matters for square plots or plots with few, long rows.
    """
    xy = np.asarray(xy, dtype=float)
    xy = xy - xy.mean(axis=0)

    def score(theta_deg):
        t = np.deg2rad(theta_deg)
        proj = xy @ np.array([-np.sin(t), np.cos(t)])
        bins = np.arange(proj.min(), proj.max() + bin_size, bin_size)
        counts, _ = np.histogram(proj, bins=bins)
        return counts.var()

    coarse = np.arange(0.0, 180.0, coarse_step)
    best = coarse[np.argmax([score(a) for a in coarse])]
    fine = np.arange(best - coarse_step, best + coarse_step + 1e-9, fine_step)
    best = fine[np.argmax([score(a) for a in fine])]
    t = np.deg2rad(best)
    return np.array([np.cos(t), np.sin(t)])


def segment_major_rows(plot_cloud, n_rows=3, planting_direction=None):
    """Split a canopy cloud into ``n_rows`` rows.

    Points are projected onto the axis perpendicular to the planting
    direction and clustered with 1-D K-means. If ``planting_direction`` is
    None it is estimated with :func:`estimate_row_direction`.

    Returns
    -------
    rows : list of (row_id, PointCloud), ordered along the separation axis
    labels : ndarray of row ids per point, or None for an empty cloud
    """
    points = np.asarray(plot_cloud.points)
    if points.shape[0] == 0:
        return [], None
    xy = points[:, :2]

    if planting_direction is None:
        row_direction = estimate_row_direction(xy)
        print(f"Estimated planting direction: ({row_direction[0]:.3f}, {row_direction[1]:.3f})")
    else:
        row_direction = np.asarray(planting_direction, dtype=float)
        row_direction = row_direction / np.linalg.norm(row_direction)

    separation_axis = np.array([-row_direction[1], row_direction[0]])
    projections = xy @ separation_axis

    kmeans = KMeans(n_clusters=n_rows, n_init=10, random_state=0)
    labels = kmeans.fit_predict(projections.reshape(-1, 1))

    # Relabel so that row 0 has the smallest mean projection
    order = np.argsort([projections[labels == i].mean() for i in range(n_rows)])
    remap = np.empty(n_rows, dtype=int)
    remap[order] = np.arange(n_rows)
    labels = remap[labels]

    rows = [(i, plot_cloud.select_by_index(np.flatnonzero(labels == i).tolist()))
            for i in range(n_rows)]
    return rows, labels


def fill_lower_surface(row_cloud, num_points=1000, fill_z=None):
    """Close the bottom of a row cloud so its convex hull reaches the ground.

    A regular grid of about ``num_points`` points is laid inside the XY
    convex hull of the row at height ``fill_z`` (default: the row minimum).
    """
    points = np.asarray(row_cloud.points)
    if fill_z is None:
        fill_z = points[:, 2].min()
    xy = points[:, :2]

    hull = ConvexHull(xy)
    polygon = xy[hull.vertices]
    minx, miny = polygon.min(axis=0)
    maxx, maxy = polygon.max(axis=0)
    step = np.sqrt(hull.volume / float(num_points))

    gx, gy = np.meshgrid(np.arange(minx, maxx, step), np.arange(miny, maxy, step))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    inside = grid[points_in_polygon(grid, polygon)]
    if inside.shape[0] == 0:
        inside = polygon[:1]

    fill = np.column_stack([inside, np.full(inside.shape[0], fill_z)])
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(np.vstack([points, fill]))
    return out


def row_traits(row_id, row_pc, plot_projected_area, n_rows):
    """Per-row trait dictionary and the row's oriented bounding box.

    volume_ratio       convex-hull volume / cube built on the longest OBB edge
    canopy_area_ratio  (row projected area * n_rows) / whole-plot projected area
    plant_height       max z - min z inside the row (ground is at z = 0)
    """
    pts = np.asarray(row_pc.points)

    try:
        hull, _ = fill_lower_surface(row_pc).compute_convex_hull()
        hull_volume = float(compute_mesh_volume(hull))
    except (RuntimeError, QhullError) as err:
        print(f"Row {row_id}: convex hull volume failed ({err}).")
        hull_volume = None

    oobb = row_pc.get_oriented_bounding_box()
    dims = np.asarray(oobb.extent)
    cube_volume = float(dims.max() ** 3)
    row_area = projected_area_xy(pts)

    traits = {
        "row_id": row_id,
        "num_points": int(pts.shape[0]),
        "height_max": float(pts[:, 2].max()),
        "height_min": float(pts[:, 2].min()),
        "height_mean": float(pts[:, 2].mean()),
        "plant_height": float(np.ptp(pts[:, 2])),
        "convex_hull_volume": hull_volume,
        "minimal_cube_volume": cube_volume,
        "volume_ratio": float(hull_volume / cube_volume) if hull_volume and cube_volume > 0 else None,
        "row_projected_area": row_area,
        "canopy_area_ratio": row_area * n_rows / plot_projected_area if plot_projected_area > 0 else None,
        "obb_extent_x": float(dims[0]),
        "obb_extent_y": float(dims[1]),
        "obb_extent_z": float(dims[2]),
    }
    return traits, oobb


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _show(geoms, title, enabled):
    if enabled:
        o3d.visualization.draw_geometries(geoms, window_name=title)


def process_point_cloud(file_path, planting_direction=None, n_rows=3,
                        cloth_resolution=0.1, visualize=False, interactive=False):
    """Run the full CanopyPC pipeline on one ``.ply`` file.

    Parameters
    ----------
    file_path : str or Path
        Input point cloud.
    planting_direction : sequence of two floats, optional
        Row direction in the XY plane, e.g. ``[1, 2.3]``. Estimated from the
        point distribution if omitted (see :func:`estimate_row_direction`).
    n_rows : int
        Number of rows in the plot.
    cloth_resolution : float
        CSF cloth grid size in cloud units.
    visualize : bool
        Open an Open3D window after each step.
    interactive : bool
        Open the manual clean-up window.

    Returns
    -------
    dict with keys ``pcd_rotated``, ``soil_cloud_rotated``, ``canopy_final``,
    ``major_rows`` and ``all_results`` (list of trait dicts).
    """
    pcd = o3d.io.read_point_cloud(str(file_path))
    if len(pcd.points) == 0:
        raise ValueError(f"No points read from {file_path}")
    print(f"Loaded {len(pcd.points)} points from {file_path}")
    _show([pcd], "Original point cloud", visualize)

    # 1. Ground / canopy split
    soil, canopy = csf_soil_segmentation(pcd, cloth_resolution=cloth_resolution)
    print(f"Ground points: {len(soil.points)}; canopy candidates: {len(canopy.points)}")
    _show([soil], "Ground points", visualize)
    _show([canopy], "Canopy candidates", visualize)

    # 2. Levelling
    plane, _ = soil.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
    normal = np.asarray(plane[:3])
    if normal[2] < 0:
        normal = -normal  # point the normal upwards, otherwise the cloud ends up upside down
    rot = get_rotation_matrix_from_vectors(normal, [0, 0, 1])
    print("Ground plane ax+by+cz+d=0:", np.round(plane, 5))

    pcd_rot, soil_rot, canopy_rot = (copy.deepcopy(c) for c in (pcd, soil, canopy))
    for cloud in (pcd_rot, soil_rot, canopy_rot):
        cloud.rotate(rot, center=(0, 0, 0))
    shift = np.array([0.0, 0.0, -np.asarray(soil_rot.points)[:, 2].min()])
    for cloud in (pcd_rot, soil_rot, canopy_rot):
        cloud.translate(shift)
    _show([pcd_rot], "Levelled point cloud", visualize)

    # 3. Cleaning
    canopy_clean = remove_noise(canopy_rot)
    print("After outlier removal:", len(canopy_clean.points))
    canopy_clean = remove_cluster_noise(canopy_clean)
    print("After small-cluster removal:", len(canopy_clean.points))
    canopy_clean = remove_landmark_sign(canopy_clean)
    print("After label removal:", len(canopy_clean.points))
    _show([canopy_clean], "Filtered canopy", visualize)

    # 4. Manual clean-up
    if interactive:
        canopy_clean = manual_remove_points(canopy_clean)
        _show([canopy_clean], "Canopy after manual clean-up", visualize)

    # 5. Rows and traits
    plot_area = projected_area_xy(np.asarray(canopy_clean.points))
    print(f"Plot projected area: {plot_area:.4f}")
    rows, _ = segment_major_rows(canopy_clean, n_rows=n_rows, planting_direction=planting_direction)

    cmap = colormaps["Set1"]
    results, colored = [], []
    for row_id, row_pc in rows:
        if len(row_pc.points) < 4:
            print(f"Row {row_id}: too few points, skipped.")
            continue
        row_pc.paint_uniform_color(cmap(row_id % 9)[:3])
        colored.append(row_pc)

        traits, oobb = row_traits(row_id, row_pc, plot_area, n_rows)
        results.append(traits)
        print(f"Row {row_id}:", {k: (round(v, 4) if isinstance(v, float) else v)
                                 for k, v in traits.items()})

        if visualize:
            geoms = [row_pc, create_oobb_lineset(oobb)]
            try:
                hull, _ = row_pc.compute_convex_hull()
                geoms.append(create_lineset_from_mesh(hull))
            except RuntimeError:
                pass
            _show(geoms, f"Row {row_id}: convex hull and OBB", True)

    _show(colored, "Row segmentation", visualize)

    return {
        "pcd_rotated": pcd_rot,
        "soil_cloud_rotated": soil_rot,
        "canopy_final": canopy_clean,
        "major_rows": rows,
        "all_results": results,
    }


def write_results_csv(results, out_path):
    """Write a list of trait dicts to CSV."""
    if not results:
        return
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def collect_inputs(paths):
    """Expand files and directories into a sorted list of .ply files."""
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files.extend(sorted(p.glob("*.ply")))
        elif p.suffix.lower() == ".ply":
            files.append(p)
        else:
            print(f"Skipping {p}: not a .ply file or folder")
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Batch row-level canopy traits from UAV point clouds (CSF ground filter).")
    parser.add_argument("inputs", nargs="+", help=".ply files and/or folders containing .ply files")
    parser.add_argument("--rows", type=int, default=3,
                        help="number of rows per plot (default: 3)")
    parser.add_argument("--direction", type=float, nargs=2, metavar=("DX", "DY"),
                        help="planting direction in the XY plane, e.g. --direction 1 2.3")
    parser.add_argument("--cloth-resolution", type=float, default=0.1,
                        help="CSF cloth resolution in cloud units (default: 0.1)")
    parser.add_argument("--show", action="store_true",
                        help="open Open3D windows for each step (one plot at a time)")
    parser.add_argument("--manual", action="store_true",
                        help="enable the manual clean-up window (implies --show)")
    parser.add_argument("-o", "--output", default="row_traits.csv",
                        help="output CSV (default: row_traits.csv)")
    args = parser.parse_args(argv)

    files = collect_inputs(args.inputs)
    if not files:
        parser.error("no .ply input found")

    table = []
    for f in files:
        print(f"\n=== {f.name} ===")
        try:
            res = process_point_cloud(
                f,
                planting_direction=args.direction,
                n_rows=args.rows,
                cloth_resolution=args.cloth_resolution,
                visualize=args.show or args.manual,
                interactive=args.manual,
            )
        except Exception as err:  # keep going with the other plots
            print(f"{f.name}: failed ({err})")
            continue
        for row in res["all_results"]:
            table.append({"plot": f.stem, **row})

    write_results_csv(table, args.output)
    print(f"\n{len(table)} rows from {len(files)} file(s) written to {args.output}")


if __name__ == "__main__":
    main()
