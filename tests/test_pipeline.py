"""Tests for the geometry helpers and an end-to-end run on a synthetic plot."""
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import process_point_cloud as P  # noqa: E402


def synthetic_plot(path, seed=0):
    """6 m x 4 m tilted ground with three rows of 0.05-0.6 m tall 'plants' along x."""
    rng = np.random.default_rng(seed)
    g = rng.uniform([0, 0], [6, 4], (40000, 2))
    tilt = lambda x, y: 0.05 * x + 0.02 * y
    pts = [np.column_stack([g, tilt(g[:, 0], g[:, 1]) + rng.normal(0, 0.005, len(g))])]
    for y0 in (1, 2, 3):
        x = rng.uniform(0.3, 5.7, 30000)
        y = y0 + rng.normal(0, 0.1, 30000)
        pts.append(np.column_stack([x, y, tilt(x, y) + rng.uniform(0.05, 0.6, 30000)]))
    o3d.io.write_point_cloud(str(path), o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack(pts))))


def test_rotation_handles_antiparallel():
    R = P.get_rotation_matrix_from_vectors([0, 0, -1], [0, 0, 1])
    assert np.allclose(R @ [0, 0, -1], [0, 0, 1])
    assert np.isclose(np.linalg.det(R), 1)


def test_projected_area_is_2d_area():
    square = np.array([[0, 0, 5], [2, 0, 1], [2, 3, 0], [0, 3, 2]], float)
    assert np.isclose(P.projected_area_xy(square), 6.0)


def test_mesh_volume_of_unit_cube():
    cube = o3d.geometry.TriangleMesh.create_box(1, 1, 1)
    assert np.isclose(P.compute_mesh_volume(cube), 1.0)


def test_batch_folder(tmp_path):
    plots = tmp_path / "plots"
    plots.mkdir()
    synthetic_plot(plots / "plotA.ply", seed=0)
    synthetic_plot(plots / "plotB.ply", seed=1)
    out = tmp_path / "all.csv"
    subprocess.run([sys.executable, str(ROOT / "process_point_cloud.py"), str(plots),
                    "--direction", "1", "0", "-o", str(out)],
                   check=True, capture_output=True)
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 6
    assert {r["plot"] for r in rows} == {"plotA", "plotB"}
    for r in rows:
        assert 0.4 < float(r["plant_height"]) < 0.6
        assert 1.8 < float(r["row_projected_area"]) < 3.2


def test_row_direction_on_square_plot():
    """Three short, wide rows at 70 deg in a square plot: PCA picks the wrong axis."""
    rng = np.random.default_rng(1)
    t = np.deg2rad(70)
    along, across = np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])
    pts = []
    for k in (-1, 0, 1):
        s = rng.uniform(-1.2, 1.2, 20000)
        w = k * 0.9 + rng.normal(0, 0.12, 20000)
        pts.append(np.outer(s, along) + np.outer(w, across))
    xy = np.vstack(pts)
    est = P.estimate_row_direction(xy)
    assert abs(abs(est @ along) - 1) < 1 - np.cos(np.deg2rad(1.0))
    _, _, vt = np.linalg.svd(xy - xy.mean(0), full_matrices=False)
    assert abs(vt[0] @ along) < 0.9  # PCA alone would be off here
