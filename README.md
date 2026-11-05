# PointCloudProcessing-CSF

English | [中文](README.zh-CN.md)

[![CI](https://github.com/smiler488/PointCloudProcessing-CSF/actions/workflows/test.yml/badge.svg)](https://github.com/smiler488/PointCloudProcessing-CSF/actions/workflows/test.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) ![Python 3.9–3.12](https://img.shields.io/badge/Python_3.9%E2%80%933.12-informational)
Batch extraction of row-level canopy traits from UAV point clouds of row-crop plots. Ground points are removed with the Cloth Simulation Filter (CSF), the plot is levelled, the canopy is cleaned and split into rows, and height, volume and area traits are written to one CSV for all plots.

This is the hands-off sibling of [CanopyPC](https://github.com/smiler488/CanopyPC). Both share the same processing code. CanopyPC is for working through one plot with viewers and manual clean-up. This repository is for running a whole trial overnight without anything waiting for a click.

## Installation

Python 3.9 to 3.12.

```bash
git clone https://github.com/smiler488/PointCloudProcessing-CSF.git
cd PointCloudProcessing-CSF
pip install -r requirements.txt
```

The `CSF` module is installed from the `cloth-simulation-filter` package. The package called `CSF` on PyPI is unrelated.

## Usage

```bash
# every .ply in a folder, three rows per plot, planting direction along (1, 2.3)
python process_point_cloud.py plots/ --rows 3 --direction 1 2.3 -o row_traits.csv

# a few files, with Open3D windows to check each step
python process_point_cloud.py site_02.ply site_09.ply --show
```

| Option | Default | Notes |
|---|---|---|
| `--rows` | 3 | rows per plot |
| `--direction DX DY` | estimated | planting direction in the XY plane; estimated from the row/gap pattern if omitted |
| `--cloth-resolution` | 0.1 | CSF grid size in cloud units. Increase on rough or sloping ground |
| `--show` | off | open viewers for each step |
| `--manual` | off | enable the manual point-removal window |
| `-o` | `row_traits.csv` | output table |

If one plot fails (for example, too few ground points), the error is printed and the run moves on to the next file.

## Output

One line per row and plot:

`plot, row_id, num_points, height_max, height_min, height_mean, plant_height, convex_hull_volume, minimal_cube_volume, volume_ratio, row_projected_area, canopy_area_ratio, obb_extent_x, obb_extent_y, obb_extent_z`

- Heights are above the fitted ground plane (z = 0 at the lowest ground point).
- `convex_hull_volume` is computed after closing the row base at its lowest point.
- `volume_ratio` = hull volume / cube on the longest oriented-bounding-box edge.
- `row_projected_area` is the area of the row's 2-D convex hull in the XY plane.
- `canopy_area_ratio` = row projected area × number of rows / plot projected area.

## Parameter notes

The defaults come from my wheat and maize plots reconstructed at roughly 1 cm point spacing:

- outlier removal: 20 neighbours, 2.0 standard deviations;
- small clusters: DBSCAN `eps = 0.05`, clusters under 100 points dropped;
- plot labels: DBSCAN `eps = 0.1`, clusters with hull surface area under 0.2 dropped.

Sparser clouds need larger `eps` values. These are function arguments in `process_point_cloud.py`.

## Changes in this revision

- Projected areas were roughly doubled (surface area of a flattened 3-D hull). They now use the 2-D convex hull.
- Clouds could end up upside down when the fitted ground normal pointed down. Fixed.
- The convex-hull volume now closes the row base, as in CanopyPC.
- Batch input, CSV output and a command line replace the hard-coded file path.
- The README described the run as fully automatic while the code still opened a manual window. Manual clean-up is now opt-in.
- Added the missing `requirements.txt`.
- The planting direction is estimated from the row/gap pattern instead of PCA, which cut across rows on near-square plots.

## Citation

```bibtex
@software{deng_pcp_csf,
  author = {Deng, Liangchao},
  title  = {PointCloudProcessing-CSF: batch canopy traits from UAV point clouds},
  year   = {2025},
  url    = {https://github.com/smiler488/PointCloudProcessing-CSF}
}
```

CSF: Zhang, W. et al. (2016). An easy-to-use airborne LiDAR data filtering method based on cloth simulation. *Remote Sensing*, 8(6), 501.

## License

MIT, see [LICENSE](LICENSE). Liangchao Deng (邓良超).
