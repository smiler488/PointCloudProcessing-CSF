# PointCloudProcessing-CSF

[English](README.md) | 中文

[![CI](https://github.com/smiler488/PointCloudProcessing-CSF/actions/workflows/test.yml/badge.svg)](https://github.com/smiler488/PointCloudProcessing-CSF/actions/workflows/test.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) ![Python 3.9–3.12](https://img.shields.io/badge/Python_3.9%E2%80%933.12-informational)
从行播作物小区的无人机点云里批量提取按行统计的冠层性状。处理流程依次是：用布料模拟滤波（CSF）去地面、调平、去噪、分行，最后把所有小区的株高、体积、面积性状汇总到同一个 CSV。

它和 [CanopyPC](https://github.com/smiler488/CanopyPC) 是一对，处理代码相同。CanopyPC 适合单个小区边看边调、手动清理；这个仓库适合整个试验的数据丢进去跑一晚上，中间不需要人点任何东西。

## 安装

Python 3.9–3.12。

```bash
git clone https://github.com/smiler488/PointCloudProcessing-CSF.git
cd PointCloudProcessing-CSF
pip install -r requirements.txt
```

`CSF` 模块由 `cloth-simulation-filter` 包提供。PyPI 上叫 `CSF` 的包是另一个东西。

## 使用

```bash
# 处理文件夹内所有 .ply，每个小区 3 行，播种方向 (1, 2.3)
python process_point_cloud.py plots/ --rows 3 --direction 1 2.3 -o row_traits.csv

# 处理几个指定文件，并打开 Open3D 窗口逐步检查
python process_point_cloud.py site_02.ply site_09.ply --show
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--rows` | 3 | 每个小区的行数 |
| `--direction DX DY` | 自动估计 | XY 平面内的播种方向；不指定时根据行和行间空隙的分布自动估计 |
| `--cloth-resolution` | 0.1 | CSF 网格大小（点云单位），地面起伏大或有坡度时调大 |
| `--show` | 关 | 每一步弹出可视化窗口 |
| `--manual` | 关 | 启用手动删点窗口 |
| `-o` | `row_traits.csv` | 输出表格 |

某个小区处理失败时（比如地面点太少），会打印错误信息，然后继续处理下一个文件。

## 输出

每个小区的每一行占一条记录：

`plot, row_id, num_points, height_max, height_min, height_mean, plant_height, convex_hull_volume, minimal_cube_volume, volume_ratio, row_projected_area, canopy_area_ratio, obb_extent_x, obb_extent_y, obb_extent_z`

- 高度均相对拟合的地面平面（最低地面点为 z = 0）。
- `convex_hull_volume` 是在该行最低点处封底后计算的凸包体积。
- `volume_ratio` = 凸包体积 / 以有向包围盒最长边为边长的立方体体积。
- `row_projected_area` 是该行在 XY 平面上二维凸包的面积。
- `canopy_area_ratio` = 行投影面积 × 行数 / 小区投影面积。

## 参数经验

默认参数来自我的小麦和玉米小区，重建点间距约 1 cm：

- 离群点：20 个邻域点，2.0 倍标准差；
- 小簇：DBSCAN `eps = 0.05`，少于 100 个点的簇删除；
- 标牌：DBSCAN `eps = 0.1`，凸包表面积小于 0.2 的簇删除。

点云更稀疏时要把 `eps` 调大。这些都是 `process_point_cloud.py` 里各函数的参数。

## 本次修订

- 投影面积之前约为真实值的 2 倍（取的是压扁三维凸包的表面积），现改用二维凸包。
- 拟合出的地面法向量朝下时点云会被上下翻转，已修正。
- 凸包体积改为封底后计算，与 CanopyPC 一致。
- 用批量输入、CSV 输出和命令行参数替代原来写死的文件路径。
- 原 README 说流程全自动，代码里却还会弹出手动删点窗口。现在手动清理改为可选，默认关闭。
- 补上之前缺失的 `requirements.txt`。
- 播种方向改为根据行和行间空隙的分布来估计，不再用 PCA。PCA 在接近正方形的小区上会斜着切过行。

## 引用

```bibtex
@software{deng_pcp_csf,
  author = {Deng, Liangchao},
  title  = {PointCloudProcessing-CSF: batch canopy traits from UAV point clouds},
  year   = {2025},
  url    = {https://github.com/smiler488/PointCloudProcessing-CSF}
}
```

CSF：Zhang, W. et al. (2016). An easy-to-use airborne LiDAR data filtering method based on cloth simulation. *Remote Sensing*, 8(6), 501.

## 许可证

MIT，见 [LICENSE](LICENSE)。邓良超（Liangchao Deng）。
