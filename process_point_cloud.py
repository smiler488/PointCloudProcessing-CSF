import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.cluster import KMeans
import os
import tempfile
import CSF

def get_rotation_matrix_from_vectors(vec1, vec2):
    """
    计算旋转矩阵，将 vec1 旋转到 vec2 的方向
    """
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s**2 + 1e-10))
    return R

def create_lineset_from_mesh(mesh):
    """
    根据三角网格生成凸包边界线，用于可视化凸包
    """
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    edges = set()
    for tri in triangles:
        edges.add(tuple(sorted((tri[0], tri[1]))))
        edges.add(tuple(sorted((tri[1], tri[2]))))
        edges.add(tuple(sorted((tri[2], tri[0]))))
    edges = list(edges)
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(vertices),
        lines=o3d.utility.Vector2iVector(edges)
    )
    line_set.paint_uniform_color([1, 0, 0])  # 红色显示凸包
    return line_set

def create_oobb_lineset(oobb):
    """
    利用 Open3D 自带的 LineSet.create_from_oriented_bounding_box
    直接创建 OBB 线框，避免手动连线顺序错误。
    """
    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(oobb)
    line_set.paint_uniform_color([0, 0, 1])  # 统一给 OBB 线框涂成蓝色
    return line_set

def remove_noise(pcd, nb_neighbors=20, std_ratio=2.0):
    """
    利用统计离群点剔除方法去除噪音
    """
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return cl

def remove_cluster_noise(pcd, eps=0.05, min_cluster_size=100):
    """
    利用 DBSCAN 聚类去除小群噪声点。
    只有聚类内点数大于 min_cluster_size 的保留，其余剔除。
    """
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=10, print_progress=True))
    unique_labels, counts = np.unique(labels, return_counts=True)
    good_labels = [label for label, count in zip(unique_labels, counts)
                   if label != -1 and count >= min_cluster_size]
    indices = [i for i, lab in enumerate(labels) if lab in good_labels]
    filtered_pcd = pcd.select_by_index(indices)
    return filtered_pcd

def remove_landmark_sign(pcd, eps=0.1, min_points=20, area_threshold=0.2):
    """
    利用 DBSCAN 聚类和凸包面积过滤，移除那些小团的点云（如地标牌点云）。
    低于阈值 area_threshold 的聚类认为为地标牌，将其删除。
    """
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    unique_labels = np.unique(labels)
    good_indices = []
    for label in unique_labels:
        if label == -1:
            continue
        idx = np.where(labels == label)[0]
        cluster = pcd.select_by_index(idx)
        try:
            hull, _ = cluster.compute_convex_hull()
            area = hull.get_surface_area()
        except Exception as e:
            area = 0
        if area >= area_threshold:
            good_indices.extend(idx.tolist())
    filtered_pcd = pcd.select_by_index(good_indices)
    return filtered_pcd

def csf_soil_segmentation(pcd, cloth_resolution=0.5, bSloopSmooth=False):
    """
    利用 cloth-simulation-filter（CSF）方法对点云进行地面分割
    """
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    np.savetxt(tmp_file.name, np.asarray(pcd.points), fmt="%.6f")
    tmp_file.close()
    
    csf = CSF.CSF()
    csf.readPointsFromFile(tmp_file.name)
    
    csf.params.bSloopSmooth = bSloopSmooth
    csf.params.cloth_resolution = cloth_resolution
    csf.params.rigidness = 3
    csf.params.time_step = 0.65
    csf.params.class_threshold = 0.12
    csf.params.interations = 500

    ground = CSF.VecInt()
    non_ground = CSF.VecInt()
    csf.do_filtering(ground, non_ground)
    
    soil_cloud = pcd.select_by_index([int(i) for i in ground])
    canopy_cloud = pcd.select_by_index([int(i) for i in non_ground])
    
    os.remove(tmp_file.name)
    
    return soil_cloud, canopy_cloud

def segment_major_rows(plot_cloud, n_rows=3, planting_direction=None):
    """
    对单个小区内的冠层点云分割成 n_rows 行
    """
    points = np.asarray(plot_cloud.points)
    if points.shape[0] == 0:
        return [], None

    xy = points[:, :2]
    
    if planting_direction is None:
        mean_xy = np.mean(xy, axis=0)
        centered = xy - mean_xy
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        row_direction = Vt[0]
    else:
        row_direction = np.array(planting_direction, dtype=float)
        row_direction = row_direction / np.linalg.norm(row_direction)
    
    # 垂直于行方向
    separation_axis = np.array([-row_direction[1], row_direction[0]])
    separation_axis = separation_axis / np.linalg.norm(separation_axis)
    print("使用的分离轴（垂直于行方向）：", separation_axis)
    
    projections = xy.dot(separation_axis)
    
    kmeans = KMeans(n_clusters=n_rows, random_state=0).fit(projections.reshape(-1, 1))
    labels = kmeans.labels_
    
    cluster_means = [projections[labels == i].mean() for i in range(n_rows)]
    order = np.argsort(cluster_means)
    ordered_labels = np.zeros_like(labels)
    for new_label, orig_label in enumerate(order):
        ordered_labels[labels == orig_label] = new_label

    major_rows = []
    for i in range(n_rows):
        indices = np.where(ordered_labels == i)[0]
        row_pc = plot_cloud.select_by_index(indices)
        major_rows.append((i, row_pc))
    return major_rows, ordered_labels

def compute_mesh_volume(mesh):
    """
    使用散度定理计算三角网格的体积
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    volume = 0.0
    for tri in triangles:
        p1, p2, p3 = vertices[tri]
        volume += np.dot(p1, np.cross(p2, p3))
    return abs(volume) / 6.0

def manual_remove_points(pcd):
    """
    手动删除点：
      - 在弹出的窗口中，使用 Shift + 左键 点选要删除的点
      - 关闭窗口后会返回删除后剩余的点云
    """
    print("\n=========== 手动删除点功能 ===========")
    print("1) 请在弹出的窗口中按住 Shift + 左键，点击要删除的点")
    print("2) 选完后按 Q 或关闭窗口，即可完成选择\n")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="手动选择需要删除的点")
    vis.add_geometry(pcd)
    vis.run()  # 用户此时在窗口进行操作
    vis.destroy_window()

    picked_ids = vis.get_picked_points()
    if len(picked_ids) > 0:
        pcd_out = pcd.select_by_index(picked_ids, invert=True)
        print(f"成功删除选中的 {len(picked_ids)} 个点.")
        return pcd_out
    else:
        print("未选中任何点，无需删除.")
        return pcd


def process_point_cloud(file_path, planting_direction=None):
    """
    完整处理流程：
      1. 读取点云
      2. 利用 CSF 分割土壤点云
      3. 利用土壤点云计算地面平面, 旋转+平移校正整个点云
      4. 冠层去噪、聚类滤噪、小物体（地标牌）剔除
      5. **提供手动删除点功能**
      6. 分割为多大行并计算各项指标
    """
    # 1. 读取并显示原始点云
    pcd = o3d.io.read_point_cloud(file_path)
    print("加载点云，点数：", len(pcd.points))
    o3d.visualization.draw_geometries([pcd], window_name="原始点云")
    
    # 2. 利用 CSF 分割土壤和冠层
    soil_cloud, canopy_cloud = csf_soil_segmentation(pcd, cloth_resolution=0.1, bSloopSmooth=False)
    print("识别土壤点云数：", len(soil_cloud.points))
    print("初步冠层候选点云数：", len(canopy_cloud.points))
    o3d.visualization.draw_geometries([soil_cloud], window_name="土壤点云")
    o3d.visualization.draw_geometries([canopy_cloud], window_name="初步冠层候选点云")
    
    # 3. 地面平面计算+旋转平移校正
    plane_model, _ = soil_cloud.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
    print("地面平面模型: ax+by+cz+d =", plane_model)
    a, b, c, d_param = plane_model
    ground_normal = np.array([a, b, c])
    target_normal = np.array([0, 0, 1])
    R_mat = get_rotation_matrix_from_vectors(ground_normal, target_normal)
    print("旋转矩阵：\n", R_mat)
    
    pcd_rotated = copy.deepcopy(pcd)
    pcd_rotated.rotate(R_mat, center=(0, 0, 0))
    soil_cloud_rotated = copy.deepcopy(soil_cloud)
    canopy_cloud_rotated = copy.deepcopy(canopy_cloud)
    soil_cloud_rotated.rotate(R_mat, center=(0, 0, 0))
    canopy_cloud_rotated.rotate(R_mat, center=(0, 0, 0))

    # 将最低地面对齐到 z=0
    soil_points = np.asarray(soil_cloud_rotated.points)
    min_z = np.min(soil_points[:, 2])
    translation = np.array([0, 0, -min_z])
    pcd_rotated.translate(translation)
    soil_cloud_rotated.translate(translation)
    canopy_cloud_rotated.translate(translation)
    o3d.visualization.draw_geometries([pcd_rotated], window_name="校正后点云")
    
    # 4. 冠层去噪、小群聚类滤噪、去除地标牌
    canopy_clean = remove_noise(canopy_cloud_rotated, nb_neighbors=20, std_ratio=2.0)
    print("噪音剔除后冠层点云数：", len(canopy_clean.points))
    canopy_filtered = remove_cluster_noise(canopy_clean, eps=0.05, min_cluster_size=100)
    print("删除小群噪声后冠层点云数：", len(canopy_filtered.points))
    canopy_filtered = remove_landmark_sign(canopy_filtered, eps=0.1, min_points=20, area_threshold=0.2)
    print("去除地标牌后冠层点云数：", len(canopy_filtered.points))
    o3d.visualization.draw_geometries([canopy_filtered], window_name="自动过滤后冠层点云")

    # 5. 提供手动删除点功能（可选）
    canopy_filtered = manual_remove_points(canopy_filtered)
    o3d.visualization.draw_geometries([canopy_filtered], window_name="手动删除后冠层点云")

    # 6. 分割冠层为 n_rows 大行，并计算各大行指标
    canopy_final = canopy_filtered
    canopy_final_xy = np.asarray(canopy_final.points)[:, :2]
    epsilon = 1e-2
    random_noise = np.random.uniform(-1e-4, 1e-4, size=(canopy_final_xy.shape[0], 1))
    canopy_points_3d = np.hstack([canopy_final_xy, np.full((canopy_final_xy.shape[0], 1), epsilon) + random_noise])
    pcd_canopy_xy = o3d.geometry.PointCloud()
    pcd_canopy_xy.points = o3d.utility.Vector3dVector(canopy_points_3d)
    
    hull_canopy, _ = pcd_canopy_xy.compute_convex_hull()
    original_projected_area = hull_canopy.get_surface_area()
    print("最终冠层在 XY 平面的凸包面积：", original_projected_area)
    
    major_rows, row_labels = segment_major_rows(canopy_final, n_rows=3, planting_direction=planting_direction)
    cmap_row = plt.get_cmap("Set1")
    all_results = []
    colored_rows = []
    
    for row_id, row_pc in major_rows:
        color = cmap_row(row_id % 9)[:3]
        row_pc.paint_uniform_color(color)
        colored_rows.append(row_pc)
        pts = np.asarray(row_pc.points)
        if pts.size == 0:
            continue
        
        # 高度统计
        height_max = pts[:, 2].max()
        height_min = pts[:, 2].min()
        height_mean = pts[:, 2].mean()
        plant_height = height_max - height_min
        
        # 计算大行点云凸包体积
        try:
            row_hull, _ = row_pc.compute_convex_hull()
            convex_hull_volume = compute_mesh_volume(row_hull)
        except Exception:
            convex_hull_volume = None
        
        # 计算大行 OOBB，并基于此计算“最小立方体”体积
        oobb = row_pc.get_oriented_bounding_box()
        dims = oobb.extent  # OOBB 的长宽高
        side = np.max(dims)  
        minimal_cube_volume = side ** 3
        
        if convex_hull_volume is not None and minimal_cube_volume > 0:
            volume_ratio = convex_hull_volume / minimal_cube_volume
        else:
            volume_ratio = None
        
        # 计算大行在 XY 平面上的投影面积
        row_xy = np.asarray(row_pc.points)[:, :2]
        random_noise_row = np.random.uniform(-1e-4, 1e-4, size=(row_xy.shape[0], 1))
        row_points_3d = np.hstack([row_xy, np.full((row_xy.shape[0], 1), epsilon) + random_noise_row])
        pcd_row_xy = o3d.geometry.PointCloud()
        pcd_row_xy.points = o3d.utility.Vector3dVector(row_points_3d)
        try:
            row_proj_hull, _ = pcd_row_xy.compute_convex_hull()
            row_projected_area = row_proj_hull.get_surface_area()
        except Exception:
            row_projected_area = None
        
        if row_projected_area is not None and original_projected_area > 0:
            canopy_area_ratio = (row_projected_area * 3) / original_projected_area
        else:
            canopy_area_ratio = None
        
        
        result = {
            "row_id": row_id,
            "num_points": pts.shape[0],
            "height_max": height_max,
            "height_min": height_min,
            "height_mean": height_mean,
            "plant_height": plant_height,
            "convex_hull_volume": convex_hull_volume,
            "minimal_cube_volume": minimal_cube_volume,
            "volume_ratio": volume_ratio,
            "row_projected_area": row_projected_area,
            "canopy_area_ratio": canopy_area_ratio,
            "obb_extent_x": dims[0],
            "obb_extent_y": dims[1],
            "obb_extent_z": dims[2],
        }
        all_results.append(result)
        print(f"大行 {row_id} 指标：", result)
        
        # 可视化：OOBB（蓝色）+ 凸包（红色）
        oobb = row_pc.get_oriented_bounding_box()
        oobb_lineset = create_oobb_lineset(oobb)
        
        # 计算大行点云凸包
        try:
            row_hull, _ = row_pc.compute_convex_hull()
            convex_hull_volume = compute_mesh_volume(row_hull)
            # 在这里创建 hull_lineset
            hull_lineset = create_lineset_from_mesh(row_hull)
        except Exception:
            row_hull = None
            convex_hull_volume = None
            hull_lineset = None

        # 可视化时直接复用 hull_lineset
        oobb = row_pc.get_oriented_bounding_box()
        oobb_lineset = create_oobb_lineset(oobb)

        geometries = [row_pc]
        if hull_lineset is not None:
            geometries.append(hull_lineset)
        geometries.append(oobb_lineset)

        o3d.visualization.draw_geometries(
            geometries, 
            window_name=f"大行 {row_id} 与其凸包及OOBB"
        )
    
    # 总体显示三大行分割结果（不同颜色）
    o3d.visualization.draw_geometries(colored_rows, window_name="三大行分割结果")
    
    return {
        "pcd_rotated": pcd_rotated,
        "soil_cloud_rotated": soil_cloud_rotated,
        "canopy_final": canopy_final,
        "major_rows": major_rows,
        "all_results": all_results
    }

if __name__ == "__main__":
    file_path = "/Users/liangchaodeng/Documents/VScode/canopypc/site_20230623_09.ply"
    # 可以根据实际情况设置种植方向，例如 [1, 2.25]
    results = process_point_cloud(file_path, planting_direction=[1, 2.3])