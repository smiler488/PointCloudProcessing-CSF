# PointCloudProcessing-CSF

PointCloudProcessing-CSF is a Python-based pipeline for processing UAV-acquired point cloud data. It leverages the Cloth Simulation Filter (CSF) for ground (soil) segmentation, and then applies automated noise removal, clustering, and canopy segmentation into rows. The pipeline also quantifies key metrics for each row, including canopy volume ratio, canopy area ratio, and plant height.

## Features

- **Ground Segmentation with CSF:**  
  Uses the Cloth Simulation Filter to separate ground (soil) points from canopy points.

- **Point Cloud Alignment:**  
  Computes the ground plane from the segmented soil points and rotates/translates the point cloud so that the ground is leveled (parallel to the XY plane).

- **Noise and Small Cluster Removal:**  
  Applies statistical outlier removal and DBSCAN clustering to remove isolated noise and small clusters (e.g., remaining signboard points).

- **Row Segmentation:**  
  Projects the cleaned canopy points onto the XY plane and uses KMeans clustering to segment the canopy into three rows based on the planting direction.

- **Quantitative Metrics:**  
  For each row, the following metrics are computed:
  - **Canopy Volume Ratio:** The ratio of the convex hull volume of the row's canopy points to the volume of its minimal cube (derived from the axis-aligned bounding box).
  - **Canopy Area Ratio:** The ratio of the row's projected canopy area to one-third of the projected area of the original canopy candidate cloud.
  - **Plant Height:** The difference between the maximum and minimum heights of the row's canopy points.

## Installation

### Prerequisites

- Python 3.6 or higher

### Dependencies

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

Usage
	1.	Place your point cloud file (in PLY format) in the repository directory.
	2.	Modify the file_path variable in the main script (process_point_cloud.py) to point to your PLY file.
	3.	Optionally, adjust parameters (e.g., CSF settings, DBSCAN eps, KMeans planting direction) as needed.
	4.	Run the script:
 
```bash
python process_point_cloud.py
```

During execution, several Open3D visualization windows will display intermediate results (e.g., raw point cloud, segmented soil and canopy, aligned point cloud, row segmentation with convex hulls, etc.). The final output includes quantitative metrics for each canopy row.

File Structure

•	process_point_cloud.py: Main Python script containing the complete processing pipeline.
 
•	README.md: This project description file.
 
•	requirements.txt: A list of the required Python packages.

Notes

•	Parameter Tuning:The pipeline parameters (CSF parameters, DBSCAN eps and min_cluster_size, KMeans settings, etc.) might need tuning depending on your specific dataset and the quality of the point cloud.

•	Automation vs. Manual Correction:
This version removes any manual deletion (point selection) steps. All filtering and segmentation are performed automatically.
