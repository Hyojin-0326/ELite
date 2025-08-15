import os
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
import matplotlib.cm as cm
from typing import List, Optional


class SessionMap():
    def __init__(self, map: np.ndarray = None, eph: np.ndarray = None):
        self.map = None if map is None else map
        self.eph = None if eph is None else self._clamp_eph(eph, 1e-1, 1)
        if self.map is not None:
            self.kdtree = KDTree(self.map)

    def set_poses(self, poses: List[np.ndarray]):
        assert isinstance(poses, list), "poses must be a list of numpy arrays"
        assert all(isinstance(p, np.ndarray) for p in poses), "all poses must be numpy arrays"
        assert all(p.shape == (4, 4) for p in poses), "all poses must be 4x4 matrices"
        assert len(poses) > 0, "poses list cannot be empty"
        self.poses = poses
    
    def get_poses(self):
        return self.poses
    
    def load_poses(self, pose_path: str):
        # check existence of file
        if not os.path.exists(pose_path):
            raise FileNotFoundError(f"File {pose_path} does not exist.")
        poses = []
        with open(pose_path, 'r') as f:
            for line in f:
                vals = list(map(float, line.strip().split()))
                if len(vals) != 12:
                    raise ValueError(f"Invalid pose line: {line.strip()}")
                pose = np.array(vals).reshape(3, 4)
                pose = np.vstack([pose, [0, 0, 0, 1]])
                poses.append(pose)
        self.set_poses(poses)
        return poses

    def _clamp_eph(self, eph, min_eph = 0, max_eph = 1):
        eph = np.where(eph < min_eph, min_eph, eph)
        eph = np.where(eph > max_eph, max_eph, eph)
        return eph
    
    def _multiply_eph(self, eph, factor = 1.1):
        eph = np.clip(eph * factor, 0, 1)
        return eph
    
    def get(self):
        session_map = o3d.geometry.PointCloud()
        session_map.points = o3d.utility.Vector3dVector(self.map)
        if self.eph is None:
            session_map.paint_uniform_color([0.5, 0.5, 0.5])
        else:
            # save ephemerality as colors
            colors = np.zeros((self.map.shape[0], 3))
            colors[:, 0] = self.eph
            session_map.colors = o3d.utility.Vector3dVector(colors)
    
        return session_map
    
    def visualize(self):
        session_map = self.get()
        if self.eph is None:
            session_map.paint_uniform_color([0.5, 0.5, 0.5])
        else:
            colormap = cm.get_cmap('jet')
            colors = colormap(self.eph)[:, :3]
            session_map.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([session_map])

    def save(self, path: str, is_global: bool):
        if is_global:
            session_map_points_path = os.path.join(path, "lifelong_map.pcd")
            session_map_eph_path = os.path.join(path, "global_ephemerality.npy")
        else:
            session_map_points_path = os.path.join(path, "cleaned_session_map.pcd")
            session_map_eph_path = os.path.join(path, "local_ephemerality.npy")
        
        # save the map as a point cloud
        if self.map is None:
            raise ValueError("Session map is None. Cannot save.")
        else:
            session_map = o3d.geometry.PointCloud()
            session_map.points = o3d.utility.Vector3dVector(self.map)
            o3d.io.write_point_cloud(session_map_points_path, session_map)

        # save the ephemerality as a numpy array
        if self.eph is None:
            raise ValueError("Ephemerality is None. Cannot save.")
        else:
            np.save(session_map_eph_path, self.eph)

    def load(self, path: str, is_global: bool):
        if is_global:
            session_map_points_path = os.path.join(path, "lifelong_map.pcd")
            session_map_eph_path = os.path.join(path, "global_ephemerality.npy")
            pose_path = os.path.join(path, "rev_final_transform.txt")
        else:
            session_map_points_path = os.path.join(path, "cleaned_session_map.pcd")
            session_map_eph_path = os.path.join(path, "local_ephemerality.npy")
            pose_path = None

        if not os.path.exists(session_map_points_path):
            raise FileNotFoundError(f"File {session_map_points_path} does not exist.")
        session_map = o3d.io.read_point_cloud(session_map_points_path)
        self.map = np.asarray(session_map.points)
        self.kdtree = KDTree(self.map)

        if not os.path.exists(session_map_eph_path):
            raise FileNotFoundError(f"File {session_map_eph_path} does not exist.")
        eph = np.load(session_map_eph_path)
        self.eph = self._clamp_eph(eph, 1e-1, 1)

        if pose_path is not None:
            self.load_poses(pose_path)    
    
    # TODO: add __repr__ method for better debugging