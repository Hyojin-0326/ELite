import os
import yaml
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from tqdm import tqdm
import torch
import faiss
import pyflann

from utils.session_map import SessionMap

class FastKDTree:
    def __init__(self, data, num_trees=8, checks=64):
        # 입력이 torch면 numpy로 변환
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        self.flann = pyflann.FLANN()
        self.data = np.asarray(data, dtype=np.float32)
        # kdtree 여러 개로 근사 정확도 높이기
        self.params = self.flann.build_index(self.data, algorithm="kdtree", trees=num_trees)
        self.checks = checks

    def query(self, points, k=1):
        # 1) remember original device only if torch input
        orig_device = None
        if isinstance(points, torch.Tensor):
            orig_device = points.device
            points = points.detach().cpu().numpy()
        points = np.asarray(points, dtype=np.float32)

        # 2) flann query -> returns (idxs, d2)
        idxs, d2 = self.flann.nn_index(points, k, checks=self.checks)

        # 3) ensure 2D shape (N, k)
        if d2.ndim == 1 and k == 1:
            d2 = d2[:, None]
            idxs = idxs[:, None]

        # 4) if metric is L2 (squared), convert to Euclidean distance
        #    -> keep a flag in your class like self.squared = True if using L2
        if getattr(self, "squared", True):
            d = np.sqrt(np.maximum(d2, 0.0, dtype=np.float32)).astype(np.float32)
        else:
            d = d2.astype(np.float32)

        if orig_device is not None:
            return (
                torch.as_tensor(d, device=orig_device, dtype=torch.float32),
                torch.as_tensor(idxs, device=orig_device, dtype=torch.int64)
            )
        return d, idxs


class MapUpdater():
    def __init__(
            self,
            config_path: str
    ):
        # Load parameters
        with open(config_path, 'r') as f:
            self.params = yaml.safe_load(f)
        p_update = self.params["map_update"]
        self.voxel_size = p_update["voxel_size"]
        self.coexist_threshold = p_update["coexist_threshold"]
        self.overlap_threshold = p_update["overlap_threshold"]
        self.density_radius = p_update["density_radius"]
        self.rho_factor = p_update["rho_factor"]
        self.uncertainty_factor = p_update["uncertainty_factor"]
        self.global_eph_threshold = p_update["global_eph_threshold"]
        self.remove_dynamic_points = p_update["remove_dynamic_points"]
        self.remove_outlier_points = p_update["remove_outlier_points"]

        self.lifelong_map : SessionMap = None
        self.new_session_map : SessionMap = None


    def load(self, lifelong_map: SessionMap, new_session_map: SessionMap):
        self.lifelong_map = lifelong_map
        self.new_session_map = new_session_map


    #이거 토치로 바로 쌓을수도? 일단 모듈 각각에서 토치로 만들고 나중에 session 로더에서 인풋자체를 한번에 토치로 바꾸면될듯
    def _get_merged_map(self):
        if self.lifelong_map is None or self.new_session_map is None:
            raise ValueError("Both lifelong_map and new_session_map must be loaded before merging.")
        merged_map_o3d = o3d.geometry.PointCloud()
        merged_map_o3d.points = o3d.utility.Vector3dVector(
            np.vstack((self.lifelong_map.map, self.new_session_map.map)))
        merged_map_o3d = merged_map_o3d.voxel_down_sample(voxel_size=self.voxel_size)
        return np.asarray(merged_map_o3d.points)


    def classify_map_points(self):

        prev_dists, prev_idx = self.lifelong_map.kdtree.query(merged_map)
        new_dists,  new_idx  = self.new_session_map.kdtree.query(merged_map)



        # Classify points in the merged map into coexist, deleted, and emerged
        merged_map = self._get_merged_map()
        for point in tqdm(merged_map, desc="Point Classification (M_t)", ncols=100):
            prev_dist, _ = self.lifelong_map.kdtree.query(point)
            new_dist, _ = self.new_session_map.kdtree.query(point)
            if prev_dist < self.coexist_threshold and new_dist < self.coexist_threshold:
                pts_coexist.append(point)
            elif prev_dist < self.coexist_threshold:
                pts_deleted.append(point)
            elif new_dist < self.coexist_threshold:
                pts_emerged.append(point)
        pts_coexist = np.array(pts_coexist)
        kdtree_coexist = KDTree(pts_coexist)

        # Classify deleted and emerged points into overlap and non-overlap
        pts_deleted_overlap = []
        pts_deleted_nonoverlap = []
        for point in tqdm(pts_deleted, desc="Point Classification (D_t)", ncols=100):
            dist, _ = kdtree_coexist.query(point)
            if dist < self.overlap_threshold:
                pts_deleted_overlap.append(point)
            else:
                pts_deleted_nonoverlap.append(point)
        pts_deleted_overlap = np.array(pts_deleted_overlap)
        pts_deleted_nonoverlap = np.array(pts_deleted_nonoverlap)

        # Classify emerged points into overlap and non-overlap
        pts_emerged_overlap = []
        pts_emerged_nonoverlap = []
        for point in tqdm(pts_emerged, desc="Point Classification (E_t)", ncols=100):
            dist, _ = kdtree_coexist.query(point)
            if dist < self.overlap_threshold:
                pts_emerged_overlap.append(point)
            else:
                pts_emerged_nonoverlap.append(point)
        pts_emerged_overlap = np.array(pts_emerged_overlap)
        pts_emerged_nonoverlap = np.array(pts_emerged_nonoverlap)

        return {
            "pts_coexist": pts_coexist,
            "pts_deleted_overlap": pts_deleted_overlap,
            "pts_deleted_nonoverlap": pts_deleted_nonoverlap,
            "pts_emerged_overlap": pts_emerged_overlap,
            "pts_emerged_nonoverlap": pts_emerged_nonoverlap,
        }


    def update_global_ephemerality(self, classified_points: dict = None):
        if classified_points is not None:
            pts_coexist = classified_points["pts_coexist"]
            pts_deleted_overlap = classified_points["pts_deleted_overlap"]
            pts_deleted_nonoverlap = classified_points["pts_deleted_nonoverlap"]
            pts_emerged_overlap = classified_points["pts_emerged_overlap"]
            pts_emerged_nonoverlap = classified_points["pts_emerged_nonoverlap"]

        eph_g_coexist = self._update_global_eph_coexist(pts_coexist)
        eph_g_deleted_overlap = self._update_global_eph_deleted(pts_deleted_overlap, overlap=True)
        eph_g_deleted_nonoverlap = self._update_global_eph_deleted(pts_deleted_nonoverlap, overlap=False)
        eph_g_emerged_overlap = self._update_global_eph_emerged(pts_emerged_overlap, overlap=True)
        eph_g_emerged_nonoverlap = self._update_global_eph_emerged(pts_emerged_nonoverlap, overlap=False)

        return {
            "eph_g_coexist": eph_g_coexist,
            "eph_g_deleted_overlap": eph_g_deleted_overlap,
            "eph_g_deleted_nonoverlap": eph_g_deleted_nonoverlap,
            "eph_g_emerged_overlap": eph_g_emerged_overlap,
            "eph_g_emerged_nonoverlap": eph_g_emerged_nonoverlap,
        }

    def _update_global_eph_coexist(self, pts_coexist):
        # Coexist points
        eph_g_coexist = []
        for point in tqdm(pts_coexist, 
                          desc="Updating \u03B5_g (Coexist)", ncols=100):
            _, prev_idx = self.lifelong_map.kdtree.query(point)
            _, new_idx = self.new_session_map.kdtree.query(point)
            eph_g_prev = self.lifelong_map.eph[prev_idx]
            eph_l_new = self.new_session_map.eph[new_idx]
            eph_g_new = (
                (eph_g_prev * eph_l_new) /
                (eph_g_prev * eph_l_new + (1 - eph_g_prev) * (1 - eph_l_new))
            )# Eq. 7
            eph_g_coexist.append(eph_g_new)
        return np.asarray(eph_g_coexist)


    def _update_global_eph_deleted(self, pts_deleted, overlap=True):
        eph_g_deleted = []
        if overlap:
            gamma_deleted_overlap = self._calc_objectness_factor(pts_deleted)
            for idx, point in enumerate(tqdm(pts_deleted, 
                                             desc="Updating \u03B5_g (Deleted, Overlap)", ncols=100)):
                _, prev_idx = self.lifelong_map.kdtree.query(point)
                eph_g_prev = self.lifelong_map.eph[prev_idx]
                gamma = gamma_deleted_overlap[idx]
                eph_g_new = (
                    (eph_g_prev * gamma) /
                    (eph_g_prev * gamma + (1 - eph_g_prev) * (1 - gamma))
                )# Eq. 9
                eph_g_deleted.append(eph_g_new)
        else: # non-overlap
            for point in tqdm(pts_deleted,
                            desc="Updating \u03B5_g (Deleted, Non-Overlap)", ncols=100):
                _, prev_idx = self.lifelong_map.kdtree.query(point)
                eph_g_prev = self.lifelong_map.eph[prev_idx]
                eph_g_deleted.append(eph_g_prev) # unchanged
        return np.asarray(eph_g_deleted)


    def _update_global_eph_emerged(self, pts_emerged, overlap=True):
        eph_g_emerged = []
        if overlap:
            gamma_emerged_overlap = self._calc_objectness_factor(pts_emerged)
            for idx, point in enumerate(tqdm(pts_emerged, 
                                             desc="Updating \u03B5_g (Emerged, Overlap)", ncols=100)):
                _, new_idx = self.new_session_map.kdtree.query(point)
                eph_l_new = self.new_session_map.eph[new_idx]
                gamma = gamma_emerged_overlap[idx]
                eph_g_new = self.uncertainty_factor * (2 - gamma) * eph_l_new
                eph_g_emerged.append(eph_g_new)
        else: # non-overlap
            for point in tqdm(pts_emerged,
                            desc="Updating \u03B5_g (Emerged, Non-Overlap)", ncols=100):
                _, new_idx = self.new_session_map.kdtree.query(point)
                eph_l_new = self.new_session_map.eph[new_idx]
                eph_g_emerged.append(eph_l_new)
        return np.asarray(eph_g_emerged)


    def _remove_dynamic_points(self, points: np.ndarray, eph: np.ndarray):
        if len(points) == 0:
            return points, eph
        criteria = np.where(eph < self.global_eph_threshold, True, False)
        return points[criteria], eph[criteria]


    def _remove_outlier_points(self, points: np.ndarray, eph: np.ndarray, 
                               neighbors: int = 6, std_ratio: float = 1.0):
        if len(points) == 0:
            return points, eph
        points_o3d = o3d.geometry.PointCloud()
        points_o3d.points = o3d.utility.Vector3dVector(points)
        _, ind = points_o3d.remove_statistical_outlier(nb_neighbors=neighbors, std_ratio=std_ratio)
        return np.asarray(points_o3d.points)[ind], eph[ind]


    def _calc_objectness_factor(self, points: np.ndarray): # Eq. 8
        if len(points) == 0:
            return np.zeros(0)
        kdtree = KDTree(points)
        densities = [len(kdtree.query_ball_point(point, r=self.density_radius)) for point in points]
        densities = np.array(densities)
        densities = densities / np.max(densities)
        densities = np.power(densities, 1/self.rho_factor)
        return densities    


    def run(self):
        classified_points = self.classify_map_points()
        global_ephemerality = self.update_global_ephemerality(classified_points)
        updated_map = np.vstack((classified_points["pts_coexist"],
                                 classified_points["pts_deleted_overlap"],
                                 classified_points["pts_deleted_nonoverlap"],
                                 classified_points["pts_emerged_overlap"],
                                 classified_points["pts_emerged_nonoverlap"]))
        updated_eph = np.hstack((global_ephemerality["eph_g_coexist"],
                                 global_ephemerality["eph_g_deleted_overlap"],
                                 global_ephemerality["eph_g_deleted_nonoverlap"],
                                 global_ephemerality["eph_g_emerged_overlap"],
                                 global_ephemerality["eph_g_emerged_nonoverlap"]))
        if self.remove_dynamic_points:
            updated_map, updated_eph = self._remove_dynamic_points(updated_map, updated_eph)
        if self.remove_outlier_points:
            updated_map, updated_eph = self._remove_outlier_points(updated_map, updated_eph)
        updated_lifelong_map = SessionMap(updated_map, updated_eph)
        self.save(updated_lifelong_map)
        return updated_lifelong_map
    

    def save(self, lifelong_map: SessionMap):
        lifelong_map.save(self.params["settings"]["output_dir"], is_global=True)