import os
import yaml
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from tqdm import tqdm
import faiss

from utils.session_map import SessionMap

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


        # Global FAISS indices (lifelong vs new session)
        self.faiss_prev = None  # for lifelong_map
        self.faiss_new  = None  # for new_session_map

        #idx for Global eph update 
        self.coexist_prev_idx = None
        self.coexist_new_idx = None
        self.deleted_overlap_prev_idx = None
        self.deleted_overlap_new_idx = None
        self.emerged_overlap_new_idx = None
        self.emerged_nonoverlap_new_idx = None


    def load(self, lifelong_map: SessionMap, new_session_map: SessionMap):
        self.lifelong_map = lifelong_map
        self.new_session_map = new_session_map

    def _build_global_faiss(self):
        # Flat L2 for exact KNN on 3D points
        self.faiss_prev = faiss.IndexFlatL2(3)
        self.faiss_new = faiss.IndexFlatL2(3)
        self.faiss_prev.add(self.lifelong_map.map.astype('float32'))
        self.faiss_new.add(self.new_session_map.map.astype('float32'))

    @staticmethod
    def _build_local_faiss_index(points: np.ndarray, m: int = 32):
        # HNSW for faster local overlap checks on coexist set
        if points is None or points.size == 0:
            return None
        index = faiss.IndexHNSWFlat(3, m)
        index.add(points.astype('float32'))
        return index

    @staticmethod
    def _faiss_knn_with(index, queries: np.ndarray, k: int = 1):
        if queries is None or queries.size == 0:
            # shape-safe empty
            return np.empty((0, k), dtype=np.float32), np.empty((0, k), dtype=np.int64)
        if index is None:
            # coexist가 비었으면 분류 시 전부 non-overlap로 빠지게 됨
            n = len(queries)
            return np.full((n, k), np.inf, dtype=np.float32), np.full((n, k), -1, dtype=np.int64)
        d2, idx = index.search(queries.astype('float32'), k)
        d = np.sqrt(d2).astype(np.float32)
        return d, idx

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
        
        #1) load map, build fiass idx
        merged_map = self._get_merged_map().astype('float32')

        if self.faiss_prev is None or self.faiss_new is None:
            self._build_global_faiss()


        # 1-1) knn query
        prev_dists2, prev_idx = self.faiss_prev.search(merged_map, k=1)
        new_dists2,  new_idx  = self.faiss_new.search(merged_map, k=1)

        prev_idx = prev_idx.ravel()
        new_idx  = new_idx.ravel()

        prev_dists = np.sqrt(prev_dists2).ravel()
        new_dists = np.sqrt(new_dists2).ravel()

        #2) 
        prev_close = prev_dists<self.coexist_threshold
        new_close = new_dists<self.coexist_threshold

        mask_coexist =  prev_close &  new_close
        mask_deleted =  prev_close & ~new_close
        mask_emerged = ~prev_close &  new_close

        pts_coexist = merged_map[mask_coexist]
        pts_deleted = merged_map[mask_deleted]
        pts_emerged = merged_map[mask_emerged]

        pts_coexist = np.array(pts_coexist)


        #3) overlap vs non_overlap among coexist
        coexist_index = self._build_local_faiss_index(pts_coexist, m=32)

        dist_del, _ = self._faiss_knn_with(coexist_index, pts_deleted)
        dist_emg, _ = self._faiss_knn_with(coexist_index, pts_emerged)

        mask_del_overlap = dist_del.ravel()<self.overlap_threshold
        mask_emg_overlap = dist_emg.ravel()<self.overlap_threshold

        
        #cache ----------
        coexist_prev_idx = prev_idx[mask_coexist]
        coexist_new_idx  = new_idx[mask_coexist]
        deleted_prev_idx = prev_idx[mask_deleted]
        emerged_new_idx  = new_idx[mask_emerged]

        deleted_overlap_prev_idx    = deleted_prev_idx[mask_del_overlap]
        deleted_nonoverlap_prev_idx = deleted_prev_idx[~mask_del_overlap]
        emerged_overlap_new_idx     = emerged_new_idx[mask_emg_overlap]
        emerged_nonoverlap_new_idx  = emerged_new_idx[~mask_emg_overlap]

        self.coexist_prev_idx = coexist_prev_idx
        self.coexist_new_idx = coexist_new_idx
        self.deleted_overlap_prev_idx = deleted_overlap_prev_idx
        self.deleted_nonoverlap_prev_idx = deleted_nonoverlap_prev_idx
        self.emerged_overlap_new_idx = emerged_overlap_new_idx
        self.emerged_nonoverlap_new_idx = emerged_nonoverlap_new_idx
        
        #cache -----------------



        return {
            "pts_coexist": pts_coexist,
            "pts_deleted_overlap": pts_deleted[mask_del_overlap],
            "pts_deleted_nonoverlap": pts_deleted[~mask_del_overlap],
            "pts_emerged_overlap": pts_emerged[mask_emg_overlap],
            "pts_emerged_nonoverlap": pts_emerged[~mask_emg_overlap],
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
        eph_g_prev = self.lifelong_map.eph[self.coexist_prev_idx]
        eph_l_new  = self.new_session_map.eph[self.coexist_new_idx]

        denom = (eph_g_prev * eph_l_new + (1 - eph_g_prev) * (1 - eph_l_new))
        denom = np.clip(denom, 1e-12, None)
        eph_g_coexist = (eph_g_prev * eph_l_new) / denom
        return eph_g_coexist


    def _update_global_eph_deleted(self, pts_deleted, overlap=True):
        if overlap: 
            if len(pts_deleted) == 0:
                eph_g_deleted= np.zeros(0, dtype=np.float32)
            
            else:
                gamma_del = self._calc_objectness_factor(pts_deleted)
                eph_g_prev = self.lifelong_map.eph[self.deleted_overlap_prev_idx]

                denom = (eph_g_prev * gamma_del + (1 - eph_g_prev) * (1 - gamma_del))
                denom = np.clip(denom, 1e-12, None)
                eph_g_deleted = (eph_g_prev * gamma_del) / denom

        else: # non-overlap
            eph_g_deleted = self.lifelong_map.eph[self.deleted_nonoverlap_prev_idx] \
                               if len(self.deleted_nonoverlap_prev_idx) > 0 else np.zeros(0, dtype=np.float32)
        return np.asarray(eph_g_deleted)


    def _update_global_eph_emerged(self, pts_emerged, overlap=True):
        if overlap:
            if pts_emerged is None or len(pts_emerged) == 0:
                eph_g_emerged = np.zeros(0, dtype=np.float32)
            else: 
                gamma_emg = self._calc_objectness_factor(pts_emerged)
                eph_l_new = self.new_session_map.eph[self.emerged_overlap_new_idx]
                eph_g_merged = self.uncertainty_factor * (2.0 - gamma_emg) * eph_l_new

        else: # non-overlap
            eph_g_emerged = self.new_session_map.eph[self.emerged_nonoverlap_new_idx] \
                               if len(self.emerged_nonoverlap_new_idx) > 0 else np.zeros(0, dtype=np.float32)
        return eph_g_emerged 


    def _remove_dynamic_points(self, points: np.ndarray, eph: np.ndarray):
        if points is None or len(points) == 0:
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