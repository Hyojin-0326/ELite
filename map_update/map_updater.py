import os
import yaml
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from tqdm import tqdm
import faiss
import gc
from utils.logger import logger
import psutil
from utils.session_map import SessionMap

def log_mem(stage):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 ** 2)
    logger.info(f"[MEM] {stage}: {mem_mb:.2f} MB")

def save_chunks(self, point_chunks, eph_chunks):
    os.makedirs(self.params["settings"]["output_dir"], exist_ok=True)
    map_path = os.path.join(self.params["settings"]["output_dir"], "map.bin")
    eph_path = os.path.join(self.params["settings"]["output_dir"], "eph.bin")

    with open(map_path, "wb") as fm, open(eph_path, "wb") as fe:
        for pts, eph in zip(point_chunks, eph_chunks):
            np.asarray(pts, dtype=np.float32).tofile(fm)
            np.asarray(eph, dtype=np.float32).tofile(fe)

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

    def _build_faiss_index(self, points: np.ndarray, m: int = 32, ef_construction: int = 80, ef_search: int = 64):
        pts = np.ascontiguousarray(points, dtype=np.float32)
        index = faiss.IndexHNSWFlat(3, m)
        index.hnsw.efConstruction = ef_construction
        index.hnsw.efSearch = ef_search
        try:
            faiss.omp_set_num_threads(os.cpu_count())
        except Exception:
            pass
        index.add(pts)
        return index

    def load(self, lifelong_map, new_session_map):
        self.lifelong_map = lifelong_map
        self.new_session_map = new_session_map
        self.faiss_prev = self._build_faiss_index(self.lifelong_map.map, m=32, ef_construction=80, ef_search=64)
        self.faiss_new  = self._build_faiss_index(self.new_session_map.map, m=32, ef_construction=80, ef_search=64)

    def _build_global_faiss(self):
        # Flat L2 for exact KNN on 3D points
        self.faiss_prev = faiss.IndexFlatL2(3)
        self.faiss_new = faiss.IndexFlatL2(3)
        self.faiss_prev.add(self.lifelong_map.map.astype('float32'))
        self.faiss_new.add(self.new_session_map.map.astype('float32'))

    @staticmethod
    def _build_local_faiss_index(points: np.ndarray, m: int = 32, ef_search: int = 64):
        # CHANGED: KDTree → HNSW(FAISS)로 로컬 오버랩 체크도 근사화
        if points is None or points.size == 0:
            return None
        pts = np.ascontiguousarray(points, dtype=np.float32)  
        index = faiss.IndexHNSWFlat(3, m)
        index.hnsw.efConstruction = 80
        index.hnsw.efSearch = ef_search
        index.add(pts)
        return index

    @staticmethod
    def _faiss_knn_with(index, queries: np.ndarray, k: int = 1):
        # 동일: 제곱거리 반환 전제
        if queries is None or queries.size == 0:
            return (np.empty((0, k), dtype=np.float32),
                    np.empty((0, k), dtype=np.int64))
        if index is None:
            n = len(queries)
            return (np.full((n, k), np.inf, dtype=np.float32),
                    np.full((n, k), -1, dtype=np.int64))
        q = np.ascontiguousarray(queries, dtype=np.float32) 
        d2, idx = index.search(q, k)
        return d2, idx  
    

    #이거 토치로 바로 쌓을수도? 일단 모듈 각각에서 토치로 만들고 나중에 session 로더에서 인풋자체를 한번에 토치로 바꾸면될듯
    def _get_merged_map(self):
        if self.lifelong_map is None or self.new_session_map is None:
            raise ValueError("Both lifelong_map and new_session_map must be loaded before merging.")
        merged_map_o3d = o3d.geometry.PointCloud()
        merged_map_o3d.points = o3d.utility.Vector3dVector(
            np.vstack((self.lifelong_map.map, self.new_session_map.map)))
        merged_map_o3d = merged_map_o3d.voxel_down_sample(voxel_size=self.voxel_size)
        merged = np.asarray(merged_map_o3d.points)
        return merged


    def classify_map_points(self):
        logger.info("[classify_map_points] Start")

        # 1) load map, build faiss idx
        merged_map = self._get_merged_map()
        logger.info(f"[classify_map_points] merged_map shape: {merged_map.shape}")
        merged_f32 = np.ascontiguousarray(merged_map, dtype=np.float32)

        if self.faiss_prev is None or self.faiss_new is None:
            logger.info("[classify_map_points] Building global FAISS index")
            self._build_global_faiss()

        # 1-1) knn query
        logger.info("[classify_map_points] Running KNN queries")
        prev_d2, prev_idx = self.faiss_prev.search(merged_f32, 1)
        new_d2,  new_idx  = self.faiss_new.search(merged_f32, 1)
        prev_idx = prev_idx.ravel()
        new_idx  = new_idx.ravel()
        logger.info(f"[classify_map_points] KNN done. prev_idx size: {prev_idx.size}, new_idx size: {new_idx.size}")

        # 2) threshold masks
        th2 = float(self.coexist_threshold) * float(self.coexist_threshold)
        prev_close = (prev_d2.ravel() < th2)
        new_close  = (new_d2.ravel()  < th2)

        mask_coexist =  prev_close &  new_close
        mask_deleted =  prev_close & ~new_close
        mask_emerged = ~prev_close &  new_close

        logger.info(f"[classify_map_points] mask_coexist: {mask_coexist.sum()}, "
                    f"mask_deleted: {mask_deleted.sum()}, mask_emerged: {mask_emerged.sum()}")

        pts_coexist = merged_map[mask_coexist]
        pts_deleted = merged_map[mask_deleted]
        pts_emerged = merged_map[mask_emerged]

        logger.info(f"[classify_map_points] pts_coexist: {pts_coexist.shape}, "
                    f"pts_deleted: {pts_deleted.shape}, pts_emerged: {pts_emerged.shape}")

        # 3) overlap vs non_overlap among coexist
        coexist_index = self._build_local_faiss_index(pts_coexist, m=32, ef_search=64)
        logger.info("[classify_map_points] Built local FAISS index for coexist points")

        del_d2, _ = self._faiss_knn_with(coexist_index, pts_deleted)
        emg_d2, _ = self._faiss_knn_with(coexist_index, pts_emerged)

        ovl2 = float(self.overlap_threshold) * float(self.overlap_threshold)
        mask_del_overlap = del_d2.ravel() < ovl2
        mask_emg_overlap = emg_d2.ravel() < ovl2

        logger.info(f"[classify_map_points] mask_del_overlap: {mask_del_overlap.sum()}, "
                    f"mask_emg_overlap: {mask_emg_overlap.sum()}")

        # cache ----------
        coexist_prev_idx = prev_idx[mask_coexist]
        coexist_new_idx  = new_idx[mask_coexist]
        deleted_prev_idx = prev_idx[mask_deleted]
        emerged_new_idx  = new_idx[mask_emerged]

        self.coexist_prev_idx = coexist_prev_idx
        self.coexist_new_idx  = coexist_new_idx
        self.deleted_overlap_prev_idx    = deleted_prev_idx[mask_del_overlap]
        self.deleted_nonoverlap_prev_idx = deleted_prev_idx[~mask_del_overlap]
        self.emerged_overlap_new_idx     = emerged_new_idx[mask_emg_overlap]
        self.emerged_nonoverlap_new_idx  = emerged_new_idx[~mask_emg_overlap]
        logger.info("[classify_map_points] Cached index arrays")

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
                eph_g_emerged = self.uncertainty_factor * (2.0 - gamma_emg) * eph_l_new

        else:  # non-overlap
            if hasattr(self, "emerged_nonoverlap_new_idx") and len(self.emerged_nonoverlap_new_idx) > 0:
                eph_g_emerged = self.new_session_map.eph[self.emerged_nonoverlap_new_idx]
            else:
                eph_g_emerged = np.zeros(0, dtype=np.float32)

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
        logger.info("[RUN] classify_map_points() 시작")
        classified_points = self.classify_map_points()
        log_mem("After classify_map_points")

        logger.info("[RUN] update_global_ephemerality() 시작")
        global_ephemerality = self.update_global_ephemerality(classified_points)    
        log_mem("After update_global_ephemerality")

        logger.info("[RUN] updated_map / updated_eph 병합 시작")
        updated_map = np.vstack((
            classified_points["pts_coexist"],
            classified_points["pts_deleted_overlap"],
            classified_points["pts_deleted_nonoverlap"],
            classified_points["pts_emerged_overlap"],
            classified_points["pts_emerged_nonoverlap"]
        ))
        updated_eph = np.hstack((
            global_ephemerality["eph_g_coexist"],
            global_ephemerality["eph_g_deleted_overlap"],
            global_ephemerality["eph_g_deleted_nonoverlap"],
            global_ephemerality["eph_g_emerged_overlap"],
            global_ephemerality["eph_g_emerged_nonoverlap"]
        ))
        log_mem("After stack updated_map/updated_eph")

        if self.remove_dynamic_points:
            logger.info("[RUN] remove_dynamic_points 적용")
            updated_map, updated_eph = self._remove_dynamic_points(updated_map, updated_eph)
            log_mem("After remove_dynamic_points")

        if self.remove_outlier_points:
            logger.info("[RUN] remove_outlier_points 적용")
            updated_map, updated_eph = self._remove_outlier_points(updated_map, updated_eph)
            log_mem("After remove_outlier_points")

        logger.info("[RUN] npy 임시 저장 시작")
        map_path = os.path.join(self.params["settings"]["output_dir"], "map.npy")
        eph_path = os.path.join(self.params["settings"]["output_dir"], "eph.npy")

        os.makedirs(os.path.dirname(map_path), exist_ok=True)
        os.makedirs(os.path.dirname(eph_path), exist_ok=True)
        np.save(map_path, updated_map.astype(np.float32))
        np.save(eph_path, updated_eph.astype(np.float32))
        log_mem("After np.save()")

        logger.info("[RUN] updated_map / updated_eph 메모리 해제")
        del updated_map
        del updated_eph
        gc.collect()
        log_mem("After gc.collect()")

        logger.info("[RUN] mmap으로 다시 로드")
        loaded_map = np.load(map_path, mmap_mode='r')
        loaded_eph = np.load(eph_path, mmap_mode='r')
        updated_lifelong_map = SessionMap(loaded_map, loaded_eph)
        log_mem("After SessionMap 생성")

        logger.info("[RUN] 완료")
        return updated_lifelong_map
    
    def save(self, lifelong_map: SessionMap):
        lifelong_map.save(self.params["settings"]["output_dir"], is_global=True)