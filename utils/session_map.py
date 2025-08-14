import os
import copy
import yaml
import numpy as np
import open3d as o3d
import faiss
from tqdm import tqdm
import torch

from utils.session import Session
from utils.session_map import SessionMap
from utils.logger import logger
from alignment.matcher.open3d_scan_matcher import Open3DScanMatcher
from alignment.matcher.pygicp_scan_matcher import PyGICPScanMatcher
from alignment.global_registration import register_with_fpfh_ransac

def downsample_points_torch(points, voxel_size: float):
    """
    points: torch.Tensor (N,3) or numpy.ndarray (N,3)
    voxel_size: float
    returns: same type as input, downsampled
    """
    # normalize input -> torch tensor
    if isinstance(points, np.ndarray):
        t = torch.as_tensor(points)            # CPU
        return_numpy = True
    elif isinstance(points, torch.Tensor):
        t = points
        return_numpy = False
    else:
        raise TypeError(f"Unsupported type: {type(points)}")

    if t.numel() == 0:
        return points

    device = t.device
    dtype  = t.dtype

    # 1) compute voxel integer coordinates
    v = torch.floor(t / voxel_size)            # float
    v = v.to(torch.int64)                      # int64 for stable hashing

    # 2) hash each voxel coord (64-bit safe)
    keys = v[:, 0] * 73856093 + v[:, 1] * 19349663 + v[:, 2] * 83492791

    # 3) group by key without return_index:
    #    sort by key -> take the first index of each group
    idx_sort = torch.argsort(keys)             # [N]
    keys_sorted = keys[idx_sort]               # [N]

    # mask marking first occurrence of each key in the sorted list
    first_mask = torch.ones_like(keys_sorted, dtype=torch.bool)
    first_mask[1:] = keys_sorted[1:] != keys_sorted[:-1]

    first_idx = idx_sort[first_mask]           # original indices of first reps
    out = t[first_idx].to(device=device, dtype=dtype)

    if return_numpy:
        return out.cpu().numpy()
    return out


class MapZipper:
    def __init__(
        self,
        config_path: str,
    ):
        # Load parameters and initialize data loaders
        with open(config_path, 'r') as f:
            self.params = yaml.safe_load(f)

        matcher_name = self.params["alignment"].get("matcher", "Open3DScanMatcher")
        self.matcher_cls = {"Open3DScanMatcher": Open3DScanMatcher,
                            "PyGICPScanMatcher": PyGICPScanMatcher,
                            }.get(matcher_name, None)
        
        self.source_session = None
        self.tgt_session_map = None
    

    def build_faiss_index_from_poses(self):
        poses = self.tgt_session_map.get_poses()
        if len(poses) == 0:
            raise ValueError("Target session map has no poses.")

        positions_np = np.vstack([pose[:3, 3] for pose in poses]).astype(np.float32)  # (N,3)

        dim, m = 3, 32
        self.faiss_index = faiss.IndexHNSWFlat(dim, m)  # CPU HNSW
        self.faiss_index.add(positions_np)
        logger.info(f"Built FAISS HNSW index (poses) with {positions_np.shape[0]} anchors")

    def faiss_knn(self, queries_np: np.ndarray, k: int):
        """
        queries_np: (Q,3) float32 numpy array
        return: (dists, idx) as numpy arrays
        """
        dists, idx = self.faiss_index.search(queries_np.astype('float32'), k)
        return dists, idx

    def load_target_session_map(self, tgt_session_map: SessionMap=None):
        p_settings = self.params["settings"]
        prev_output_dir = p_settings["prev_output_dir"]

        if tgt_session_map is None:
            logger.debug(f"Loading target session map from {prev_output_dir}")
            try:
                tgt_session_map = SessionMap()
                tgt_session_map.load(prev_output_dir, is_global=True)
            except FileNotFoundError:
                raise FileNotFoundError(f"Cannot find {prev_output_dir}")
        else:
            if not isinstance(tgt_session_map, SessionMap):
                raise TypeError("tgt_session_map must be an instance of SessionMap")
        self.tgt_session_map = tgt_session_map
    
    def load_source_session(self, src_session: Session=None):
        p_settings = self.params["settings"]
        if src_session is None:
            logger.debug(f"Loading source session from {p_settings['scans_dir']} and {p_settings['poses_file']}")
            try:
                src_session = Session(p_settings["scans_dir"], p_settings["poses_file"])
            except FileNotFoundError:
                raise FileNotFoundError(f"Cannot find {p_settings['scans_dir']} or {p_settings['poses_file']}")
        else:
            if not isinstance(src_session, Session):
                raise TypeError("src_session must be an instance of Session")
        self.src_session = src_session

    def _crop_legacy(self, pcd: o3d.geometry.PointCloud, center: np.ndarray, radius: float):
        min_b = center - radius
        max_b = center + radius
        aabb = o3d.geometry.AxisAlignedBoundingBox(min_b, max_b)
        return pcd.crop(aabb)
    
    def _crop(self, pcd, center, radius):
        mins = center - radius
        maxs = center + radius

        aabb_mask = ((pcd >= mins) & (pcd <= maxs)).all(dim=1)
        cand_idx = torch.nonzero(aabb_mask, as_tuple=False).squeeze(1)
        if cand_idx.numel() == 0:
            return o3d.geometry.PointCloud()
        
        cand = pcd.index_select(0, cand_idx)
        d2 = ((cand-center)**2).sum(dim=1)
        keep = torch.nonzero(d2<=(radius*radius), as_tuple=False).squeeze(1)
        if keep.numel() == 0:
            return o3d.geometry.PointCloud()
        
        final_pts = cand.index_select(0, keep)
        pcd_crop = o3d.geometry.PointCloud()
        pcd_crop.points = o3d.utility.Vector3dVector(final_pts.cpu().numpy())
        return pcd_crop


    def _init_transform(self) -> np.ndarray:
        p = self.params["alignment"]
        init_tf = np.eye(4)
        if "init_transform" in p:
            init_tf = np.array(p["init_transform"]).reshape(4, 4)
        return init_tf

    def _is_nonoverlapping_legacy(self, point: np.ndarray, threshold: float) -> bool:
        # Nearest-neighbor distance test
        tgt_session_map_poses = self.tgt_session_map.get_poses()
        if len(tgt_session_map_poses) == 0:
            raise ValueError("Target session map has no poses.")
        positions = np.vstack([pose[:3, 3] for pose in tgt_session_map_poses])
        kdtree = o3d.geometry.KDTreeFlann()
        kdtree.set_matrix_data(positions.T)
        _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
        return np.linalg.norm(positions[idx] - point) > threshold
    

    def _is_nonoverlapping(self, query, threshold: float) -> bool:
        """
        query: np.ndarray (3,) or torch.Tensor (3,)
        threshold: meters. FAISS L2는 '제곱거리' 반환하므로 threshold^2와 비교.
        """
        if not hasattr(self, 'faiss_index'):
            raise RuntimeError("FAISS index not built. Call build_faiss_index_from_poses() first.")

        if isinstance(query, torch.Tensor):
            q = query.detach().cpu().numpy().astype(np.float32).reshape(1, 3)
        else:
            q = np.asarray(query, dtype=np.float32).reshape(1, 3)

        dists, _ = self.faiss_index.search(q, 1)   # (1,1) 제곱 L2 거리
        return float(dists[0, 0]) > (threshold * threshold)
        


    def _update_poses_from(self, start: int, transform: np.ndarray, reverse: bool):
        rng = (range(start, -1, -1) if reverse 
               else range(start, len(self.src_session)))
        for j in rng:
            pose = self.src_session.get_pose(j)
            self.src_session.update_pose(j, transform @ pose)

    def _save_iteration(self, idx: int, cloud: o3d.geometry.PointCloud, reverse: bool):
        p_settings = self.params["settings"]        
        suffix = "rev_" if reverse else ""
        os.makedirs(p_settings["output_dir"], exist_ok=True)
        os.makedirs(os.path.join(p_settings["output_dir"], "aligned_poses"), exist_ok=True)
        os.makedirs(os.path.join(p_settings["output_dir"], "aligned_scans"), exist_ok=True)

        # Save updated poses
        pose_path = os.path.join(
            os.path.join(p_settings["output_dir"], "aligned_poses"), f"{suffix}aft_idx_{idx:06d}.txt"
        )
        self.src_session.save_pose(pose_path)
        
        # Color & write cloud
        cloud.paint_uniform_color([0, 1, 0])
        scan_path = os.path.join(
            os.path.join(p_settings["output_dir"], "aligned_scans"), f"{suffix}{idx:06d}.pcd"
        )
        o3d.io.write_point_cloud(scan_path, cloud)

    def _save_final_transform(self, reverse: bool):
        p_settings = self.params["settings"]        
        os.makedirs(p_settings["output_dir"], exist_ok=True)
    
        suffix = "rev_" if reverse else ""
        final_path = os.path.join(
            p_settings["output_dir"], f"{suffix}final_transform.txt"
        )
        self.src_session.save_pose(final_path)

    def _run_direction(self, reverse: bool):
        p_settings = self.params["settings"]
        p_alignment = self.params["alignment"]
        p = {**p_settings, **p_alignment}
        
        # forward: compute & apply init; reverse: reload forward-final
        if not reverse:
            init_tf = self._init_transform()
            logger.debug(f"Initial TF:\n{init_tf}")
            for i in range(len(self.src_session)):
                self.src_session.update_pose(
                    i,
                    init_tf @ self.src_session.get_pose(i)
                )
            init_path = os.path.join(p["output_dir"], "init_transform.txt")
            self.src_session.save_pose(init_path)
            logger.info(f"Saved init transform to {init_path}")
        else:
            final_path = os.path.join(p["output_dir"], "final_transform.txt")
            if not os.path.exists(final_path):
                raise RuntimeError(f"Cannot run reverse: missing {final_path}")
            self.src_session = Session(p["scans_dir"], final_path)
            logger.info(f"Loaded forward-final from {final_path}")

        # pre-merge target once
        merged_tgt = self.tgt_session_map.get()
        merged_tgt_pts = np.asarray(merged_tgt.points)
        self.build_faiss_index_from_poses() #building faiss idx for 'is_nonoverlapping'
        merged_tgt_pts_torch = torch.from_numpy(merged_tgt_pts).float().cuda(non_blocking=True)
        merged_tgt_pts_torch = downsample_points_torch(merged_tgt_pts_torch,p["tgt_voxel_size"])

        # scan loop
        desc = "Forward" if not reverse else "Reverse"
        logger.info(f"Running {'reverse' if reverse else 'forward'} pass")
        logger.debug(f"Using matcher: {self.matcher_cls.__name__}")
        idxs = range(len(self.src_session))
        idxs = [i for i in reversed(idxs)] if reverse else idxs
        for i in tqdm(idxs, desc=f"Alignment ({desc})", ncols=100):

            src_pc = self.src_session[i].get().points #.get은 o3dd 객체 반환하니까... 
            src_pc_pts = np.asarray(src_pc)
            src_pc_pts_torch = torch.from_numpy(src_pc_pts).float().cuda(non_blocking=True)
            src_pc_pts_torch = downsample_points_torch(src_pc_pts_torch, p["src_voxel_size"])
            src_pc_o3d = o3d.geometry.PointCloud()
            src_pc_o3d.points = o3d.utility.Vector3dVector(src_pc_pts_torch.detach().cpu().numpy())

            query = self.src_session.get_pose(i)[:3, 3]
            query_torch = torch.from_numpy(query).float().cuda(non_blocking=True)

            tgt_crop = self._crop(merged_tgt_pts_torch, query_torch, p.get("crop_radius", 100.0))

            if self._is_nonoverlapping(query, p.get("non_overlap_threshold", 10.0)):
                logger.debug(f"Scan {i} non-overlapping; skipping")
                # tf_pc = copy.deepcopy(src_pc)
            else:
                maxd = p["gicp_max_correspondence_distance"]
                if reverse:
                    maxd /= 2.0
                matcher = self.matcher_cls(max_correspondence_distance=maxd)
                matcher.set_input_src(src_pc_o3d)
                matcher.set_input_tgt(tgt_crop)
                matcher.align()
                tf = matcher.get_final_transformation()
                # tf_pc = copy.deepcopy(src_pc).transform(tf)
                self._update_poses_from(i, tf, reverse)
                fit, rmse = matcher.get_registration_result()
                logger.debug(f"Scan {i}: fit={fit:.2f}, rmse={rmse:.2f}")

            # self._save_iteration(i, tf_pc, reverse) # only for debug

        # write final
        self._save_final_transform(reverse)
        logger.info(f"{'Reverse' if reverse else 'Forward'} pass done.")

    def run(self):
        """Run both forward and then reverse registration."""
        p_settings = self.params["settings"]

        # if only one session is provided, return early
        if not self.tgt_session_map:
            self._save_final_transform(reverse=False)
            self._save_final_transform(reverse=True)
            return self.src_session
        
        # forward pass
        self._run_direction(reverse=False)
        # reverse pass
        self._run_direction(reverse=True)

        # return the updated session
        return Session(
            self.src_session.scans_dir,
            os.path.join(p_settings["output_dir"], "rev_final_transform.txt"),
        )


# Example usage:
if __name__ == "__main__":
    config = "./config/parkinglot.yaml"
    zipper = MapZipper(config)
    zipper.load_source_session()
    zipper.load_target_session_map()
    zipper.run()
    logger.info("Zipper finished.")  