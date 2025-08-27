import os
import yaml
import numpy as np
import open3d as o3d
from tqdm import trange
import torch
import faiss
from utils.session import Session
from utils.session_map import SessionMap
from utils.logger import logger


def downsample_points_torch(points, voxel_size: float):
    """
    Voxel downsampling using PyTorch (selects the first point in each voxel).

    Args:
        points: torch.Tensor (N,3) or numpy.ndarray (N,3)
        voxel_size: voxel size for downsampling
    Returns:
        Downsampled points (same type as input)
    """
    if isinstance(points, np.ndarray):
        t = torch.as_tensor(points, dtype=torch.float32)
        return_numpy = True
    elif isinstance(points, torch.Tensor):
        t = points
        return_numpy = False
    else:
        raise TypeError(f"Unsupported type: {type(points)}")

    if t.numel() == 0:
        return points

    # CPU-only
    t = t.to(dtype=torch.float32, device='cpu')

    # 1) voxel integer coords
    v = torch.floor(t / voxel_size).to(torch.int64)

    # 2) hash voxel coords
    keys = v[:, 0] * 73856093 + v[:, 1] * 19349663 + v[:, 2] * 83492791

    # 3) group by key -> first in each voxel
    idx_sort = torch.argsort(keys)
    keys_sorted = keys[idx_sort]
    first_mask = torch.ones_like(keys_sorted, dtype=torch.bool)
    first_mask[1:] = keys_sorted[1:] != keys_sorted[:-1]
    first_idx = idx_sort[first_mask]

    out = t[first_idx]
    return out.numpy() if return_numpy else out


def downsample_points(points, voxel_size: float):
    """
    Voxel downsampling using Open3D (CPU).
    Returns torch.Tensor on CPU with original dtype if torch.Tensor in, else float32.
    """
    if isinstance(points, torch.Tensor):
        dtype = points.dtype
        points = points.detach().cpu().numpy()
    else:
        dtype = torch.float32

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    downsampled = np.asarray(pcd.points, dtype=np.float32)
    return torch.as_tensor(downsampled, device='cpu', dtype=dtype)


class MapRemover:
    def __init__(self, config_path: str):
        # Load parameters
        with open(config_path, 'r') as f:
            self.params = yaml.safe_load(f)

        p_settings = self.params["settings"]
        os.makedirs(p_settings["output_dir"], exist_ok=True)

        # Default parameters
        self.std_dev_o = 0.025
        self.std_dev_f = 0.025
        self.alpha = 0.5
        self.beta = 0.1

        self.session_loader: Session = None
        self.session_map: SessionMap = None

        # CPU objects
        self.cpu_scans = []  # torch.Tensor on CPU
        self.cpu_poses = []  # torch.Tensor on CPU
        self.faiss_index = None

    def load(self, new_session: Session = None):
        """Load scans and poses into CPU memory."""
        p_settings = self.params["settings"]

        self.session_loader = new_session or Session(
            p_settings["scans_dir"],
            p_settings["poses_file"]
        )
        logger.info("Loaded new session, converting to CPU tensors")

        self.num_scans = len(self.session_loader)

        for i in range(self.num_scans):
            logger.info(f"Processing scan {i+1}/{self.num_scans}")
            legacy_pcd = self.session_loader[i].get()
            points_np = np.asarray(legacy_pcd.points)

            if np.isnan(points_np).any() or np.isinf(points_np).any():
                logger.warning(f"Found NaN/Inf in scan #{i}")

            # simple: numpy -> torch (CPU)
            positions_torch = torch.as_tensor(points_np, dtype=torch.float32, device='cpu')
            self.cpu_scans.append(positions_torch)

            # poses: take translation only (3,) as before
            pose_np = self.session_loader.get_pose(i)[:3, 3].astype(np.float32)
            cpu_pose = torch.as_tensor(pose_np, dtype=torch.float32, device='cpu')
            self.cpu_poses.append(cpu_pose)

        logger.info(f"Loaded {len(self.cpu_scans)} scans and {len(self.cpu_poses)} poses (CPU)")

    def build_faiss_index(self, anchor_points_tensor: torch.Tensor):
        """Build FAISS HNSW index on CPU for nearest neighbor search."""
        dim, m = 3, 32
        self.faiss_index = faiss.IndexHNSWFlat(dim, m)

        anchor_np = anchor_points_tensor.detach().cpu().numpy().astype('float32')
        self.faiss_index.add(anchor_np)
        logger.info(f"Built FAISS HNSW index with {anchor_np.shape[0]} points (CPU)")

    def faiss_knn(self, queries: torch.Tensor, k: int):
        """Run k-NN search with FAISS (CPU)."""
        queries_np = queries.detach().cpu().numpy().astype('float32')
        d2, idx = self.faiss_index.search(queries_np, k)  # d2: squared L2
        d = np.sqrt(d2, dtype=np.float32)
        d = torch.from_numpy(d)           # CPU tensor
        idx = torch.from_numpy(idx).to(torch.int64)
        return d, idx

    def run(self):
        """Main pipeline: update ephemerality and remove dynamic objects (CPU)."""
        p_settings = self.params["settings"]
        p_dor = self.params["dynamic_object_removal"]

        assert len(self.cpu_scans) > 0, "cpu_scans is empty!"
        session_map_tensor = torch.cat(self.cpu_scans, dim=0).to('cpu', dtype=torch.float32)
        eph_l = torch.zeros(session_map_tensor.shape[0], device='cpu', dtype=torch.float32)
        logger.info("Initialized session map (CPU)")

        # Create anchor points
        anchor_points_tensor = downsample_points_torch(
            session_map_tensor, float(p_dor["anchor_voxel_size"])
        )
        num_anchor_points = anchor_points_tensor.shape[0]
        if num_anchor_points == 0:
            raise RuntimeError("Empty anchor points, check voxel size or input data.")

        k_req = int(p_dor["num_k"])
        k = min(k_req, num_anchor_points)
        if k < k_req:
            logger.warning(f"num_k({k_req}) > #anchors({num_anchor_points}) → using k={k}")

        self.build_faiss_index(anchor_points_tensor)

        # Bayesian update (logit domain)
        anchor_logits = torch.zeros(num_anchor_points, device='cpu', dtype=torch.float32)
        anchor_counts = torch.zeros(num_anchor_points, device='cpu', dtype=torch.float32)

        def logit(p): return torch.log(p / (1 - p + 1e-9))
        def inv_logit(l): return torch.sigmoid(l)

        # Update loop
        for i in trange(0, self.num_scans, int(p_dor["stride"]), desc="Updating ε_l", ncols=100):
            logger.info(f"Processing scan {i+1}/{self.num_scans}")
            scan = self.cpu_scans[i]          # (N,3) CPU
            pose = self.cpu_poses[i]          # (3,)  CPU

            # Occupied update
            dists, inds = self.faiss_knn(scan, k)
            # shape alignment
            # update_rate in (0, alpha] roughly
            update_rate = torch.minimum(
                self.alpha * (1 - torch.exp(- (dists ** 2) / self.std_dev_o)) + self.beta,
                torch.tensor(self.alpha, dtype=torch.float32)
            )
            logit_update = logit(update_rate)              # (N, k)
            anchor_logits.scatter_add_(0, inds.flatten(), logit_update.flatten())
            anchor_counts.scatter_add_(0, inds.flatten(), torch.ones_like(logit_update, dtype=torch.float32).flatten())

            # Free space update
            shifted_scan = scan - pose                      # (N,3)
            sample_ratios = torch.linspace(
                float(p_dor["min_ratio"]),
                float(p_dor["max_ratio"]),
                int(p_dor["num_samples"]),
                dtype=torch.float32,
                device='cpu'
            )                                               # (S,)
            # (N,3) -> (N,1,3) * (1,S,1) -> (N,S,3)
            free_space_samples = pose + shifted_scan[:, None, :] * sample_ratios[None, :, None]
            free_space_samples = free_space_samples.reshape(-1, 3)
            free_space_samples = downsample_points_torch(free_space_samples, 0.1)

            dists_fs, inds_fs = self.faiss_knn(free_space_samples, k)
            # larger update for free space evidence, clamp min at alpha
            update_rate_fs = torch.clamp(
                self.alpha * (1 + torch.exp(- (dists_fs.flatten() ** 2) / self.std_dev_f)) - self.beta,
                min=self.alpha
            )
            logit_update_fs = logit(update_rate_fs)
            anchor_logits.scatter_add_(0, inds_fs.flatten(), logit_update_fs.flatten())
            anchor_counts.scatter_add_(0, inds_fs.flatten(), torch.ones_like(logit_update_fs, dtype=torch.float32))

        # convert logits -> probabilities per anchor
        anchor_eph_l = inv_logit(anchor_logits)

        # Propagate anchor ephemerality to full session map
        distances, indices = self.faiss_knn(session_map_tensor, k)
        distances = torch.clamp(distances, min=1e-6)
        weights = 1.0 / (distances ** 2)                   # (M, k)
        weights = weights / weights.sum(dim=1, keepdim=True)

        eph_vals = anchor_eph_l[indices]                   # (M, k)
        eph_l = (weights * eph_vals).sum(dim=1)            # (M,)
        eph_l = torch.clamp(eph_l, 0.0, 1.0)

        # Split static/dynamic points
        static_mask = eph_l <= float(p_dor["dynamic_threshold"])
        dynamic_mask = ~static_mask

        static_points = session_map_tensor[static_mask]
        dynamic_points = session_map_tensor[dynamic_mask]
        static_eph_l = eph_l[static_mask]

        static_points_np = static_points.detach().cpu().numpy()
        dynamic_points_np = dynamic_points.detach().cpu().numpy()
        static_eph_l_np = static_eph_l.detach().cpu().numpy()

        logger.info(f"Static points: {static_points_np.shape}, Dynamic points: {dynamic_points_np.shape}")

        # Build cleaned session map
        cleaned_session_map = SessionMap(static_points_np, static_eph_l_np)
        self.session_map = cleaned_session_map

        # Save results
        static_pcd, dynamic_pcd = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
        static_pcd.points = o3d.utility.Vector3dVector(static_points_np.astype(np.float64))
        dynamic_pcd.points = o3d.utility.Vector3dVector(dynamic_points_np.astype(np.float64))
        static_pcd.paint_uniform_color([0.5, 0.5, 0.5])
        dynamic_pcd.paint_uniform_color([1.0, 0.0, 0.0])

        if p_dor.get("save_static_dynamic_map", True):
            o3d.io.write_point_cloud(os.path.join(p_settings["output_dir"], "static_points.pcd"), static_pcd)
            o3d.io.write_point_cloud(os.path.join(p_settings["output_dir"], "dynamic_points.pcd"), dynamic_pcd)

        return self.session_map

    def get(self):
        """Return cleaned session map."""
        return self.session_map


if __name__ == "__main__":
    config = "../config/sample.yaml"
    remover = MapRemover(config)
    remover.load()
    remover.run()
    cleaned_session_map = remover.get()
