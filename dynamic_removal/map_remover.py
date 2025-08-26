import os
import yaml
import numpy as np
import open3d as o3d
from tqdm import trange
import torch
import torch.utils.dlpack
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
        t = torch.as_tensor(points)
        return_numpy = True
    elif isinstance(points, torch.Tensor):
        t = points
        return_numpy = False
    else:
        raise TypeError(f"Unsupported type: {type(points)}")

    if t.numel() == 0:
        return points

    device, dtype = t.device, t.dtype

    # 1) compute voxel integer coordinates
    v = torch.floor(t / voxel_size).to(torch.int64)

    # 2) hash voxel coords
    keys = v[:, 0] * 73856093 + v[:, 1] * 19349663 + v[:, 2] * 83492791

    # 3) group by key -> select first element in each voxel
    idx_sort = torch.argsort(keys)
    keys_sorted = keys[idx_sort]
    first_mask = torch.ones_like(keys_sorted, dtype=torch.bool)
    first_mask[1:] = keys_sorted[1:] != keys_sorted[:-1]
    first_idx = idx_sort[first_mask]

    out = t[first_idx].to(device=device, dtype=dtype)
    return out.cpu().numpy() if return_numpy else out


def downsample_points(points, voxel_size: float):
    """
    Voxel downsampling using Open3D (more accurate but slower).
    
    Args:
        points: torch.Tensor or numpy.ndarray (N,3)
        voxel_size: voxel size for downsampling
    Returns:
        torch.Tensor (same device & dtype as input)
    """
    if isinstance(points, torch.Tensor):
        device, dtype = points.device, points.dtype
        points = points.detach().cpu().numpy()
    else:
        device, dtype = torch.device("cpu"), torch.float32

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    downsampled = np.asarray(pcd.points)
    return torch.as_tensor(downsampled, device=device, dtype=dtype)


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

        # GPU objects
        self.gpu_scans = []  # torch.Tensor
        self.gpu_poses = []  # torch.Tensor
        self.faiss_index = None
            
    def load(self, new_session: Session = None):
        """Load scans and poses into GPU memory."""
        p_settings = self.params["settings"]

        self.session_loader = new_session or Session(
            p_settings["scans_dir"], 
            p_settings["poses_file"]
        )
        logger.info("Loaded new session, converting to tensors")

        self.num_scans = len(self.session_loader)

        for i in range(self.num_scans):
            logger.info(f"Processing scan {i+1}/{self.num_scans}")
            legacy_pcd = self.session_loader[i].get()
            points_np = np.asarray(legacy_pcd.points)

            if np.isnan(points_np).any() or np.isinf(points_np).any():
                logger.warning(f"Found NaN/Inf in scan #{i}")

            try:
                tpcd = o3d.t.geometry.PointCloud.from_legacy(legacy_pcd)
                positions_o3c = tpcd.point["positions"]
                positions_torch = torch.utils.dlpack.from_dlpack(
                    positions_o3c.to_dlpack()
                ).to(device="cuda", dtype=torch.float32)
                self.gpu_scans.append(positions_torch)
            except Exception as e:
                logger.error(f"Failed to convert scan #{i}: {e}")
                raise

            try:
                pose_np = self.session_loader.get_pose(i)[:3, 3].astype(np.float32)
                gpu_pose = torch.as_tensor(pose_np, dtype=torch.float32, device='cuda')
                self.gpu_poses.append(gpu_pose)
            except Exception as e:
                logger.error(f"Failed to load pose #{i}: {e}")
                raise

        logger.info(f"Loaded {len(self.gpu_scans)} scans and {len(self.gpu_poses)} poses")

    def build_faiss_index(self, anchor_points_tensor):
        """Build FAISS HNSW index on CPU for nearest neighbor search."""
        res = faiss.StandardGpuResources()
        dim, m = 3, 32
        self.faiss_index = faiss.IndexHNSWFlat(dim, m)

        anchor_np = anchor_points_tensor.detach().cpu().numpy().astype('float32')
        self.faiss_index.add(anchor_np)  
        logger.info(f"Built FAISS HNSW index with {anchor_np.shape[0]} points")

    def faiss_knn(self, queries: torch.Tensor, k: int):
        """Run k-NN search with FAISS."""
        queries_np = queries.detach().cpu().numpy().astype('float32')
        d2, idx = self.faiss_index.search(queries_np, k)
        d = np.sqrt(d2, dtype=np.float32)
        d = torch.as_tensor(d, device=queries.device)
        idx = torch.as_tensor(idx, device=queries.device, dtype=torch.int64)
        return d, idx

    def run(self):
        """Main pipeline: update ephemerality and remove dynamic objects."""
        p_settings = self.params["settings"]
        p_dor = self.params["dynamic_object_removal"]

        assert len(self.gpu_scans) > 0, "gpu_scans is empty!"
        session_map_tensor = torch.cat(self.gpu_scans, dim=0)
        eph_l = torch.zeros(session_map_tensor.shape[0], device=session_map_tensor.device)
        logger.info("Initialized session map")

        # Create anchor points
        anchor_points_tensor = downsample_points_torch(
            session_map_tensor, p_dor["anchor_voxel_size"]
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
        anchor_logits = torch.zeros(num_anchor_points, device=session_map_tensor.device)
        anchor_counts = torch.zeros(num_anchor_points, device=session_map_tensor.device)

        def logit(p): return torch.log(p / (1 - p + 1e-9))
        def inv_logit(l): return torch.sigmoid(l)

        # Update loop
        for i in trange(0, self.num_scans, p_dor["stride"], desc="Updating ε_l", ncols=100):
            logger.info(f"Processing scan {i+1}/{self.num_scans}")
            scan = self.gpu_scans[i]
            pose = self.gpu_poses[i]
            
            # Occupied update
            dists, inds = self.faiss_knn(scan, p_dor["num_k"])
            update_rate = torch.minimum(
                self.alpha * (1 - torch.exp(-1 * dists**2 / self.std_dev_o)) + self.beta,
                torch.tensor(self.alpha, device=dists.device)
            )
            logit_update = logit(update_rate)
            anchor_logits.scatter_add_(0, inds.flatten(), logit_update.flatten())  
            anchor_counts.scatter_add_(0, inds.flatten(), torch.ones_like(logit_update).flatten())

            # Free space update
            shifted_scan = scan - pose
            sample_ratios = torch.as_tensor(
                np.linspace(p_dor["min_ratio"], p_dor["max_ratio"], p_dor["num_samples"]),
                device=scan.device, dtype=scan.dtype
            )
            free_space_samples = pose + shifted_scan[:, None, :] * sample_ratios[None, :, None]
            free_space_samples = free_space_samples.reshape(-1, 3)
            free_space_samples = downsample_points_torch(free_space_samples, 0.1)

            dists, inds = self.faiss_knn(free_space_samples, p_dor["num_k"])
            update_rate = torch.clamp(
                self.alpha * (1 + torch.exp(-1 * dists.flatten()**2 / self.std_dev_f)) - self.beta,
                min=self.alpha
            )
            logit_update = logit(update_rate)
            anchor_logits.scatter_add_(0, inds.flatten(), logit_update.flatten()) 
            anchor_counts.scatter_add_(0, inds.flatten(), torch.ones_like(logit_update))

        anchor_eph_l = inv_logit(anchor_logits) 

        # Propagate anchor ephemerality to full session map
        distances, indices = self.faiss_knn(session_map_tensor, p_dor["num_k"])
        distances = torch.clamp(distances, min=1e-6)
        weights = (1 / (distances**2))
        weights = weights / weights.sum(dim=1, keepdim=True)
        
        eph_vals = anchor_eph_l[indices]
        eph_l = (weights * eph_vals).sum(dim=1)
        eph_l = torch.clamp(eph_l, 0.0, 1.0)

        # Split static/dynamic points
        static_mask = eph_l <= p_dor["dynamic_threshold"]
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

        if p_dor["save_static_dynamic_map"]:
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
