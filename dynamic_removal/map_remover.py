import os
import yaml
import numpy as np
import open3d as o3d
from tqdm import trange
import torch
import open3d.core as o3c
import torch.utils.dlpack

from scipy.spatial import KDTree  # <<<< KDTree 사용

from utils.session import Session
from utils.session_map import SessionMap
from utils.logger import logger


# select the first point from the voxel (torch-only fast sampler)
def downsample_points_torch(points, voxel_size: float):
    """
    points: torch.Tensor (N,3) or numpy.ndarray (N,3)
    voxel_size: float
    returns: same type as input, downsampled
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

    device = t.device
    dtype = t.dtype

    v = torch.floor(t / voxel_size).to(torch.int64)
    # simple 64-bit-safe hash (collision probability is negligible for typical ranges)
    keys = v[:, 0] * 73856093 + v[:, 1] * 19349663 + v[:, 2] * 83492791

    idx_sort = torch.argsort(keys)
    keys_sorted = keys[idx_sort]
    first_mask = torch.ones_like(keys_sorted, dtype=torch.bool)
    first_mask[1:] = keys_sorted[1:] != keys_sorted[:-1]
    first_idx = idx_sort[first_mask]

    out = t[first_idx].to(device=device, dtype=dtype)
    if return_numpy:
        return out.cpu().numpy()
    return out


# legacy(Open3D CPU) voxel downsample wrapper
def downsample_points(points, voxel_size: float):
    """
    points: torch.Tensor (GPU/CPU) 또는 numpy.ndarray (N,3)
    voxel_size: 다운샘플링 voxel 크기
    return: torch.Tensor (same device & dtype as input) - 다운샘플링된 포인트
    """
    if isinstance(points, torch.Tensor):
        device = points.device
        dtype = points.dtype
        points = points.detach().cpu().numpy()
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    downsampled = np.asarray(pcd.points)
    return torch.as_tensor(downsampled, device=device, dtype=dtype)


class MapRemover:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.params = yaml.safe_load(f)

        p_settings = self.params["settings"]
        os.makedirs(p_settings["output_dir"], exist_ok=True)

        self.std_dev_o = 0.025
        self.std_dev_f = 0.025
        self.alpha = 0.5
        self.beta = 0.1

        self.session_loader: Session = None
        self.session_map: SessionMap = None

        # GPU 메모리 상의 객체들
        self.gpu_scans = []   # torch.cuda.FloatTensor [Ni,3]
        self.gpu_poses = []   # torch.cuda.FloatTensor [3]
        self.kdtree = None    # KDTree (CPU)
        self.anchor_points_np = None  # KDTree backing array (CPU)
        self.num_scans = 0

    def load(self, new_session: Session = None):
        p_settings = self.params["settings"]
        self.session_loader = new_session or Session(p_settings["scans_dir"], p_settings["poses_file"])
        logger.info(f"Loaded new session, start converting to tpcd")

        self.num_scans = len(self.session_loader)
        for i in range(self.num_scans):
            logger.info(f"Loading scan {i+1}/{self.num_scans}")
            legacy_pcd = self.session_loader[i].get()
            points_np = np.asarray(legacy_pcd.points)
            if np.isnan(points_np).any() or np.isinf(points_np).any():
                logger.warning(f"Found NaN or Inf in legacy_pcd #{i}")

            # Legacy → Open3D Tensor → Torch (CUDA)
            tpcd = o3d.t.geometry.PointCloud.from_legacy(legacy_pcd)
            positions_o3c = tpcd.point["positions"]
            positions_torch = torch.utils.dlpack.from_dlpack(
                positions_o3c.to_dlpack()
            ).to(device="cuda", dtype=torch.float32)
            self.gpu_scans.append(positions_torch)

            pose_np = self.session_loader.get_pose(i)[:3, 3].astype(np.float32)
            gpu_pose = torch.as_tensor(pose_np, dtype=torch.float32, device='cuda')
            self.gpu_poses.append(gpu_pose)

        logger.info(f"Converted all session to tensors: {len(self.gpu_scans)} scans, {len(self.gpu_poses)} poses")

    # ===== KDTree로 교체된 부분 =====
    def build_kdtree(self, anchor_points_tensor: torch.Tensor):
        """
        anchor_points_tensor: torch.Tensor (CUDA/CPU) [Na,3]
        KDTree는 CPU numpy를 필요로 하므로 옮겨서 빌드
        """
        anchor_np = anchor_points_tensor.detach().cpu().numpy().astype(np.float64)  # KDTree는 float64도 OK
        self.anchor_points_np = anchor_np
        self.kdtree = KDTree(anchor_np)
        logger.info(f"Built KDTree with {anchor_np.shape[0]} points")

    def kdtree_knn(self, queries: torch.Tensor, k: int):
        """
        queries: torch.Tensor (CUDA/CPU) [N,3]
        return:
          d: torch.Tensor (same device) [N,k]  (L2 distance)
          idx: torch.Tensor (same device,int64) [N,k]
        """
        device = queries.device
        q_np = queries.detach().cpu().numpy().astype(np.float64)
        dists, inds = self.kdtree.query(q_np, k=k)  # dists: [N,k], inds: [N,k]
        # 보장: shape 처리 (k==1이면 1D로 나오는 경우 방지)
        dists = np.atleast_2d(dists)
        inds = np.atleast_2d(inds)
        d = torch.as_tensor(dists, device=device, dtype=torch.float32)
        idx = torch.as_tensor(inds, device=device, dtype=torch.int64)
        return d, idx
    # =============================

    def run(self):
        p_settings = self.params["settings"]
        p_dor = self.params["dynamic_object_removal"]

        # 1) Aggregate scans to create session map
        assert len(self.gpu_scans) > 0, "gpu_scans is empty!"
        session_map_tensor = torch.cat(self.gpu_scans, dim=0)  # [M,3] cuda
        eph_l = torch.zeros(session_map_tensor.shape[0], device=session_map_tensor.device)
        logger.info(f"Initialized session map")
        # ----- 2) Select anchor points for local ephemerality update -----
        anchor_points_tensor = downsample_points(session_map_tensor, p_dor["anchor_voxel_size"])
        num_anchor_points = anchor_points_tensor.shape[0]
        if num_anchor_points == 0:
            raise RuntimeError("voxel_down_sample() returned empty point cloud! Check voxel size or input data.")

        self.build_kdtree(anchor_points_tensor)
        logger.info(f"KDTree ready with {num_anchor_points} points")

        # 확률 상태(베이지안 p)로 유지
        anchor_eph_l = torch.full((num_anchor_points,), 0.5, device=session_map_tensor.device, dtype=session_map_tensor.dtype)

        # 수치안정용 clamp
        def clamp01(x, eps=1e-6):
            return torch.clamp(x, min=eps, max=1.0-eps)

        # 한 번에 베이지안 업데이트 (벡터라이즈)
        def bayes_update_batch(anchor_p: torch.Tensor, inds: torch.Tensor, u: torch.Tensor):
            """
            anchor_p: [Na] 현재 앵커 확률
            inds:     [N*k] 앵커 인덱스(플랫)
            u:        [N*k] 해당 관측의 업데이트율(성공확률 역할)
            규칙: odds_new = odds_prev * Π (u/(1-u))  (같은 앵커끼리 곱)
            """
            # 그룹별 곱을 log-합으로 모음 (수치안정 + 완전 벡터화)
            u = clamp01(u)
            log_r = torch.log(u) - torch.log1p(-u)  # log(u/(1-u))
            # 같은 인덱스로 들어온 것들 합치기
            add_buf = torch.zeros_like(anchor_p)
            add_buf.index_add_(0, inds, log_r)

            # 기존 확률을 log-odds로 변환해 더하고 다시 확률로 복원
            p = clamp01(anchor_p)
            log_odds_prev = torch.log(p) - torch.log1p(-p)
            log_odds_new  = log_odds_prev + add_buf
            p_new = torch.sigmoid(log_odds_new)
            return p_new

        for i in trange(0, self.num_scans, p_dor["stride"], desc="Updating \u03B5_l", ncols=100):
            logger.debug(f"Processing scan {i + 1}/{self.num_scans}")
            scan = self.gpu_scans[i]   # [Ni,3] cuda
            pose = self.gpu_poses[i]   # [3]    cuda

            # ===== Occupied space update =====
            dists, inds = self.kdtree_knn(scan, p_dor["num_k"])  # dists, inds: [Ni,k]
            # 원래 코드의 u(d) 그대로 유지 (Eq.5 주석 그대로 따라감)
            u_occ = torch.minimum(
                self.alpha * (1 - torch.exp(-1 * (dists**2) / self.std_dev_o)) + self.beta,
                torch.tensor(self.alpha, device=dists.device, dtype=dists.dtype)
            )
            anchor_eph_l = bayes_update_batch(anchor_eph_l, inds.flatten(), u_occ.flatten())

            # ===== Free space update =====
            shifted_scan = scan - pose  # [Ni,3]
            ratios = torch.as_tensor(
                np.linspace(p_dor["min_ratio"], p_dor["max_ratio"], p_dor["num_samples"]),
                device=scan.device, dtype=scan.dtype
            )  # [S]
            free_samples = pose + shifted_scan[:, None, :] * ratios[None, :, None]  # (Ni,S,3)
            free_samples = free_samples.reshape(-1, 3)
            free_samples = downsample_points_torch(free_samples, 0.1)  # 과샘플 방지

            dists_f, inds_f = self.kdtree_knn(free_samples, p_dor["num_k"])
            # 원래 코드의 free-space u(d) 그대로
            u_free = torch.clamp(
                self.alpha * (1 + torch.exp(-1 * (dists_f**2) / self.std_dev_f)) - self.beta,
                min=self.alpha
            )
            anchor_eph_l = bayes_update_batch(anchor_eph_l, inds_f.flatten(), u_free.flatten())


        # 3) Propagate anchor local ephemerality to session map
        distances, indices = self.kdtree_knn(session_map_tensor, p_dor["num_k"])
        distances = torch.clamp(distances, min=1e-6)
        weights = 1.0 / (distances ** 2)
        weights = weights / weights.sum(dim=1, keepdim=True)

        eph_vals = anchor_eph_l[indices]  # (M,k)
        eph_l = (weights * eph_vals).sum(dim=1)  # (M,)
        eph_l = torch.clamp(eph_l, 0.0, 1.0)

        # 4) Remove dynamic objects to create cleaned session map
        static_mask = eph_l <= p_dor["dynamic_threshold"]
        dynamic_mask = ~static_mask

        static_points = session_map_tensor[static_mask]
        dynamic_points = session_map_tensor[dynamic_mask]
        static_eph_l = eph_l[static_mask]

        static_points_np = static_points.detach().cpu().numpy()
        dynamic_points_np = dynamic_points.detach().cpu().numpy()
        static_eph_l_np = static_eph_l.detach().cpu().numpy()

        static_pcd = o3d.geometry.PointCloud()
        dynamic_pcd = o3d.geometry.PointCloud()
        static_pcd.points = o3d.utility.Vector3dVector(static_points_np.astype(np.float64))
        dynamic_pcd.points = o3d.utility.Vector3dVector(dynamic_points_np.astype(np.float64))
        static_pcd.paint_uniform_color([0.5, 0.5, 0.5])
        dynamic_pcd.paint_uniform_color([1.0, 0.0, 0.0])

        if p_dor["save_static_dynamic_map"]:
            o3d.io.write_point_cloud(os.path.join(p_settings["output_dir"], "static_points.pcd"), static_pcd)
            o3d.io.write_point_cloud(os.path.join(p_settings["output_dir"], "dynamic_points.pcd"), dynamic_pcd)

        cleaned_session_map = SessionMap(static_points_np, static_eph_l_np)
        self.session_map = cleaned_session_map
        return self.session_map


# Example usage
if __name__ == "__main__":
    config = "../config/sample.yaml"
    remover = MapRemover(config)
    remover.load()
    remover.run()
    cleaned_session_map = remover.session_map
