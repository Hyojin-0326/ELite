import os
import yaml
import numpy as np
import open3d as o3d
from tqdm import trange
from scipy.spatial import KDTree
import torch
import open3d.core as o3c
import torch.utils.dlpack
import pyflann
import faiss

import cupy as cp
from utils.session import Session
from utils.session_map import SessionMap
from utils.logger import logger


#select the first point from the voxel
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

# replaced downsample_points to downsample_points_torch. 
# (more faster but less accurate, It will be determined after evaluating the outputs.)
def downsample_points(points, voxel_size: float):
    """
    points: torch.Tensor (GPU/CPU) 또는 numpy.ndarray (N,3)
    voxel_size: 다운샘플링 voxel 크기
    return: torch.Tensor (same device & dtype as input) - 다운샘플링된 포인트
    """
    # 1. 입력이 torch면 디바이스/타입 저장
    if isinstance(points, torch.Tensor):
        device = points.device
        dtype = points.dtype
        points = points.detach().cpu().numpy()
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    # 2. Open3D Legacy PointCloud로 변환
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 3. voxel downsample (CPU)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # 4. 다시 torch로 (원래 디바이스 & dtype 유지)
    downsampled = np.asarray(pcd.points)
    return torch.as_tensor(downsampled, device=device, dtype=dtype)


class MapRemover:
    def __init__(
        self, 
        config_path: str
    ):
        # Load parameters
        with open(config_path, 'r') as f:
            self.params = yaml.safe_load(f)

        p_settings = self.params["settings"]        
        os.makedirs(p_settings["output_dir"], exist_ok=True)
        
        self.std_dev_o = 0.025
        self.std_dev_f = 0.025
        self.alpha = 0.5
        self.beta = 0.1

        self.session_loader : Session = None
        self.session_map : SessionMap = None

        #아래 객체들은 gpu 메모리의 객체임
        self.gpu_scans = [] #텐서
        self.gpu_poses = [] #텐서
        self.faiss_index = None
            
    def load(self, new_session: Session = None):
        p_settings = self.params["settings"]

        self.session_loader = new_session or Session(p_settings["scans_dir"], p_settings["poses_file"])
        logger.info(f"Loaded new session, start converting to tpcd")

        self.num_scans = len(self.session_loader)

        for i in range(self.num_scans):
            logger.info(f"Processing scan {i+1}/{self.num_scans}")
            legacy_pcd = self.session_loader[i].get()
            points_np = np.asarray(legacy_pcd.points)
            # logger.info(f"Legacy pcd #{i} has {points_np.shape[0]} points")

            # NaN/Inf 체크 (numpy → torch 안 거치고 바로)
            if np.isnan(points_np).any() or np.isinf(points_np).any():
                logger.warning(f"Found NaN or Inf in legacy_pcd #{i}")

            try:
                # Legacy → Open3D Tensor → Torch (GPU)
                tpcd = o3d.t.geometry.PointCloud.from_legacy(legacy_pcd)
                positions_o3c = tpcd.point["positions"]
                positions_torch = torch.utils.dlpack.from_dlpack(positions_o3c.to_dlpack()).to(device="cuda", dtype=torch.float32)
                # logger.info(f"Converted to torch tensor: shape={positions_torch.shape}, dtype={positions_torch.dtype}, device={positions_torch.device}")

                self.gpu_scans.append(positions_torch)
            except Exception as e:
                logger.error(f"Failed at scan #{i} during conversion: {e}")
                raise

            try:
                pose_np = self.session_loader.get_pose(i)[:3, 3].astype(np.float32)
                gpu_pose = torch.as_tensor(pose_np, dtype=torch.float32, device='cuda')
                # logger.info(f"Pose #{i}: {gpu_pose.cpu().numpy()}")
                self.gpu_poses.append(gpu_pose)
            except Exception as e:
                logger.error(f"Failed to load pose #{i}: {e}")
                raise

        logger.info(f"Converted all session to tensors: {len(self.gpu_scans)} scans, {len(self.gpu_poses)} poses")

    def build_faiss_index(self, anchor_points_tensor):
        # **HNSW 인덱스 생성**
        res = faiss.StandardGpuResources()
        dim = 3
        m = 32  # 그래프 연결 개수
        cpu_index = faiss.IndexHNSWFlat(dim, m)
        self.faiss_index = faiss.IndexHNSWFlat(dim, m)

        anchor_np = anchor_points_tensor.detach().cpu().numpy().astype('float32')
        self.faiss_index.add(anchor_np)  
        logger.info(f"Built FAISS HNSW index with {anchor_np.shape[0]} points")

    def faiss_knn(self, queries: torch.Tensor, k: int):
        # **쿼리 (GPU→numpy→FAISS→Torch)**
        queries_np = queries.detach().cpu().numpy().astype('float32')
        dists, idx = self.faiss_index.search(queries_np, k)
        dists = torch.as_tensor(dists, device=queries.device)
        idx = torch.as_tensor(idx, device=queries.device, dtype=torch.int64)
        return dists, idx


    def run(self):
        p_settings = self.params["settings"]
        p_dor = self.params["dynamic_object_removal"]

        # 1) Aggregate scans to create session map
        assert len(self.gpu_scans) > 0, "gpu_scans is empty!"

        session_map_tensor = torch.cat(self.gpu_scans, dim=0)
        eph_l = torch.zeros(session_map_tensor.shape[0], device=session_map_tensor.device)
        logger.info(f"Initialized session map")

        anchor_points_tensor = downsample_points_torch(session_map_tensor, p_dor["anchor_voxel_size"])
        num_anchor_points = anchor_points_tensor.shape[0]

        if num_anchor_points == 0:
            raise RuntimeError("voxel_down_sample() returned empty point cloud! Check voxel size or input data.")

    

        self.build_faiss_index(anchor_points_tensor)
        logger.info(f"Built HNSW with {num_anchor_points} points")


        #_________ 베이지안 업뎃을 logit으로 하기_________        
        anchor_logits = torch.zeros(num_anchor_points, device=session_map_tensor.device)
        anchor_counts = torch.zeros(num_anchor_points, device=session_map_tensor.device)

        def logit(p):
            return torch.log(p / (1 - p + 1e-9))

        def inv_logit(l):
            return torch.sigmoid(l)
        #________----__________________________________

        
        #To-do: 완전 텐서화(가능...?), gpu/cpu간 메모리 복사 최소화, 쿠다 stream 올리기, 짜잘한 연산 최적화 (eph 계속 딥카피나 재할당같은거)
        #occupied, free sp 업뎃에서 j 루프 돌 때 덮어쓰기 하던데... 난 걍 다 더해서 가중합 내려고 했음 

        for i in trange(0, self.num_scans, p_dor["stride"], desc="Updating \u03B5_l", ncols=100):

            logger.debug(f"Processing scan {i + 1}/{self.num_scans}")
            scan = self.gpu_scans[i]
            pose = self.gpu_poses[i]
            
            # occupied space update -------------------
            dists, inds = self.faiss_knn(scan, p_dor["num_k"])
            #update rate : (N, K)
            update_rate = torch.minimum(self.alpha * (1 - torch.exp(-1 * dists**2 / self.std_dev_o)) + self.beta,torch.tensor(self.alpha, device=dists.device))
                #로짓으로 업뎃
            logit_update = logit(update_rate) 
            anchor_logits.scatter_add_(0, inds.flatten(), logit_update.flatten())  
            anchor_counts.scatter_add_(0, inds.flatten(), torch.ones_like(logit_update).flatten())

            # torch.cuda.synchronize()
            # print("이거뜨면 occuppied까지는 잘된거임")


            # free space update --------------------
            shifted_scan = scan - pose # local coordinates
            sample_ratios = np.linspace(
                p_dor["min_ratio"],  
                p_dor["max_ratio"],  
                p_dor["num_samples"] 
            )
            sample_ratios = torch.as_tensor(sample_ratios, device=scan.device, dtype=scan.dtype)
            free_space_samples = pose + shifted_scan[:, None, :] * sample_ratios[None, :, None]  # (N, K, 3)
            free_space_samples = free_space_samples.reshape(-1, 3)  # (N*K, 3)


            ## free samples 너무 많을때 제한, gpu 상태 체크하고 동적으로 되게 바꿀수도 잇음 ----
            #기본비율 0.2, 600000개 넘을때 제한함
            # max_free_samples_ratio = p_dor.get("max_free_samples_ratio", 0.2)
            # max_free_samples_abs = p_dor.get("max_free_samples_abs", 600_000)

            # num_samples = free_space_samples.size(0)
            # if num_samples > max_free_samples_abs:
            #     target = int(num_samples * max_free_samples_ratio)
            #     sel = torch.randperm(num_samples, device=free_space_samples.device)[:target]
            #     free_space_samples = free_space_samples[sel]

            # #----

            free_space_samples = downsample_points_torch(free_space_samples, 0.1)
            dists, inds = self.faiss_knn(free_space_samples, p_dor["num_k"])

            # 마스크처럼 쓰려고 플래튼
            dists_flat = dists.flatten()

            update_rate = torch.clamp(
                self.alpha * (1 + torch.exp(-1 * dists_flat**2 / self.std_dev_f)) - self.beta,
                min=self.alpha
            )

            #___ 로짓 업뎃
            logit_update = logit(update_rate)
            anchor_logits.scatter_add_(0, inds.flatten(), logit_update.flatten()) 
            anchor_counts.scatter_add_(0, inds.flatten(), torch.ones_like(logit_update))

            # torch.cuda.synchronize()
            # print("이거뜨면 free space 업뎃까지는 잘된거임")

        anchor_eph_l = inv_logit(anchor_logits) 


        # 3) Propagate anchor local ephemerality to session map
        distances, indices = self.faiss_knn(session_map_tensor, p_dor["num_k"])
        distances = torch.clamp(distances, min=1e-6)
        weights = 1 / (distances**2)
        weights = weights / weights.sum(dim=1, keepdim=True)  # 정규화 (M, k)
        
        eph_vals = anchor_eph_l[indices]  # (M, k)
        eph_l = (weights * eph_vals).sum(dim=1)  # (M,)
        eph_l = torch.clamp(eph_l, 0.0, 1.0)

        # torch.cuda.synchronize()
        # print("이거뜨면 propagate까지는 잘된거임")


                
        # 4) Remove dynamic objects to create cleaned session map

        # 정적/동적 마스크 생성
        static_mask = eph_l <= p_dor["dynamic_threshold"]
        dynamic_mask = ~static_mask

        # 정적/동적 포인트 추출
        print(f"Static mask size: {static_mask.sum()}")  # 정적 포인트 개수 출력
        print(f"Dynamic mask size: {dynamic_mask.sum()}")  # 동적 포인트 개수 출력

        static_points = session_map_tensor[static_mask]     # (Ns, 3)
        dynamic_points = session_map_tensor[dynamic_mask]   # (Nd, 3)
        static_eph_l = eph_l[static_mask]

        # numpy로 변환
        static_points_np = static_points.detach().cpu().numpy()
        dynamic_points_np = dynamic_points.detach().cpu().numpy()
        static_eph_l_np = static_eph_l.detach().cpu().numpy()

        # 배열의 크기 확인
        print(f"Static points shape: {static_points_np.shape}")
        print(f"Dynamic points shape: {dynamic_points_np.shape}")
        print(f"Static eph_l shape: {static_eph_l_np.shape}")

        # SessionMap 생성
        cleaned_session_map = SessionMap(
            static_points_np,
            static_eph_l_np
        )

        # 빈 객체 만들기
        static_pcd  = o3d.geometry.PointCloud()
        dynamic_pcd = o3d.geometry.PointCloud()

        # 빈 객체에 넣고 색칠
        static_pcd.points = o3d.utility.Vector3dVector(static_points_np.astype(np.float64))
        dynamic_pcd.points = o3d.utility.Vector3dVector(dynamic_points_np.astype(np.float64))

        print("Assigned points to PointCloud objects.")

        # 색칠
        static_pcd.paint_uniform_color([0.5, 0.5, 0.5])
        dynamic_pcd.paint_uniform_color([1.0, 0.0, 0.0])

        # 저장 여부 및 경로 확인
        if p_dor["save_static_dynamic_map"]:
            print("Saving static and dynamic point clouds.")
            o3d.io.write_point_cloud(os.path.join(p_settings["output_dir"], "static_points.pcd"), static_pcd)  
            o3d.io.write_point_cloud(os.path.join(p_settings["output_dir"], "dynamic_points.pcd"), dynamic_pcd)

        if p_dor["viz_static_dynamic_map"]:
            print("Visualizing static and dynamic point clouds.")
            total_points = static_pcd + dynamic_pcd
            o3d.visualization.draw_geometries([total_points])

        # 세션 맵 업데이트
        self.session_map = cleaned_session_map

        # 세션 맵 저장 여부 확인
        if p_dor["save_cleaned_session_map"]:
            print("Saving cleaned session map.")
            cleaned_session_map.save(p_settings["output_dir"], is_global=False)

        if p_dor["viz_cleaned_session_map"]:
            print("Visualizing cleaned session map.")
            cleaned_session_map.visualize()

        return cleaned_session_map

    def get(self):
        return self.session_map
        

# Example usage
if __name__ == "__main__":
    config = "../config/sample.yaml"
    remover = MapRemover(config)
    # Load session using the config file or from an alingment module
    remover.load()
    # Run the dynamic object removal
    remover.run()
    # Get the cleaned session map
    cleaned_session_map = remover.get()