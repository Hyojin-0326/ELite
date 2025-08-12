import open3d as o3d
import open3d.core as o3c
import numpy as np
import os
import copy
from typing import List, Optional, Union
from dataclasses import dataclass, field

#TO-Do: minimize transform between legacy pcd and Tensor

# CUDA 커널 바인딩 모듈
from . import mycuda

@dataclass
class Scan:
    pose: np.ndarray
    path: str
    _pcd: Optional[o3d.geometry.PointCloud] = field(default=None, init=False)

    @property
    def pcd(self) -> o3d.geometry.PointCloud:
        if self._pcd is None:
            self._pcd = o3d.io.read_point_cloud(self.path)
            self._pcd.transform(self.pose)
        return self._pcd

class PointCloud:
    def __init__(self, cloud: o3d.geometry.PointCloud):
        self.cloud = cloud

    # Open3D legacy -> t.geometry.Tensor 변환
    def to_positions_tensor(self) -> o3c.Tensor:
        tpcd = o3d.t.geometry.PointCloud.from_legacy(self.cloud).cuda()
        return tpcd.point["positions"]

    # Tensor -> Open3D legacy 객체로 복원
    def from_positions_tensor(self, tensor: o3c.Tensor) -> 'PointCloud':
        tpcd = o3d.t.geometry.PointCloud()
        tpcd.point["positions"] = tensor
        legacy = tpcd.to_legacy()
        self.cloud = legacy
        return self

    # 지정된 CUDA 커널 호출 (DLpack 이용)
    def apply_cuda(self, kernel_name: str, *args, **kwargs) -> 'PointCloud':
        # positions Tensor 추출
        tensor = self.to_positions_tensor()
        # DLPack capsule 변환
        capsule = tensor.to_dlpack()
        # pybind11 바인딩 함수 호출
        out_capsule = getattr(mycuda, kernel_name)(capsule, *args, **kwargs)
        # 결과 Tensor 복원
        out_tensor = o3c.Tensor.from_dlpack(out_capsule)
        # Open3D 객체로 변환
        return self.from_positions_tensor(out_tensor)

    # 기존 CPU 메서드들 (필요 시 GPU 버전으로 대체 가능)
    def downsample(self, voxel_size: float = 0.2) -> 'PointCloud':
        self.cloud = self.cloud.voxel_down_sample(voxel_size)
        return self

    def colorize(self, color: List[float] = [1, 0, 0]) -> 'PointCloud':
        self.cloud.paint_uniform_color(color)
        return self

    # GPU: height_filter를 CUDA 커널로 대체
    def height_filter(self, height: float = 0.5) -> 'PointCloud':
        return self.apply_cuda("height_filter", height)

    def visualize(self) -> 'PointCloud':
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([self.cloud, mesh])
        return self
    
    def save(self, path: str) -> 'PointCloud':
        tpcd = o3d.t.geometry.PointCloud.from_legacy(self.cloud).cuda()
        z = tpcd.point["positions"][:, 2]  # GPU Tensor
        z_min = z.min()
        z_max = z.max()
        inten = (z - z_min) / (z_max - z_min + 1e-8)
        tpcd.point["colors"] = inten.unsqueeze(1).repeat(1, 3)
        legacy = tpcd.to_legacy()
        o3d.io.write_point_cloud(path, legacy)
        return self

    def get(self) -> o3d.geometry.PointCloud:
        return copy.deepcopy(self.cloud)

class Session:
    def __init__(self, scans_dir: str, pose_file: str):
        self.scans_dir = scans_dir
        self.pose_file = pose_file
        self.scans = self._load_scans(scans_dir, pose_file) #scan 객체들 리스트, map_zipper도 이거 안 쓰게 수정해야됨
        self.scans_np = np.stack([scan.pose for scan in self.scans]) #(N, 4, 4) 행렬

    @property
    def poses_np(self) -> np.ndarray:
        """빠른 접근용 pose ndarray (수정 시 sync 필요)"""
        return self._poses_np

    def get_pose(self, idx: int) -> np.ndarray:
        return self.scans[idx].pose

    def update_pose(self, idx: int, new_pose: np.ndarray):
        self.scans[idx].pose = new_pose
        self._poses_np[idx] = new_pose  # 캐시 동기화

    def save_pose(self, path: str):
        np.savetxt(path, self._poses_np.reshape(len(self._poses_np), -1))
    
    def _load_scans(self, scans_dir: str, pose_file: str) -> List[Scan]:
        poses = self._load_poses(pose_file) #poses: (N, 4, 4) 
        with os.scandir(scans_dir) as entries:
                scan_files = sorted(
                    (entry.name for entry in entries if entry.is_file() and entry.name.lower().endswith('.pcd'))
                )
        if poses.shape[0] != len(scan_files):
            raise ValueError(f"Mismatch: poses {poses.shape[0]} vs scan files {len(scan_files)}")
        scans = [Scan(pose, os.path.join(scans_dir, fname)) for pose, fname in zip(poses, scan_files)]
        return scans
    
    def _load_poses(self, pose_file: str) -> List[np.ndarray]:
        data = np.loadtxt(pose_file) #(N, 12) 어레이
        if data.shape[1] != 12:
            raise ValueError("Each pose must have 12 values")
        mats = data.reshape(-1, 3, 4)
        poses = []
        bottom = np.tile(np.array([[0,0,0,1]]), (mats.shape[0],1,1))
        poses = np.concatenate([mats, bottom], axis=1) # (N, 4, 4) 
        return poses

    def __len__(self):
        return len(self.scans)

    def __getitem__legacy(self, idx: Union[int, slice]) -> PointCloud:
        if isinstance(idx, int):
            cloud = self.scans[idx].pcd
            return PointCloud(cloud)
        elif isinstance(idx, slice):
            clouds = [self.scans[i].pcd for i in range(*idx.indices(len(self)))]#scan lists의 모든 pcd를 로드한다.
            combined = o3d.geometry.PointCloud() # 빈 포인트클라우드 생성 후 합침
            for pcd in clouds:
                combined += pcd # 딥카피
            return PointCloud(combined)
        else:
            raise TypeError("Index must be int or slice")
        
    def __getitem__(self, idx: Union[int, slice]) -> PointCloud: ## 포인트클라우드를 배치로 처리하게 하면 for문 없에기 가능< 일단 나중에 해야됨... 전체 파이프라인 수정 필요
        # tpcd = o3d.t.geometry.PointCloud.from_legacy(self.cloud).cuda()
        if isinstance(idx, int):
            cloud = self.scans[idx].pcd
            return PointCloud(cloud)
        elif isinstance(idx, slice):
            clouds = [self.scans[i].pcd for i in range(*idx.indices(len(self)))]#scan lists의 모든 scan(pcd)를 로드한다.
            combined = o3d.geometry.PointCloud() # 빈 포인트클라우드 생성 후 합침
            for pcd in clouds:
                combined += pcd # 딥카피
            return PointCloud(combined)
        else:
            raise TypeError("Index must be int or slice")

    def __repr__(self):
        return f"Session with {len(self.scans)} scans."

    def get_pose(self, idx: int) -> np.ndarray:
        return self.scans[idx].pose

    def update_pose(self, idx: int, pose: np.ndarray):
        self.scans[idx].pose = pose

    def save_pose(self, pose_file: str):
        with open(pose_file, 'w') as f:
            for scan in self.scans:
                line = ' '.join(map(str, scan.pose[:3].reshape(-1)))
                f.write(line + '\n')

    def save_pointcloud(self, idx: int, path: str, voxel_size: float = 0.1):
        pc = self[idx].downsample(voxel_size).get()
        z = np.asarray(pc.points)[:, 2]
        inten = (z - z.min()) / (z.max() - z.min() + 1e-8)
        pc.colors = o3d.utility.Vector3dVector(np.c_[inten, inten, inten])
        o3d.io.write_point_cloud(path, pc)

### legacy code
    def _load_scans_legacy(self, scans_dir: str, pose_file: str) -> List[Scan]:
        poses = self._load_poses(pose_file)
        scan_files = sorted([f for f in os.listdir(scans_dir) if f.endswith('.pcd')])
        if len(poses) != len(scan_files):
            raise ValueError("Mismatch between number of poses and scan files.")
        scans = []
        for i, pose in enumerate(poses):
            path = os.path.join(scans_dir, f'{i:06d}.pcd')
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing scan file: {path}")
            scans.append(Scan(pose, path))
        return scans
    def height_filter_legacy(self, height: float = 0.5) -> 'PointCloud':
        points = np.asarray(self.cloud.points)
        mask = points[:, 2] > height
        self.cloud = self.cloud.select_by_index(np.where(mask)[0])
        return self
    
    def _load_poses_legacy(self, pose_file: str) -> List[np.ndarray]:
        poses = []
        with open(pose_file, 'r') as f:
            for line in f:
                vals = list(map(float, line.strip().split()))
                if len(vals) != 12:
                    raise ValueError(f"Invalid pose line: {line.strip()}")
                mat = np.array(vals).reshape(3, 4)
                poses.append(np.vstack([mat, [0,0,0,1]]))
        return poses

    def save_legacy(self, path: str) -> 'PointCloud':
        z = np.asarray(self.cloud.points)[:, 2]
        inten = (z - z.min()) / (z.max() - z.min() + 1e-8)
        self.cloud.colors = o3d.utility.Vector3dVector(np.c_[inten, inten, inten])
        o3d.io.write_point_cloud(path, self.cloud)
        return self