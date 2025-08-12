import open3d as o3d
import open3d.core as o3c
import numpy as np
import mycuda

# 1. PointCloud 로드
pcd = o3d.io.read_point_cloud("/home/hjkwon/Desktop/ELite/utils/utils_cuda/test/000000.pcd")
tpcd = o3d.t.geometry.PointCloud.from_legacy(pcd).cuda()

# 2. Tensor 추출 → DLPack 변환
positions = tpcd.point["positions"]
capsule = positions.to_dlpack()

# CUDA 호출 전 최대 z값
z_before = tpcd.point["positions"][:, 2].max().item()
print(f"Before CUDA, max z: {z_before}")

# 3. CUDA 커널 호출
result_capsule = mycuda.height_filter(capsule, 0.5)

# 4. Tensor 복구 → PointCloud 복원
out_tensor = o3c.Tensor.from_dlpack(result_capsule)

print(f"After CUDA, max z: {out_tensor[:, 2].max().item()}")

tpcd.point["positions"] = out_tensor
filtered = tpcd.to_legacy()

# 5. 시각화
o3d.visualization.draw_geometries([filtered])
