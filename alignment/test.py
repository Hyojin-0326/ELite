import os
import numpy as np

base_dir = "./data/parkinglot/02"
scans_dir = os.path.join(base_dir, "Scans")
poses_path = os.path.join(base_dir, "poses.txt")

os.makedirs(scans_dir, exist_ok=True)

# 정상 PCD(ASCII) 한 개 내용
pcd_template = """# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH 1
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 1
DATA ascii
0.0 0.0 0.0
"""

N = 5  # 필요한 개수로 바꿔도 됨

# 1) 더미 스캔 파일들 만들기 (000000.pcd, 000001.pcd, ...)
for i in range(N):
    fname = os.path.join(scans_dir, f"{i:06d}.pcd")
    with open(fname, "w") as f:
        f.write(pcd_template)

# 2) poses.txt 만들기: 각 줄에 12개 숫자(3x4 변환행렬: R|t)
# 형식: r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
poses = []
for i in range(N):
    tx = float(i)  # 예: i번째 스캔은 x=i 지점에 있다고 가정
    T = np.array([
        [1.0, 0.0, 0.0, tx],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ], dtype=np.float32)  # 3x4
    poses.append(T.reshape(-1))  # 12원소

poses = np.vstack(poses)  # (N, 12)

# np.savetxt는 2D를 그대로 써주니 로더에서 shape[1]==12 충족
np.savetxt(poses_path, poses, fmt="%.6f")

print(f"✅ Created {N} dummy scans and poses at: {base_dir}")
print(f"   Scans dir: {scans_dir}")
print(f"   Poses file: {poses_path} (shape: {poses.shape})")
