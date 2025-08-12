import open3d as o3d

# 데이터 경로 딕셔너리 (순서 1~4)
pcd_paths = {
    1: "/home/hjkwon/Desktop/Elite_fork/ELite/data/parkinglot/01/outputs/cleaned_session_map.pcd",
    2: "/home/hjkwon/Desktop/Elite_fork/ELite/data/parkinglot/01/outputs/lifelong_map.pcd",
    3: "/home/hjkwon/Desktop/Elite_fork/ELite/data/parkinglot/02/outputs/cleaned_session_map.pcd",
    4: "/home/hjkwon/Desktop/Elite_fork/ELite/data/parkinglot/02/outputs/lifelong_map.pcd",
}

def visualize_pcd(idx: int):
    if idx not in pcd_paths:
        raise ValueError(f"Invalid index {idx}. Valid indices: {list(pcd_paths.keys())}")
    
    pcd = o3d.io.read_point_cloud(pcd_paths[idx])
    o3d.visualization.webrtc_server.enable_webrtc()
    o3d.visualization.draw([pcd])

if __name__ == "__main__":
    # 인덱스 입력받아 실행
    idx = int(input(f"Select index {list(pcd_paths.keys())}: "))
    visualize_pcd(idx)
