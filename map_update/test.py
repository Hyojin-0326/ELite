import numpy as np
from scipy.spatial import KDTree
from utils.session_map import SessionMap
from map_updater import MapUpdater
import os


# ---- Mock SessionMap 생성 함수 ----
def make_dummy_sessionmap(num_points=100, eph_value=0.5, noise=0.01):
    # 임의 포인트 생성
    points = np.random.rand(num_points, 3) + np.random.normal(0, noise, (num_points, 3))
    eph = np.ones(num_points) * eph_value
    smap = SessionMap(points, eph)
    smap.kdtree = KDTree(points)  # MapUpdater가 사용하는 kdtree 속성 직접 추가
    return smap

# ---- 테스트용 config 파일 저장 ----
import yaml
config_data = {
    "map_update": {
        "voxel_size": 0.05,
        "coexist_threshold": 0.1,
        "overlap_threshold": 0.15,
        "density_radius": 0.2,
        "rho_factor": 2.0,
        "uncertainty_factor": 0.8,
        "global_eph_threshold": 0.5,
        "remove_dynamic_points": True,
        "remove_outlier_points": True
    },
    "settings": {
        "output_dir": "./test_output"
    }
}

with open("test_config.yaml", "w") as f:
    yaml.dump(config_data, f)

# ---- MapUpdater 테스트 ----
if __name__ == "__main__":
    # 1. 더미 맵 2개 생성
    lifelong_map = make_dummy_sessionmap(200, eph_value=0.6)
    new_session_map = make_dummy_sessionmap(150, eph_value=0.4)

    # 2. MapUpdater 초기화
    updater = MapUpdater("test_config.yaml")

    # 3. 맵 로드, 아웃풋경로 만들기
    updater.load(lifelong_map, new_session_map)
    output_dir = "./test_output"
    os.makedirs(output_dir, exist_ok=True)

    # 4. run 실행
    updated_map = updater.run()

    # 5. 결과 출력
    print("Updated map size:", updated_map.map.shape)
    print("Updated eph size:", updated_map.eph.shape)
    print("First 5 eph values:", updated_map.eph[:5])
