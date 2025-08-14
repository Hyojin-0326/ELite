from utils.session_map import SessionMap
from map_updater import MapUpdater
import os
import yaml
import psutil

def log_mem(stage):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 ** 2)
    print(f"[MEM] {stage}: {mem_mb:.2f} MB")

# ---- 테스트용 config 파일 저장 ----
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

if __name__ == "__main__":
    outputs_dir = "/home/hjkwon/Desktop/Elite_fork/ELite/data/parkinglot/02/outputs"

    print("[1] Loading SessionMap (global)")
    log_mem("Before lifelong_map.load")
    lifelong_map = SessionMap()
    lifelong_map.load(outputs_dir, is_global=True)
    log_mem("After lifelong_map.load")

    print("[2] Loading SessionMap (local)")
    new_session_map = SessionMap()
    new_session_map.load(outputs_dir, is_global=False)
    log_mem("After new_session_map.load")

    print("[3] Initializing MapUpdater")
    updater = MapUpdater("parkinglot.yaml")
    updater.load(lifelong_map, new_session_map)
    log_mem("After updater.load")

    print("[4] Running MapUpdater.run()")
    updated_map = updater.run()
    log_mem("After updater.run")

    print("[5] Output check")
    print("Updated map size:", updated_map.map.shape)
    print("Updated eph size:", updated_map.eph.shape)
    print("First 5 eph values:", updated_map.eph[:5])
