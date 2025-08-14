import os, sys, yaml
from map_update.map_updater import MapUpdater
from utils.session_map import SessionMap
from utils.logger import logger
from utils.mem import drop_caches

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m map_update.run_updater <config.yaml>")
    cfg = sys.argv[1]
    with open(cfg) as f:
        params = yaml.safe_load(f)

    cache_dir = params["settings"]["cache_dir"]
    prev_dir  = params["settings"].get("prev_output_dir")
    out_dir   = params["settings"]["output_dir"]
    clean_dir = os.path.join(cache_dir, "cleaned_session_map")
    os.makedirs(out_dir, exist_ok=True)

    logger.info("[updater] start")
    # 입력 맵 로드
    new_map = SessionMap(); new_map.load(clean_dir, is_global=False)
    updater = MapUpdater(cfg)

    if prev_dir and os.path.exists(prev_dir):
        prev_map = SessionMap(); prev_map.load(prev_dir, is_global=True)
        updater.load(prev_map, new_map)
        updated = updater.run() 
    else:
        # prev 없으면 그대로 저장
        updater.save(new_map)

    # 메모리 정리
    new_map = None
    drop_caches()
    logger.info("[updater] done -> %s", out_dir)
