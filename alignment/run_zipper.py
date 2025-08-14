import os, sys, yaml
from alignment.map_zipper import MapZipper
from utils.session import Session
from utils.logger import logger
from utils.mem import drop_caches

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m alignment.run_zipper <config.yaml>")
    cfg = sys.argv[1]
    with open(cfg) as f:
        params = yaml.safe_load(f)

    cache_dir = params["settings"]["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)

    logger.info("[zipper] start")
    zipper = MapZipper(cfg)
    zipper.load_source_session()
    if "prev_output_dir" in params["settings"] and os.path.exists(params["settings"]["prev_output_dir"]):
        zipper.load_target_session_map()

    aligned_session: Session = zipper.run()

    # 캐시에 저장
    aligned_path = os.path.join(cache_dir, "aligned_session")
    os.makedirs(aligned_path, exist_ok=True)
    aligned_session.save(aligned_path)

    # 메모리 정리
    aligned_session = None
    drop_caches()
    logger.info("[zipper] done -> %s", aligned_path)
