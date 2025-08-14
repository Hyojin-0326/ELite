import os, sys, yaml
from dynamic_removal.map_remover import MapRemover
from utils.session import Session
from utils.session_map import SessionMap
from utils.logger import logger
from utils.mem import drop_caches

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m dynamic_removal.run_remover <config.yaml>")
    cfg = sys.argv[1]
    with open(cfg) as f:
        params = yaml.safe_load(f)

    cache_dir = params["settings"]["cache_dir"]
    aligned_path = os.path.join(cache_dir, "aligned_session")
    out_clean = os.path.join(cache_dir, "cleaned_session_map")
    os.makedirs(out_clean, exist_ok=True)

    logger.info("[remover] start")
    # 캐시에서 aligned_session 로드
    aligned = Session()
    aligned.load(aligned_path)

    remover = MapRemover(cfg)
    remover.load(aligned)
    cleaned: SessionMap = remover.run()

    # 저장
    cleaned.save(out_clean)
    logger.info("[remover] saved -> %s", out_clean)

    # 메모리 정리
    aligned = None
    cleaned = None
    drop_caches()
    logger.info("[remover] done")
