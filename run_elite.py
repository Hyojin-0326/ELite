import os
import sys
import yaml
import gc
import ctypes

from alignment.map_zipper import MapZipper
from dynamic_removal.map_remover import MapRemover
from map_update.map_updater import MapUpdater

from utils.session import Session
from utils.session_map import SessionMap
from utils.logger import logger


def _malloc_trim():
    # glibc가 있으면 힙을 OS로 반납
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

def _clean(tag=""):
    # 강제 가비지 컬렉션 + 힙 트림
    gc.collect()
    _malloc_trim()
    if tag:
        logger.info(f"[CLEAN] Memory trimmed after: {tag}")


class ELite:
    def __init__(
        self,
        config_path: str
    ):
        # (선택) BLAS/GLIBC 메모리 붓기 억제 — 프로세스 시작 초반에 잡히면 효과 큼
        os.environ.setdefault("MALLOC_ARENA_MAX", "2")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        self.map_zipper = MapZipper(config_path)
        self.map_remover = MapRemover(config_path)
        self.map_updater = MapUpdater(config_path)

        with open(config_path, 'r') as f:
            self.params = yaml.safe_load(f)

    def run_elite(self):
        ###
        logger.info("Starting ELite...")

        # 1) Map Zipper
        logger.info("Running Map Zipper...")
        self.map_zipper.load_source_session()
        if "prev_output_dir" in self.params["settings"] and \
           os.path.exists(self.params["settings"]["prev_output_dir"]):
            self.map_zipper.load_target_session_map()

        aligned_session: Session = self.map_zipper.run()
        logger.info("Map Zipper finished.")

        # ---- CLEAN #1: Zipper 내부 큰 상태 끊기 ----
        # 더 이상 안 쓰는 객체 참조 제거
        self.map_zipper = None
        _clean("Map Zipper")

        # 2) Map Remover
        logger.info("Running Map Remover...")
        self.map_remover.load(aligned_session)
        cleaned_session_map: SessionMap = self.map_remover.run()
        logger.info("Map Remover finished.")

        # ---- CLEAN #2: Remover와 원본 세션 정리 (업데이터만 쓸 데이터만 남김) ----
        # remover 단계 산출물만 남기고 나머지 강제 해제
        del aligned_session
        self.map_remover = None
        _clean("Map Remover")

        # 3) Map Updater
        logger.info("Running Map Updater...")
        prev_lifelong_map = SessionMap()
        has_prev = "prev_output_dir" in self.params["settings"] and \
                   os.path.exists(self.params["settings"]["prev_output_dir"])
        if has_prev:
            prev_lifelong_map.load(self.params["settings"]["prev_output_dir"], is_global=True)

        # ---- CLEAN #3: updater 로딩 직전(특히 큰 인덱스/포인트클라우드가 있을 경우) ----
        _clean("Pre Map Updater load")

        if has_prev:
            self.map_updater.load(prev_lifelong_map, cleaned_session_map)

            # prev는 load 후 참조 끊어도 됨(업데이터 내부에 복사/참조를 가졌다고 가정)
            del prev_lifelong_map
            _clean("Map Updater.load")

            self.map_updater.run()
        else:
            # 최초 실행: 바로 저장
            self.map_updater.save(cleaned_session_map)

        logger.info("Map Updater finished.")

        # ---- CLEAN #4: 전체 파이프라인 종료 전 최종 정리 ----
        del cleaned_session_map
        self.map_updater = None
        _clean("Final")

        ###
        logger.info("ELite finished.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        sys.exit("Need config file path: python run_elite.py <config_path>")

    elite = ELite(config_path)
    elite.run_elite()
