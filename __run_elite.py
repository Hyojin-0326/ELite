# run_elite.py
import os, sys, subprocess

def run(cmd, env=None):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Need config: python run_elite.py <config_path>")
    cfg = sys.argv[1]

    # 메모리 억제만, 쓰레드 제한 해제
    env = os.environ.copy()
    env.setdefault("MALLOC_ARENA_MAX", "2")

    # 1) Zipper
    run([sys.executable, "-m", "alignment.run_zipper", cfg], env=env)
    # 2) Remover
    run([sys.executable, "-m", "dynamic_removal.run_remover", cfg], env=env)
    # 3) Updater
    run([sys.executable, "-m", "map_update.run_updater", cfg], env=env)

    print("ELite pipeline finished.")
