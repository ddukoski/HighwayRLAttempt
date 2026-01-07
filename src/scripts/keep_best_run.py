import os
import shutil
from datetime import datetime
from scripts.tensorboard_plot import find_event_dirs, load_scalars_from_dir


def select_best_run(logdir: str):
    runs = find_event_dirs(logdir)
    best = None
    best_max = -1
    for d in runs:
        scalars = load_scalars_from_dir(d)
        max_step = 0
        for tag, (steps, vals) in scalars.items():
            if steps:
                max_step = max(max_step, max(steps))
        if max_step > best_max:
            best_max = max_step
            best = d
    return best, best_max


def archive_other_runs(logdir: str, keep_run: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = os.path.join(logdir, f"archive_{timestamp}")
    os.makedirs(archive_dir, exist_ok=True)
    for root, dirs, files in os.walk(logdir):
        break
    for name in dirs:
        path = os.path.join(logdir, name)
        if os.path.abspath(path) == os.path.abspath(keep_run):
            continue
        if any(f.startswith("events.out.tfevents") for f in os.listdir(path)):
            shutil.move(path, os.path.join(archive_dir, name))
    return archive_dir


def rename_run_to(logdir: str, run_path: str, new_name: str):
    dst = os.path.join(logdir, new_name)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.move(run_path, dst)
    return dst


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--logdir", default="./logs")
    p.add_argument("--newname", default="DQN_best")
    args = p.parse_args()

    best, best_max = select_best_run(args.logdir)
    if not best:
        return
    archive = archive_other_runs(args.logdir, best)


if __name__ == "__main__":
    main()
