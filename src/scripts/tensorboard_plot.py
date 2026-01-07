import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def find_event_dirs(logdir: str) -> List[str]:
    dirs = []
    try:
        for name in os.listdir(logdir):
            if name.startswith("archive_"):
                continue
            path = os.path.join(logdir, name)
            if not os.path.isdir(path):
                continue
            for f in os.listdir(path):
                if f.startswith("events.out.tfevents"):
                    dirs.append(path)
                    break
    except FileNotFoundError:
        return []
    return sorted(dirs)


def load_scalars_from_dir(run_dir: str) -> Dict[str, List]:
    ea = event_accumulator.EventAccumulator(run_dir)
    ea.Reload()
    scalars = {}
    tags = ea.Tags().get("scalars", [])
    for tag in tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        scalars[tag] = (steps, values)
    return scalars


def collect_all_runs(logdir: str) -> Dict[str, Dict[str, tuple]]:
    runs = {}
    for d in find_event_dirs(logdir):
        run_name = os.path.relpath(d, logdir)
        runs[run_name] = load_scalars_from_dir(d)
    return runs


def merge_runs_sequentially(runs: Dict[str, Dict[str, tuple]]) -> Dict[str, tuple]:
    """Merge multiple runs into a single continuous series per tag by offsetting steps.

    Runs should be ordered by run key; caller may sort by name or timestamp.
    """
    merged = {}
    
    all_tags = set()
    for run in runs.values():
        all_tags.update(run.keys())

    
    cumulative_offset = 0
    for run_name, run_data in runs.items():
        for tag in all_tags:
            if tag not in run_data:
                continue
            steps, vals = run_data[tag]
            if tag not in merged:
                merged[tag] = ([], [])
            m_steps, m_vals = merged[tag]
            
            m_steps.extend([s + cumulative_offset for s in steps])
            m_vals.extend(vals)
        
        max_step = 0
        for rd in run_data.values():
            if rd[0]:
                max_step = max(max_step, max(rd[0]))
        cumulative_offset += max_step

    return merged


def plot_runs(runs: Dict[str, Dict[str, tuple]], outdir: str, tags: List[str] = None):
    os.makedirs(outdir, exist_ok=True)
    
    all_tags = set()
    for run in runs.values():
        all_tags.update(run.keys())
    if tags:
        plot_tags = [t for t in tags if t in all_tags]
    else:
        plot_tags = sorted(all_tags)

    for tag in plot_tags:
        plt.figure(figsize=(10, 5))
        for run_name, run_data in runs.items():
            if tag not in run_data:
                continue
            steps, vals = run_data[tag]
            plt.plot(steps, vals, label=run_name)
        plt.xlabel("Step")
        plt.ylabel(tag)
        plt.title(tag)
        plt.legend()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{tag.replace('/', '_')}_{ts}.png"
        path = os.path.join(outdir, fname)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()


def plot_merged(merged: Dict[str, tuple], outdir: str):
    os.makedirs(outdir, exist_ok=True)
    for tag, (steps, vals) in merged.items():
        plt.figure(figsize=(10, 5))
        plt.plot(steps, vals, color="tab:blue", linewidth=1.5)
        plt.xlabel("Step")
        plt.ylabel(tag)
        plt.title(f"Merged - {tag}")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"merged_{tag.replace('/', '_')}_{ts}.png"
        path = os.path.join(outdir, fname)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()


def make_publication_plot(run_dir: str, tag: str = "rollout/ep_rew_mean", outpath: str = None,
                          figsize=(6, 3), dpi: int = 300, ylabel: Optional[str] = None):
    """Create a publication-ready PDF (and PNG) for a single scalar tag from a run directory.

    - `run_dir` should be a folder containing TensorBoard event files.
    - `tag` is the scalar tag to plot (default `rollout/ep_rew_mean`).
    - If `outpath` is None, file will be saved into `run_dir` + '/plots_final/'.
    Returns the path to the saved PDF.
    """
    scalars = load_scalars_from_dir(run_dir)
    if tag not in scalars:
        raise KeyError(f"Tag '{tag}' not found in run {run_dir}")
    steps, vals = scalars[tag]
    if not steps:
        raise ValueError(f"No scalar points for tag '{tag}' in run {run_dir}")

    if outpath is None:
        outdir = os.path.join(run_dir, "plots_final")
        os.makedirs(outdir, exist_ok=True)
        base = tag.replace("/", "_")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outpath = os.path.join(outdir, f"{base}_{ts}.pdf")
    else:
        outdir = os.path.dirname(outpath) or os.getcwd()
        os.makedirs(outdir, exist_ok=True)

    plt.rcParams.update({"font.size": 10, "font.family": "serif"})
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(steps, vals, color="black", linewidth=1.5)
    ax.set_xlabel("Step")
    label = ylabel or ("Episode reward" if "ep_rew" in tag else tag)
    ax.set_ylabel(label)
    ax.grid(True, linewidth=0.3, alpha=0.8)
    fig.tight_layout()
    
    fig.savefig(outpath, dpi=dpi)
    png_path = outpath[:-4] + ".png"
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)
    return outpath


def watch_and_plot(logdir: str, outdir: str, tags: List[str] = None, poll_interval: float = 5.0):
    last_mod = 0
    while True:
        
        max_m = 0
        for d in find_event_dirs(logdir):
            for f in os.listdir(d):
                if f.startswith("events.out.tfevents"):
                    p = os.path.join(d, f)
                    try:
                        m = os.path.getmtime(p)
                        if m > max_m:
                            max_m = m
                    except OSError:
                        continue
        if max_m > last_mod:
            runs = collect_all_runs(logdir)
            plot_runs(runs, outdir, tags)
            last_mod = max_m
        time.sleep(poll_interval)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--logdir", default="./logs", help="TensorBoard log directory")
    p.add_argument("--outdir", default="./logs/plots", help="Output directory for PNGs")
    p.add_argument("--tags", default=None, help="Comma-separated scalar tags to plot")
    p.add_argument("--merge", action="store_true", help="Merge sequential runs into single continuous series")
    p.add_argument("--poll", type=float, default=5.0, help="Poll interval seconds (0 for single run)")
    args = p.parse_args()

    tags = args.tags.split(",") if args.tags else None
    if args.poll and args.poll > 0:
        watch_and_plot(args.logdir, args.outdir, tags, args.poll)
    else:
        runs = collect_all_runs(args.logdir)
        if args.merge:
            
            ordered = {k: runs[k] for k in sorted(runs.keys())}
            merged = merge_runs_sequentially(ordered)
            plot_merged(merged, args.outdir)
        else:
            plot_runs(runs, args.outdir, tags)


if __name__ == "__main__":
    main()
