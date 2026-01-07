"""scripts package init -- exposes tensorboard plotting utilities."""
from .tensorboard_plot import (
    find_event_dirs,
    load_scalars_from_dir,
    collect_all_runs,
    plot_runs,
    watch_and_plot,
    make_publication_plot,
)

__all__ = [
    "find_event_dirs",
    "load_scalars_from_dir",
    "collect_all_runs",
    "plot_runs",
    "watch_and_plot",
]
