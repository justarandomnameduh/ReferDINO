from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from visualize.common import (  # noqa: E402
    apply_plot_style,
    build_context,
    combine_video_results,
    load_davis_results,
    load_mevis_metadata,
    load_mevis_results,
    plot_metric_distributions,
    print_context_summary,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot J/F/J&F distributions for DAVIS annotators and MeViS.")
    parser.add_argument("--version", default="swinb_repro")
    parser.add_argument("--mevis-split", default="valid_u")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--save-path", default=None, help="PNG path. Default: output/visualize/<version>/metric_distributions.png")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_plot_style()
    ctx = build_context(version=args.version, mevis_split=args.mevis_split, output_root=args.output_root)
    print_context_summary(ctx)

    _, davis_video, _ = load_davis_results(ctx)
    mevis_meta = load_mevis_metadata(ctx)
    _, mevis_video = load_mevis_results(ctx, meta=mevis_meta)
    combined = combine_video_results(davis_video, mevis_video)

    fig, _ = plot_metric_distributions(combined)
    save_path = Path(args.save_path).resolve() if args.save_path else ctx.save_root / "metric_distributions.png"
    save_figure(fig, save_path, close=not args.show)
    print(f"Saved metric distributions to {save_path}")

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
