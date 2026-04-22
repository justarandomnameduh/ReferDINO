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
    load_davis_results,
    load_mevis_metadata,
    load_mevis_results,
    plot_main_video_scatter,
    print_context_summary,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the main video-level J-vs-F scatter for DAVIS and MeViS.")
    parser.add_argument("--version", default="swinb_repro")
    parser.add_argument("--mevis-split", default="valid_u")
    parser.add_argument("--output-root", default=None, help="Override ReferDINO/output root.")
    parser.add_argument("--save-path", default=None, help="PNG path. Default: output/visualize/<version>/video_scatter.png")
    parser.add_argument("--label-top-n", type=int, default=3, help="Number of worst points to label per group.")
    parser.add_argument("--show", action="store_true", help="Also display the plot interactively.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_plot_style()
    ctx = build_context(version=args.version, mevis_split=args.mevis_split, output_root=args.output_root)
    print_context_summary(ctx)

    davis_raw, davis_video, _ = load_davis_results(ctx)
    mevis_meta = load_mevis_metadata(ctx)
    mevis_expr, mevis_video = load_mevis_results(ctx, meta=mevis_meta)

    fig, _, combined = plot_main_video_scatter(davis_video, mevis_video, label_top_n=args.label_top_n)
    save_path = Path(args.save_path).resolve() if args.save_path else ctx.save_root / "video_scatter.png"
    save_figure(fig, save_path, close=not args.show)
    print(f"Saved video scatter to {save_path}")
    print(f"DAVIS video points: {len(davis_video)}")
    print(f"MeViS video points: {len(mevis_video)}")
    print(f"Combined points: {len(combined)}")

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
