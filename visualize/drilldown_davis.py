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
    create_davis_case_figure,
    load_davis_results,
    print_context_summary,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a DAVIS frame-level drill-down panel.")
    parser.add_argument("--version", default="swinb_repro")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--annotator", default="anno_0")
    parser.add_argument("--num-frames", type=int, default=6)
    parser.add_argument("--object-id", type=int, default=None)
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_plot_style()
    ctx = build_context(version=args.version, output_root=args.output_root)
    print_context_summary(ctx)

    _, davis_video, _ = load_davis_results(ctx)
    fig = create_davis_case_figure(
        ctx=ctx,
        davis_video_df=davis_video,
        source_video=args.source_video,
        annotator=args.annotator,
        num_frames=args.num_frames,
        object_id=args.object_id,
    )
    default_name = f"davis_{args.annotator}_{args.source_video}.png"
    save_path = Path(args.save_path).resolve() if args.save_path else ctx.save_root / "drilldowns" / default_name
    save_figure(fig, save_path, close=not args.show)
    print(f"Saved DAVIS drill-down to {save_path}")

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
