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
    create_mevis_case_figure,
    load_mevis_mask_dict,
    load_mevis_metadata,
    load_mevis_results,
    print_context_summary,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a MeViS frame-level drill-down panel.")
    parser.add_argument("--version", default="swinb_repro")
    parser.add_argument("--mevis-split", default="valid_u")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--expression-id", default=None, help="Optional. If omitted, choose the worst expression for the video.")
    parser.add_argument("--num-frames", type=int, default=6)
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_plot_style()
    ctx = build_context(version=args.version, mevis_split=args.mevis_split, output_root=args.output_root)
    print_context_summary(ctx)

    mevis_meta = load_mevis_metadata(ctx)
    mevis_mask_dict = load_mevis_mask_dict(ctx)
    mevis_expr, _ = load_mevis_results(ctx, meta=mevis_meta)
    fig = create_mevis_case_figure(
        ctx=ctx,
        mevis_expr_df=mevis_expr,
        source_video=args.source_video,
        expression_id=args.expression_id,
        num_frames=args.num_frames,
        mevis_meta=mevis_meta,
        mevis_mask_dict=mevis_mask_dict,
    )
    expr_suffix = f"expr_{args.expression_id}" if args.expression_id is not None else "worst_expr"
    default_name = f"mevis_{args.source_video}_{expr_suffix}.png"
    save_path = Path(args.save_path).resolve() if args.save_path else ctx.save_root / "drilldowns" / default_name
    save_figure(fig, save_path, close=not args.show)
    print(f"Saved MeViS drill-down to {save_path}")

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
