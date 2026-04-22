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
    build_ranked_tables,
    combine_video_results,
    ensure_dir,
    load_davis_results,
    load_mevis_metadata,
    load_mevis_results,
    plot_davis_raw_sequence_scatter,
    plot_main_video_scatter,
    plot_metric_distributions,
    print_context_summary,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the standard ReferDINO analysis plots and ranking tables.")
    parser.add_argument("--version", default="swinb_repro")
    parser.add_argument("--mevis-split", default="valid_u")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--save-dir", default=None, help="Default: output/visualize/<version>")
    parser.add_argument("--top-n", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_plot_style()
    ctx = build_context(version=args.version, mevis_split=args.mevis_split, output_root=args.output_root, save_root=args.save_dir)
    print_context_summary(ctx)

    davis_raw, davis_video, davis_disagreement = load_davis_results(ctx)
    mevis_meta = load_mevis_metadata(ctx)
    mevis_expr, mevis_video = load_mevis_results(ctx, meta=mevis_meta)
    combined = combine_video_results(davis_video, mevis_video)

    ensure_dir(ctx.save_root)
    tables_dir = ensure_dir(ctx.save_root / "tables")

    fig, _, _ = plot_main_video_scatter(davis_video, mevis_video, label_top_n=3)
    save_figure(fig, ctx.save_root / "video_scatter.png")

    fig, _ = plot_davis_raw_sequence_scatter(davis_raw, label_top_n=2)
    save_figure(fig, ctx.save_root / "davis_raw_sequence_scatter.png")

    fig, _ = plot_metric_distributions(combined)
    save_figure(fig, ctx.save_root / "metric_distributions.png")

    tables = build_ranked_tables(davis_raw, davis_video, davis_disagreement, mevis_expr, mevis_video, top_n=args.top_n)
    for name, df in tables.items():
        csv_path = tables_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {name} to {csv_path}")

    print(f"Saved all standard outputs under {ctx.save_root}")


if __name__ == "__main__":
    main()
