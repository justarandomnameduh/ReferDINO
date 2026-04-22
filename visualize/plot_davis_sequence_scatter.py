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
    plot_davis_raw_sequence_scatter,
    print_context_summary,
    save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DAVIS raw-sequence J-vs-F scatter.")
    parser.add_argument("--version", default="swinb_repro")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--save-path", default=None, help="PNG path. Default: output/visualize/<version>/davis_raw_sequence_scatter.png")
    parser.add_argument("--label-top-n", type=int, default=2)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_plot_style()
    ctx = build_context(version=args.version, output_root=args.output_root)
    print_context_summary(ctx)

    davis_raw, _, _ = load_davis_results(ctx)
    fig, _ = plot_davis_raw_sequence_scatter(davis_raw, label_top_n=args.label_top_n)
    save_path = Path(args.save_path).resolve() if args.save_path else ctx.save_root / "davis_raw_sequence_scatter.png"
    save_figure(fig, save_path, close=not args.show)
    print(f"Saved DAVIS raw-sequence scatter to {save_path}")
    print(f"DAVIS raw sequence points: {len(davis_raw)}")

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
