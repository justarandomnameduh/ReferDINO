from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from visualize.common import (  # noqa: E402
    build_context,
    build_ranked_tables,
    ensure_dir,
    load_davis_results,
    load_mevis_metadata,
    load_mevis_results,
    print_context_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print and save ranked bad-case tables for DAVIS and MeViS.")
    parser.add_argument("--version", default="swinb_repro")
    parser.add_argument("--mevis-split", default="valid_u")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--save-dir", default=None, help="Directory for CSV exports. Default: output/visualize/<version>/tables")
    parser.add_argument("--top-n", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ctx = build_context(version=args.version, mevis_split=args.mevis_split, output_root=args.output_root)
    print_context_summary(ctx)

    davis_raw, davis_video, davis_disagreement = load_davis_results(ctx)
    mevis_meta = load_mevis_metadata(ctx)
    mevis_expr, mevis_video = load_mevis_results(ctx, meta=mevis_meta)
    tables = build_ranked_tables(
        davis_raw=davis_raw,
        davis_video=davis_video,
        davis_disagreement=davis_disagreement,
        mevis_expr=mevis_expr,
        mevis_video=mevis_video,
        top_n=args.top_n,
    )

    save_dir = Path(args.save_dir).resolve() if args.save_dir else ctx.save_root / "tables"
    ensure_dir(save_dir)

    for name, df in tables.items():
        csv_path = save_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n=== {name} ===")
        print(df.to_string(index=False))
        print(f"Saved {name} to {csv_path}")


if __name__ == "__main__":
    main()
