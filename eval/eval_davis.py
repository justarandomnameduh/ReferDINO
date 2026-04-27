#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from PIL import Image
from tools.davis2017.evaluation import DAVISEvaluation

default_davis_path = 'data/ref_davis/DAVIS'


def davis_meta_split(eval_set):
    return "valid" if eval_set == "val" else eval_set


def resolve_meta_path(davis_path, eval_set, davis_meta_path=None):
    if davis_meta_path:
        return Path(davis_meta_path)
    return Path(davis_path).parent / "meta_expressions" / davis_meta_split(eval_set) / "meta_expressions.json"


def resolve_eval_output_path(results_path, eval_set, eval_output_path=None):
    if eval_output_path:
        return Path(eval_output_path)
    return Path(results_path).parent / "eval_davis" / eval_set


def load_davis_metadata(davis_path, eval_set, davis_meta_path=None):
    meta_path = resolve_meta_path(davis_path, eval_set, davis_meta_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing DAVIS metadata file: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as handle:
        return json.load(handle)["videos"], meta_path


def has_legacy_annotator_layout(results_path, num_anno):
    root = Path(results_path)
    return all((root / f"anno_{idx}").is_dir() for idx in range(num_anno))


def load_palette(davis_path):
    for pattern in ("Annotations_unsupervised/480p/*/*.png", "Annotations/480p/*/*.png"):
        for candidate in Path(davis_path).glob(pattern):
            return Image.open(candidate).getpalette()
    return None


def read_expression_mask(results_path, video, exp_id, frame_name):
    mask_path = Path(results_path) / video / str(exp_id) / f"{frame_name}.png"
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing DAVIS expression mask: {mask_path}")
    return np.array(Image.open(mask_path).convert("L")) > 0


def reconstruct_annotator_results(results_path, data, output_path, num_anno, palette):
    output_path = Path(output_path)
    for anno_id in range(num_anno):
        anno_root = output_path / f"anno_{anno_id}"
        anno_root.mkdir(parents=True, exist_ok=True)

        for video, video_meta in data.items():
            expressions = video_meta["expressions"]
            expression_ids = list(expressions.keys())
            frames = video_meta["frames"]
            num_obj = len(expression_ids) // num_anno
            video_root = anno_root / video
            video_root.mkdir(parents=True, exist_ok=True)

            for frame_name in frames:
                composite = None
                for obj_id in range(num_obj):
                    exp_id = expression_ids[obj_id * num_anno + anno_id]
                    mask = read_expression_mask(results_path, video, exp_id, frame_name)
                    if composite is None:
                        composite = np.zeros(mask.shape, dtype=np.uint8)
                    composite[mask] = obj_id + 1

                if composite is None:
                    raise RuntimeError(f"No expressions found for DAVIS video: {video}")
                image = Image.fromarray(composite)
                if palette:
                    image.putpalette(palette)
                image.save(video_root / f"{frame_name}.png")

    return [output_path / f"anno_{idx}" for idx in range(num_anno)]


def evaluate_annotator(args, anno_results_path, csv_name_global, csv_name_per_sequence):
    time_start = time()
    anno_results_path = Path(anno_results_path)

    csv_name_global_path = anno_results_path / csv_name_global
    csv_name_per_sequence_path = anno_results_path / csv_name_per_sequence
    if csv_name_global_path.exists() and csv_name_per_sequence_path.exists() and not args.re_eval:
        print('Using precomputed results...')
        table_g = pd.read_csv(csv_name_global_path)
        table_seq = pd.read_csv(csv_name_per_sequence_path)
    else:
        print(f'Evaluating sequences for the {args.task} task...')
        dataset_eval = DAVISEvaluation(davis_root=args.davis_path, task=args.task, gt_set=args.set)
        metrics_res = dataset_eval.evaluate(str(anno_results_path))
        J, F = metrics_res['J'], metrics_res['F']

        g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
        final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
        g_res = np.array([final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
                          np.mean(F["D"])])
        g_res = np.reshape(g_res, [1, len(g_res)])
        table_g = pd.DataFrame(data=g_res, columns=g_measures)
        with open(csv_name_global_path, 'w') as f:
            table_g.to_csv(f, index=False, float_format="%.5f")
        print(f'Global results saved in {csv_name_global_path}')

        seq_names = list(J['M_per_object'].keys())
        seq_measures = ['Sequence', 'J-Mean', 'F-Mean']
        J_per_object = [J['M_per_object'][x] for x in seq_names]
        F_per_object = [F['M_per_object'][x] for x in seq_names]
        table_seq = pd.DataFrame(data=list(zip(seq_names, J_per_object, F_per_object)), columns=seq_measures)
        with open(csv_name_per_sequence_path, 'w') as f:
            table_seq.to_csv(f, index=False, float_format="%.5f")
        print(f'Per-sequence results saved in {csv_name_per_sequence_path}')

    sys.stdout.write(f"--------------------------- Global results for {args.set} ---------------------------\n")
    print(table_g.to_string(index=False))
    sys.stdout.write(f"\n---------- Per sequence results for {args.set} ----------\n")
    print(table_seq.to_string(index=False))
    total_time = time() - time_start
    sys.stdout.write('\nTotal time:' + str(total_time))
    return table_g


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--davis_path', type=str, help='Path to the DAVIS folder containing the JPEGImages, Annotations, '
                                                       'ImageSets, Annotations_unsupervised folders',
                        required=False, default=default_davis_path)
    parser.add_argument('--set', type=str, help='Subset to evaluate the results', default='val')
    parser.add_argument('--task', type=str, help='Task to evaluate the results', default='unsupervised',
                        choices=['semi-supervised', 'unsupervised'])
    parser.add_argument('--results_path', type=str, help='Path to the DAVIS prediction Annotations folder',
                        required=True)
    parser.add_argument('--davis_meta_path', type=str, default=None,
                        help='Path to DAVIS meta_expressions.json. Defaults to the sibling ref_davis metadata file.')
    parser.add_argument('--eval_output_path', type=str, default=None,
                        help='Path for reconstructed DAVIS annotator composites and evaluation CSVs.')
    parser.add_argument("--re_eval", action='store_true')
    return parser


def main():
    parser = build_parser()
    args, _ = parser.parse_known_args()
    csv_name_global = f'global_results-{args.set}.csv'
    csv_name_per_sequence = f'per-sequence_results-{args.set}.csv'

    num_anno = 4
    if has_legacy_annotator_layout(args.results_path, num_anno):
        anno_result_paths = [Path(args.results_path) / f"anno_{idx}" for idx in range(num_anno)]
    else:
        data, meta_path = load_davis_metadata(args.davis_path, args.set, args.davis_meta_path)
        eval_output_path = resolve_eval_output_path(args.results_path, args.set, args.eval_output_path)
        print(f"Reconstructing DAVIS annotator composites from {args.results_path}")
        print(f"Using DAVIS metadata: {meta_path}")
        print(f"Saving DAVIS evaluation artifacts to: {eval_output_path}")
        anno_result_paths = reconstruct_annotator_results(
            args.results_path,
            data,
            eval_output_path,
            num_anno,
            load_palette(args.davis_path),
        )

    all_results = []
    for anno_results_path in anno_result_paths:
        table_g = evaluate_annotator(args, anno_results_path, csv_name_global, csv_name_per_sequence)
        all_results.append(table_g)
        print("\n")

    all_results = pd.concat(all_results, axis=0, ignore_index=True)
    all_results.index = [f'an{i}' for i in range(num_anno)]
    avg_results = pd.DataFrame([all_results.mean()], index=['avg'])
    all_results = pd.concat([all_results, avg_results], axis=0, ignore_index=False)
    print(all_results.to_string())


if __name__ == "__main__":
    main()
