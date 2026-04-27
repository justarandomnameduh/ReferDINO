import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.eval_davis import reconstruct_annotator_results, resolve_eval_output_path


class DavisOutputLayoutTest(unittest.TestCase):
    def test_reconstructs_annotator_composites_from_expression_layout(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            results_path = tmp_path / "Annotations"
            eval_path = tmp_path / "eval_davis" / "val"
            video = "bike-packing"
            frames = ["00000", "00001"]
            expressions = {
                str(exp_id): {"exp": f"query {exp_id}"}
                for exp_id in range(8)
            }
            data = {
                video: {
                    "expressions": expressions,
                    "frames": frames,
                }
            }

            for exp_id in expressions:
                exp_path = results_path / video / exp_id
                exp_path.mkdir(parents=True, exist_ok=True)
                mask = np.zeros((3, 3), dtype=np.uint8)
                mask[0, 0] = 255 if int(exp_id) < 4 else 0
                mask[2, 2] = 255 if int(exp_id) >= 4 else 0
                for frame in frames:
                    Image.fromarray(mask).save(exp_path / f"{frame}.png")

            anno_paths = reconstruct_annotator_results(results_path, data, eval_path, 4, None)

            self.assertEqual([path.name for path in anno_paths], ["anno_0", "anno_1", "anno_2", "anno_3"])
            self.assertFalse((results_path / "anno_0").exists())
            composite = np.array(Image.open(eval_path / "anno_2" / video / "00000.png"))
            self.assertEqual(composite[0, 0], 1)
            self.assertEqual(composite[2, 2], 2)

    def test_default_eval_path_is_sibling_of_annotations(self):
        path = resolve_eval_output_path("/tmp/run/Annotations", "val")
        self.assertEqual(path, Path("/tmp/run/eval_davis/val"))


if __name__ == "__main__":
    unittest.main()
