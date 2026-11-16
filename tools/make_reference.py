"""AUTHORS ONLY: rebuild reference/expected_results.json from the original result files of the project folder.
Readers never need this: the JSON is committed.

  python tools/make_reference.py <path to the original v0.6 folder>

It uses pipeline.report.collect, i.e. the same reading and scoring code as `run.py report`."""
import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import report   # noqa: E402

V06 = Path(sys.argv[1] if len(sys.argv) > 1 else '.').resolve()
WHERE = {   # workspace name -> original file (rounds 5-8 ran on Colab; their results were copied back into bundles)
    'crop_ablation_r5.json': V06 / 'colab_bundle_round5/results/crop_ablation_r5.json',
    'crop_ablation_r6.json': V06 / 'colab_bundle_round6/results/crop_ablation_r6.json',
    'crop_ablation_r7.json': V06 / 'colab_bundle_round7/results/crop_ablation_r7.json',
    'look_choice_r8.json': V06 / 'colab_bundle_round8/results/look_choice_r8.json',
}


def path_of(name):
    return str(WHERE.get(name, V06 / 'pilot' / name))


if __name__ == '__main__':
    vals = report.collect(path_of)
    missing = [k for k, *_ in report.ITEMS if k not in vals]
    out = dict(note='Our numbers (2026-11-05 .. 11-09). Twin accuracy %, [95% pair-bootstrap range]; differences in points.',
               sources={k: os.path.relpath(path_of(n), V06) for k, _, n, _ in report.ITEMS},
               values=vals)
    dst = Path(__file__).resolve().parent.parent / 'reference' / 'expected_results.json'
    json.dump(out, open(dst, 'w'), indent=1)
    print(f'{len(vals)} values written to {dst}; missing: {missing}')
