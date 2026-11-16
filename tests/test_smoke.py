"""Fast CPU checks (about 30 s): python -m pytest tests -q
They check the wiring, not the science: every stage builds, every script it calls exists, the report reader works,
and two data splits regenerate bit-identically to ours."""
import hashlib, json, os, subprocess, sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pipeline import config, report, stages  # noqa: E402


@pytest.mark.parametrize('quick', [False, True])
def test_every_stage_builds_and_scripts_exist(quick):
    for name, make in stages.STAGES.items():
        for s in make(quick):
            assert s.name and s.what
            if isinstance(s.cmd, list) and s.cmd[0].endswith('.py'):
                assert (config.SRC / s.cmd[0]).exists(), f'{name}/{s.name}: missing src/{s.cmd[0]}'


def test_reference_files_are_complete():
    ref = json.load(open(config.REFERENCE / 'expected_results.json'))['values']
    assert set(ref) == {k for k, *_ in report.ITEMS}
    fp = json.load(open(config.REFERENCE / 'data_fingerprints.json'))
    assert set(fp) == set(config.SPLITS)


def test_report_reader_and_verdict(tmp_path):
    f = tmp_path / 'x.json'
    json.dump({'diff': {'eval10/ca/crop-global': [3.12, [-0.31, 6.45]]}, 'acc': {'eval12/none': [61.8, [56.6, 67.3]]}}, open(f, 'w'))
    assert report._json(f, 'diff/eval10/ca/crop-global')[0] == 3.12
    assert report._json(f, 'acc/eval12/none')[1] == [56.6, 67.3]
    assert report.verdict([62.0, None], [61.8, [56.6, 67.3]]) == 'match'
    assert report.verdict([50.0, None], [61.8, [56.6, 67.3]]) == 'CHECK'


def _fingerprint(data, split):
    h = hashlib.sha256()
    for i in range(4):
        z = np.load(data / split / f'{i:05d}.npz')
        h.update(z['frames'].tobytes()); h.update(str(z['meta']).encode())
    return h.hexdigest()[:16]


def test_videos_regenerate_bit_identically(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(config.SRC), SPATIAL_LLM_WORK=str(tmp_path))
    subprocess.run([sys.executable, '-c', "import sts; sts.gen('eval3', 4, workers=2)"], cwd=tmp_path, env=env, check=True)
    subprocess.run([sys.executable, str(config.SRC / 'scenes_modes.py'), 'eval13', '2'], cwd=tmp_path, env=env, check=True)
    ref = json.load(open(config.REFERENCE / 'data_fingerprints.json'))
    for sp in ('eval3', 'eval13'):
        assert _fingerprint(tmp_path / 'sts_data', sp) == ref[sp], sp
