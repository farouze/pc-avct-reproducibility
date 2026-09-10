#!/usr/bin/env python3
"""Run the frozen V38 model-analysis workflow with explicit local data paths."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('aurora_raw_avct_features.csv', 'uci_pc_features.csv')

def check_inputs(data_dir):
    expected = json.loads((ROOT/'docs/reference_run_manifest.json').read_text())['input_sha256']
    checked = {}
    for name in NAMES:
        p = data_dir/name
        if not p.is_file():
            raise ValueError(f'Missing {p}; see data/README.md for access requirements.')
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest != expected[name]:
            raise ValueError(f'Input hash mismatch for {name}; do not label this a reference reproduction.')
        checked[name] = digest
    return checked

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'runs/reference')
    parser.add_argument('--mode', choices=['check', 'smoke', 'full'], default='check')
    args = parser.parse_args()
    try:
        inputs = check_inputs(args.data_dir.resolve())
    except ValueError as e:
        parser.error(str(e))
    print('Both input hashes match the V38 reference feature banks.', flush=True)
    if args.mode == 'check':
        return
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Output directory is not empty; choose a new directory to preserve earlier runs.')
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(PCAVCT_DATA_DIR=str(args.data_dir.resolve()), PCAVCT_OUTPUT_DIR=str(out), PCAVCT_MODE=args.mode,
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLCONFIGDIR=str(out/'matplotlib'))
    # Store actual environment separately from the reference; a successful run need not be numerically identical across environments.
    (out/'verified_inputs.json').write_text(json.dumps(inputs, indent=2)+'\n')
    subprocess.run([sys.executable, str(ROOT/'analysis/rerun.py')], check=True, env=env)
    if args.mode == 'full':
        subprocess.run([sys.executable, str(ROOT/'analysis/summarize_rerun.py')], check=True, env=env)
    print(f'{args.mode.capitalize()} run completed: {out}', flush=True)
    if args.mode == 'smoke':
        print('Smoke mode covers only the first outer fold and must not be used as paper results.')

if __name__ == '__main__':
    main()
