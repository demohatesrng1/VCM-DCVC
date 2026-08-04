"""Summarise an LRDO sweep: BD-rate plus the diagnostics that explain it.

    python scripts/lrdo_report.py --sweep_dir ./sweep

Reads ``base.json`` (no LRDO) and every ``i*_lr*.json`` beside it, and prints one
row per configuration.

Columns:
    BD-rate     per sequence and averaged. Negative = LRDO wins.
    travel      iters*lr, roughly how far Adam can move each latent element.
    symbols     fraction of coded symbols that actually changed. This is the one
                to read first: if it is ~0, the optimisation never crossed a
                quantisation boundary, so the BD-rate says nothing about whether
                LRDO helps -- only that the step budget was too small.
    |dy|        mean absolute latent displacement actually achieved.
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Read_and_Plot'))
from bjontegaard_metric import BD_RATE                          # noqa: E402


def curves(doc, ds, seq):
    keys = sorted(doc[ds][seq])
    rate = [doc[ds][seq][k]['ave_all_frame_bpp'] for k in keys]
    psnr = [doc[ds][seq][k]['ave_all_frame_psnr'] for k in keys]
    return rate, psnr


def mean_over_rates(doc, ds, seq, field):
    vals = []
    for k in doc[ds][seq]:
        v = doc[ds][seq][k].get(field)
        if isinstance(v, list) and v:
            vals.append(sum(v) / len(v))
        elif isinstance(v, (int, float)):
            vals.append(v)
    return sum(vals) / len(vals) if vals else float('nan')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_dir', default='./sweep')
    args = parser.parse_args()

    base_path = os.path.join(args.sweep_dir, 'base.json')
    if not os.path.isfile(base_path):
        raise SystemExit(f"missing {base_path} -- run scripts/lrdo_sweep.sh first")
    base = json.load(open(base_path))

    runs = sorted(glob.glob(os.path.join(args.sweep_dir, 'i*_lr*.json')))
    if not runs:
        raise SystemExit(f"no i*_lr*.json in {args.sweep_dir}")

    sequences = [(ds, seq) for ds in base for seq in base[ds]]
    name_w = max(len(s) for _, s in sequences)

    header = f"{'config':>18}  {'travel':>6}  {'symbols':>7}  {'|dy|':>6}  "
    header += "  ".join(f"{s[:name_w]:>{max(name_w, 7)}}" for _, s in sequences)
    header += f"  {'AVG':>7}"
    print(header)
    print("-" * len(header))

    for path in runs:
        doc = json.load(open(path))
        tag = os.path.splitext(os.path.basename(path))[0]

        iters = lr = None
        for ds, seq in sequences:
            for k in doc[ds][seq]:
                iters = doc[ds][seq][k].get('lrdo_iters', iters)
                lr = doc[ds][seq][k].get('lrdo_lr', lr)
        travel = (iters * lr) if (iters and lr) else float('nan')

        bd_rates = []
        for ds, seq in sequences:
            r1, p1 = curves(base, ds, seq)
            r2, p2 = curves(doc, ds, seq)
            bd_rates.append(BD_RATE(r1, p1, r2, p2))

        changed = sum(mean_over_rates(doc, ds, seq, 'lrdo_frac_symbols_changed')
                      for ds, seq in sequences) / len(sequences)
        dy = sum(mean_over_rates(doc, ds, seq, 'lrdo_mean_abs_dy')
                 for ds, seq in sequences) / len(sequences)

        row = f"{tag:>18}  {travel:6.2f}  {changed:6.2%}  {dy:6.3f}  "
        row += "  ".join(f"{b:>{max(name_w, 7)}.2f}" for b in bd_rates)
        row += f"  {sum(bd_rates)/len(bd_rates):7.2f}"
        print(row)

    print("\nBD-rate is vs the no-LRDO baseline; negative = LRDO wins.")
    print("Read 'symbols' first: near 0% means the latent never crossed a")
    print("quantisation boundary, so that row's BD-rate is uninformative.")


if __name__ == '__main__':
    raise SystemExit(main())
