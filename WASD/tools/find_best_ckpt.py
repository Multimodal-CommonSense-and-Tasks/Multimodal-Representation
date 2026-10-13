"""Print the checkpoint step with the best validation ROUGE-L of a training run.

Usage: python tools/find_best_ckpt.py results/gpt2/train/<run>

The run directory contains one numeric sub-directory per saved checkpoint and a
log.txt with one "dev | ... 'rougeL': <score>" line per evaluation. Evaluations
that precede the first checkpoint (e.g. the one before training) are skipped.
"""
import argparse
import re
import sys
from pathlib import Path

ROUGE_PATTERN = re.compile(r"dev\s*\|.*?['\"]rougeL['\"]\s*:\s*([0-9]+(?:\.[0-9]+)?)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="training run directory (results/gpt2/train/<run>)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    log_path = run_dir / "log.txt"
    if not run_dir.is_dir():
        sys.exit(f"run directory not found: {run_dir}")
    if not log_path.is_file():
        sys.exit(f"log.txt not found: {log_path}")

    steps = sorted(int(p.name) for p in run_dir.iterdir() if p.is_dir() and p.name.isdigit())
    if not steps:
        sys.exit(f"no checkpoint directories found in {run_dir}")

    scores = []
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = ROUGE_PATTERN.search(line)
            if m:
                scores.append(float(m.group(1)))
    if len(scores) < len(steps):
        sys.exit(f"fewer dev rougeL records ({len(scores)}) than checkpoints ({len(steps)}) in {log_path}")

    scores = scores[len(scores) - len(steps):]
    best_step, best_score = max(zip(steps, scores), key=lambda x: x[1])

    for step, score in zip(steps, scores):
        print(f"step {step:>8d} | dev rougeL {score:.4f}" + ("  <- best" if step == best_step else ""), file=sys.stderr)
    print(best_step)


if __name__ == "__main__":
    main()
