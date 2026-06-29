"""
=============================================================================
pipeline_runner.py — Master Orchestration Script
=============================================================================
Runs the full training pipeline end-to-end in the correct order:

  Step 1 → Preprocessing (SMOTE, PCA, Scaler)
  Step 2 → Ensemble Training (RF + XGB + DNN)
  Step 3 → Autoencoder Training (Zero-Day detection)
  Step 4 → Evaluation (Confusion matrix, ROC, Ablation)
  Step 5 → SHAP Explanation

Usage:
  cd <project_root>
  python pipeline_runner.py            # Run all steps
  python pipeline_runner.py --from 3   # Resume from step 3
  python pipeline_runner.py --only 4   # Run only step 4 (Evaluation)
=============================================================================
"""

import os
import sys
import time
import argparse
import subprocess

# ── Windows UTF-8 fix ─────────────────────────────────────────────────────────
# Prevent UnicodeEncodeError when printing box-drawing chars on cp1252 terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
os.environ["PYTHONUTF8"] = "1"   # propagate to subprocesses

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    (1, "Preprocessing",                 "pipeline/01_preprocess.py"),
    (2, "Ensemble Training (RF+XGB+DNN)","pipeline/02_train_ensemble.py"),
    (3, "Autoencoder Training",          "pipeline/04_train_autoencoder.py"),
    (4, "Evaluation & Visualization",    "pipeline/05_evaluate.py"),
    (5, "SHAP Explainability",           "pipeline/06_explain.py"),
]


def run_step(step_num: int, name: str, script: str) -> bool:
    script_path = os.path.join(BASE_DIR, script)
    if not os.path.exists(script_path):
        print(f"  [ERROR] Script not found: {script_path}")
        return False

    print(f"\n{'═' * 65}")
    print(f"  STEP {step_num} / {len(STEPS)}  ──  {name}")
    print(f"{'═' * 65}")
    t0      = time.perf_counter()
    result  = subprocess.run(
        [sys.executable, script_path],
        cwd=BASE_DIR, text=True,
    )
    elapsed = round(time.perf_counter() - t0, 1)

    if result.returncode == 0:
        print(f"\n  ✅  Step {step_num} complete in {elapsed}s")
        return True
    else:
        print(f"\n  ❌  Step {step_num} FAILED (exit {result.returncode}) after {elapsed}s")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="HyperGuard Training Pipeline Runner")
    parser.add_argument("--from", dest="from_step", type=int, default=1,
                        help="Start from this step number (default: 1)")
    parser.add_argument("--only", dest="only_step", type=int, default=None,
                        help="Run only this step number")
    parser.add_argument("--skip-eval",   action="store_true",
                        help="Skip step 5 (evaluation plots)")
    parser.add_argument("--skip-shap",   action="store_true",
                        help="Skip step 6 (SHAP — slow on large datasets)")
    args = parser.parse_args()

    print("\n" + "═" * 65)
    print("  HyperGuard AI — Hybrid Malware Detection Training Pipeline")
    print("═" * 65)
    print(f"\n  Steps will run in: {BASE_DIR}")

    overall_start = time.perf_counter()
    failures      = []

    for num, name, script in STEPS:
        if args.only_step is not None and num != args.only_step:
            continue
        if num < args.from_step:
            print(f"  Skipping step {num} ({name})")
            continue
        if args.skip_eval and num == 4:
            print(f"  Skipping step 4 (--skip-eval)")
            continue
        if args.skip_shap and num == 5:
            print(f"  Skipping step 5 (--skip-shap)")
            continue

        ok = run_step(num, name, script)
        if not ok:
            failures.append(num)
            print(f"\n  Pipeline halted at step {num}.")
            break

    total = round(time.perf_counter() - overall_start, 1)

    print("\n" + "═" * 65)
    if failures:
        print(f"  ❌  Pipeline failed at step(s): {failures}")
    else:
        print(f"  🎉  All pipeline steps complete in {total}s")
        print(f"\n  To start the web application:")
        print(f"      python app/app.py")
        print(f"\n  API will be available at:")
        print(f"      http://127.0.0.1:5000")
    print("═" * 65 + "\n")


if __name__ == "__main__":
    main()
