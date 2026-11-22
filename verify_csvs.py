"""Recompute the report's tables from the per-seed CSVs (no training needed).

Run from the folder that contains static/, dynamic/, cross_play/:
    python verify_csvs.py [path]
Prints mean ± sample SD (ddof=1) across seeds, to compare against the report.
"""
import sys, glob
from pathlib import Path
import pandas as pd

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
def ms(s): return f"{s.mean():.2f}±{s.std(ddof=1):.2f}"

print("== static: mean discounted return and PoL ==")
for f in sorted(glob.glob(str(root / "static/expts/*/*/results_by_seed.csv"))):
    s = pd.read_csv(f); name = "/".join(Path(f).parts[-3:-1])
    for pol in ("dynamic_programming", "q_learning", "highest_value"):
        g = s[s.policy == pol]
        print(f"{name:26s}{pol:20s} return {ms(g.mean_discounted_return):>14s}  PoL {ms(g.policy_loss)}")

print("\n== dynamic: learned vs highest-value ==")
for f in sorted(glob.glob(str(root / "dynamic/expts/*/*/results_by_seed.csv"))):
    s = pd.read_csv(f); name = "/".join(Path(f).parts[-3:-1])
    for pol in ("q_learning_defender", "highest_value"):
        g = s[s.policy == pol]
        if len(g): print(f"{name:42s}{pol:20s} J_D {ms(g.mean_discounted_return):>14s} J_A {ms(g.mean_discounted_attacker_return):>12s} "
                         f"loss {ms(g.mean_cumulative_loss)} cost {ms(g.mean_defense_cost)} succ {ms(g.attack_success_rate)}")

print("\n== cross-play ==")
for f in sorted(glob.glob(str(root / "cross_play/*/*/*by_seed*.csv"))):
    s = pd.read_csv(f)
    print(f"{'/'.join(Path(f).parts[-3:-1]):40s} defender {ms(s.mean_discounted_return)}  attacker {ms(s.mean_discounted_attacker_return)}  (n={len(s)})")
