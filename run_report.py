"""Run a static-attacker experiment with the configuration used in the report.

The defaults in config.py (K=10, B=5, ...) are NOT the report testbed. This
script overrides them with the report testbed (K=5, B=2, 4 seeds) so results
can be reproduced without editing any file.

Examples (run from this folder):
    python run_report.py --episodes 500   --eta 0.1 --out runs/500ep_eta0.1
    python run_report.py --episodes 10000 --out runs/10000ep
    python run_report.py --episodes 50000 --out runs/50000ep
    python run_report.py --episodes 500 --eval-episodes 50 --seeds 42 --out runs/smoke
"""
import argparse
from dataclasses import replace
from pathlib import Path

import config as C
from main import run_experiments

p = argparse.ArgumentParser()
p.add_argument("--episodes", type=int, required=True, help="training episodes (500, 10000, 50000)")
p.add_argument("--out", type=Path, required=True)
p.add_argument("--B", type=int, default=2)
p.add_argument("--eta", type=float, default=0.20)
p.add_argument("--lam", type=float, default=0.75, help="defense cost lambda")
p.add_argument("--alpha", type=float, default=0.10)
p.add_argument("--decay-steps", type=int, default=None,
               help="epsilon decay steps (default: 20%% of total training steps = 10k/200k/1M)")
p.add_argument("--eval-episodes", type=int, default=1000)
p.add_argument("--seeds", type=int, nargs="+", default=[42, 169, 6969, 696969])
p.add_argument("--no-plots", action="store_true")
a = p.parse_args()

c = C.CONFIG
K = 5
cfg = replace(
    c, K=K, B=a.B, eta=a.eta, defense_cost_lambda=a.lam, alpha=a.alpha,
    values=c.values[:K], attack_probs=c.attack_probs[:K],
    attack_costs=c.attack_costs[:K], recovery_probs=c.recovery_probs[:K],
    num_episodes=a.episodes, steps_per_episode=100,
    epsilon_decay_steps=a.decay_steps or int(0.2 * a.episodes * 100),
    eval_episodes=a.eval_episodes, seeds=tuple(a.seeds),
)
run_experiments(cfg, output_dir=a.out, make_plots=not a.no_plots, save_checkpoints=True)
