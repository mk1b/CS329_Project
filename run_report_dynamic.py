"""Run a two-agent (dynamic) experiment with the configuration used in the report.

Overrides the K=10/B=5 defaults in config.py with the report testbed
(K=5, B=2, 4 seeds). Both agents default to alpha=0.10 and an epsilon decay
of 20% of total training steps (10k / 200k / 1M steps for 500 / 10k / 50k episodes).

Examples (run from this folder):
    python run_report.py --episodes 500   --out runs/500ep
    python run_report.py --episodes 10000 --out runs/10000ep
    python run_report.py --episodes 50000 --out runs/50000ep
    # one-factor sweep, e.g. attacker learning rate 0.2 with 1000 eval rollouts:
    python run_report.py --episodes 10000 --attacker-alpha 0.2 --eval-episodes 1000 --out runs/atk_lr0.2
    python run_report.py --episodes 500 --eval-episodes 50 --seeds 42 --out runs/smoke
"""
import argparse
from dataclasses import replace
from pathlib import Path

import config as C
from main import run_experiments

p = argparse.ArgumentParser()
p.add_argument("--episodes", type=int, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--B", type=int, default=2)
p.add_argument("--eta", type=float, default=0.20)
p.add_argument("--lam", type=float, default=0.75)
p.add_argument("--alpha", type=float, default=0.10, help="defender learning rate")
p.add_argument("--attacker-alpha", type=float, default=0.10)
p.add_argument("--decay-steps", type=int, default=None, help="defender epsilon decay steps")
p.add_argument("--attacker-decay-steps", type=int, default=None)
p.add_argument("--eval-episodes", type=int, default=10000,
               help="10000 for baselines/cross-play, 1000 for most sweeps in the report")
p.add_argument("--seeds", type=int, nargs="+", default=[42, 169, 6969, 696969])
p.add_argument("--no-plots", action="store_true")
a = p.parse_args()

c = C.CONFIG
K = 5
ref_decay = int(0.2 * a.episodes * 100)
cfg = replace(
    c, K=K, B=a.B, eta=a.eta, defense_cost_lambda=a.lam,
    values=c.values[:K], attack_probs=c.attack_probs[:K],
    attack_costs=c.attack_costs[:K], recovery_probs=c.recovery_probs[:K],
    num_episodes=a.episodes, steps_per_episode=100,
    alpha=a.alpha, attacker_alpha=a.attacker_alpha,
    epsilon_decay_steps=a.decay_steps or ref_decay,
    attacker_epsilon_decay_steps=a.attacker_decay_steps or ref_decay,
    eval_episodes=a.eval_episodes, seeds=tuple(a.seeds),
    run_best_response_dp=False,
)
run_experiments(cfg, output_dir=a.out, make_plots=not a.no_plots, save_checkpoints=True)
