"""Cross-play evaluation using checkpoints produced by static/ and dynamic/ runs.

  A) dynamic-trained defender  vs  greedy attacker
  B) static-trained defender   vs  frozen dynamic RL attacker

Both use the report testbed (K=5, B=2, eta=0.2) and pair policies by training seed.

Example (run from this folder, after the two training runs below):
    (cd ../static  && python run_report.py --episodes 500 --out runs/500ep_eta0.2)
    (cd ../dynamic && python run_report.py --episodes 500 --out runs/500ep)
    python run_crossplay.py --static-ckpt ../static/runs/500ep_eta0.2/checkpoints \
                            --dynamic-ckpt ../dynamic/runs/500ep/checkpoints --out results_500ep
NOTE: the static run used for cross-play must have eta=0.2 (the report's 500-episode
cross-play uses a different static policy from its eta=0.1 reference run).
"""
import argparse, sys
from dataclasses import replace
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dynamic"))
import config as C
from game import SecurityGame
from algorithms import q_learning_policy, attacker_q_policy, greedy_attacker_policy
from main import evaluate_policy, aggregate_results, save_csv

p = argparse.ArgumentParser()
p.add_argument("--static-ckpt", type=Path, required=True, help="dir with q_seed_<seed>.npz")
p.add_argument("--dynamic-ckpt", type=Path, required=True, help="dir with multi_agent_q_seed_<seed>.npz")
p.add_argument("--out", type=Path, required=True)
p.add_argument("--eval-episodes", type=int, default=10000)
p.add_argument("--seeds", type=int, nargs="+", default=[42, 169, 6969, 696969])
a = p.parse_args()

c = C.CONFIG; K = 5
cfg = replace(c, K=K, B=2, eta=0.2, defense_cost_lambda=0.75,
              values=c.values[:K], attack_probs=c.attack_probs[:K],
              attack_costs=c.attack_costs[:K], recovery_probs=c.recovery_probs[:K],
              eval_episodes=a.eval_episodes, seeds=tuple(a.seeds))
env = SecurityGame(cfg, seed=cfg.seeds[0])
states, actions = env.states, env.actions
s2i = {s: i for i, s in enumerate(states)}
a2i = {x: i for i, x in enumerate(actions)}

rowsA, rowsB = [], []
for sd in cfg.seeds:
    z = np.load(a.dynamic_ckpt / f"multi_agent_q_seed_{sd}.npz")
    Qd, Qa = z["Q_defender"], z["Q_attacker"]
    Qs = np.load(a.static_ckpt / f"q_seed_{sd}.npz")["Q"]
    mA = evaluate_policy(config=cfg,
        defender_policy=lambda s, q=Qd: q_learning_policy(s, q, s2i, actions),
        attacker_policy=lambda s, d: greedy_attacker_policy(cfg, s, d), seed=sd + 100000)
    rowsA.append({"policy": "q_learning_defender_vs_greedy_attacker", "training_seed": int(sd),
                  "evaluation_seed": int(sd + 100000), **mA})
    mB = evaluate_policy(config=cfg,
        defender_policy=lambda s, q=Qs: q_learning_policy(s, q, s2i, actions),
        attacker_policy=lambda s, d, q=Qa: attacker_q_policy(s, d, q, s2i, a2i), seed=sd + 200000)
    rowsB.append({"policy": "static_trained_defender_vs_rl_attacker", "defender_training_seed": int(sd),
                  "attacker_training_seed": int(sd), "evaluation_seed": int(sd + 200000), **mB})
    print("seed", sd, "done", flush=True)

d1 = a.out / "rl def v static att"; d2 = a.out / "static_def v rl attacker"
save_csv(d1 / "rl_defender_vs_greedy_attacker_by_seed.csv", rowsA)
save_csv(d1 / "rl_defender_vs_greedy_attacker_summary.csv", aggregate_results(rowsA))
inp = [{"policy": r["policy"], "training_seed": r["defender_training_seed"],
        "evaluation_seed": r["evaluation_seed"],
        **{k: v for k, v in r.items() if k not in ("policy", "defender_training_seed",
                                                   "attacker_training_seed", "evaluation_seed")}} for r in rowsB]
save_csv(d2 / "static_trained_defender_vs_rl_attacker_by_seed.csv", rowsB)
save_csv(d2 / "static_trained_defender_vs_rl_attacker_summary.csv", aggregate_results(inp))
for name, rows in (("dynamic-def vs greedy-att", rowsA), ("static-def vs dynamic-att", rowsB)):
    r = np.array([[x["mean_discounted_return"], x["mean_discounted_attacker_return"]] for x in rows])
    sd_ = r.std(axis=0, ddof=1) if len(r) > 1 else [0, 0]
    print(f"{name}: defender {r[:,0].mean():.2f} ± {sd_[0]:.2f} | attacker {r[:,1].mean():.2f} ± {sd_[1]:.2f}")
print("CSVs written to", a.out)
