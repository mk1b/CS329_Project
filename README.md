# Adaptive Defense in a Repeated Attacker–Defender Security Game

Code, raw results, and reproduction scripts for the FAI final report
*"Adaptive Defense in a Repeated Attacker–Defender Security Game"*.

A defender splits a budget of `B` units across `K` targets each turn; an attacker picks one target.
Targets get infected (and later recover). We study two settings:

| Folder | Attacker | Defender | Reference |
|---|---|---|---|
| `static/` | fixed greedy myopic best response | tabular Q-learning | exact value iteration (DP) |
| `dynamic/` | independent tabular Q-learner that sees the defender's allocation | tabular Q-learning | none (general-sum game, no equilibrium computed) |
| `cross_play/` | swaps opponents: dynamic-trained defender vs greedy attacker, static-trained defender vs frozen dynamic attacker | – | – |

Each of `static/expts`, `dynamic/expts`, `cross_play/` already contains the raw per-seed and summary CSVs behind every table in the report.

## 1. Setup

Python 3.10+ (the code uses `X | None` type syntax).

```bash
pip install -r requirements.txt     # numpy (required); pandas (verify script); matplotlib (optional plots)
```

## 2. Quickest check: recompute the report tables from the shipped CSVs (seconds)

```bash
python verify_csvs.py .
```

Prints mean ± sample SD (across the 4 training seeds) for every static, dynamic and cross-play run,
to compare with the report tables.

## 3. Smoke test (about a minute)

```bash
cd static  && python run_report.py --episodes 500 --eval-episodes 50 --seeds 42 --out runs/smoke --no-plots
cd ../dynamic && python run_report.py --episodes 500 --eval-episodes 50 --seeds 42 --out runs/smoke --no-plots
```

If both finish and print a results table, the environment works.

## 4. Reproducing the report experiments

> Use the `run_report.py` wrappers below,
> which apply the report configuration (`K=5, B=2, η=0.20, λ=0.75, γ=0.95, α=0.10`, 100-step episodes,
> seeds `42 169 6969 696969`, ε decays 1.0→0.05 over 20% of training steps = 10k / 200k / 1M steps for
> 500 / 10k / 50k episodes) without editing any file. Each experiment's exact settings are also saved
> as `config.json` in `static/expts/*` and `dynamic/expts/*`.

All commands run from inside the named folder. Outputs go to `--out` (`results/*.csv`, `checkpoints/*.npz`, `figures/`).

### 4.1 Static (fixed greedy attacker)

```bash
cd static
python run_report.py --episodes 500   --eta 0.1 --out runs/500ep_eta0.1   # Table 1, 500-episode row (η=0.1)
python run_report.py --episodes 500            --out runs/500ep_eta0.2   # needed for cross-play
python run_report.py --episodes 10000          --out runs/10000ep
python run_report.py --episodes 50000          --out runs/50000ep
```

Static evaluation uses 1,000 rollouts per seed (default). One-factor sweeps (add to the 10k/50k commands):

| Sweep | Flags |
|---|---|
| Budget B ∈ {3, 5} (10k) | `--B 3` / `--B 5` |
| η ∈ {0.05, 0.40} (10k, 50k) | `--eta 0.05` / `--eta 0.4` |
| α_D ∈ {0.05, 0.20, 0.40} (10k) | `--alpha 0.05` etc. |
| Short decay (10k) | `--decay-steps 20000` |
| λ ∈ {0.25, 1.50} (50k) | `--lam 0.25` / `--lam 1.5` |

### 4.2 Dynamic (two learning agents)

```bash
cd dynamic
python run_report.py --episodes 500   --out runs/500ep
python run_report.py --episodes 10000 --out runs/10000ep
python run_report.py --episodes 50000 --out runs/50000ep
```

Baselines use 10,000 evaluation rollouts per seed (default). Report sweeps mostly use 1,000: add `--eval-episodes 1000`.
Sweeps change one learner at a time:

| Sweep | Flags |
|---|---|
| Defender learning rate 0.05 / 0.20 / 0.40 | `--alpha 0.05` … |
| Attacker learning rate 0.05 / 0.20 / 0.40 | `--attacker-alpha 0.05` … |
| Defender ε-decay 0.1× / 2× / 4× | 10k eps: `--decay-steps 20000 / 400000 / 800000`; 50k eps: `100000 / 2000000 / 4000000` |
| Attacker ε-decay 0.1× / 2× / 4× | same values via `--attacker-decay-steps` |

### 4.3 Cross-play

Cross-play needs checkpoints from one static run (**η = 0.20**) and one dynamic run of the same length, then:

```bash
cd cross_play
python run_crossplay.py --static-ckpt ../static/runs/500ep_eta0.2/checkpoints \
                        --dynamic-ckpt ../dynamic/runs/500ep/checkpoints --out results_500ep
```

Writes the same four CSVs (by-seed + summary, both directions) as the shipped `cross_play/*/500ep` folders.
Use `10000ep` / `50000ep` runs for the other rows. Policies are paired by training seed; evaluation seeds are
`seed+100000` (dynamic defender vs greedy) and `seed+200000` (static defender vs dynamic attacker).

### 4.4 Determinism

Runs are seeded, so a given command reproduces the same numbers on the same NumPy version.
Different NumPy versions may change random streams slightly; expect agreement within seed-to-seed noise.

Runtime scales with episodes × seeds × evaluation rollouts. A 500-episode run takes minutes; the 10k and especially 50k-episode
runs (and their 10,000-rollout evaluations) are much longer(~20 mins). Use `--seeds 42` and `--eval-episodes` to cut 2-3 minutes when checking.

## 5. Headline results (mean ± SD over 4 seeds; discounted defender return, higher is better)

| | Episodes | Learned | Highest-value | DP |
|---|---|---|---|---|
| static | 10,000 | −294.89 ± 7.66 | −299.83 ± 1.18 | −262.81 ± 1.36 |
| static | 50,000 | −288.96 ± 11.06 | −299.83 ± 1.18 | −262.81 ± 1.36 |
| dynamic | 10,000 | −271.48 ± 11.04 | −281.24 ± 10.06 | – |
| dynamic | 50,000 | −273.75 ± 24.44 | −287.11 ± 7.54 | – |

With only four seeds, paired per-seed differences between learned and highest-value policies are **not**
statistically significant (paired t-test p ≈ 0.07–0.38); The learned defender also
does not beat highest-value in every static sweep setting (e.g. η = 0.40, α ≥ 0.20).

## 6. Note for checkers

* `static/` and `dynamic/` each define modules named `config`, `game`, `algorithms`, `main`. Run scripts from inside their own folder
  (as above); `cross_play/run_crossplay.py` imports from `../dynamic` automatically.

## 7. Repository layout

```
requirements.txt
verify_csvs.py              # recompute report tables from shipped CSVs
static/      config.py game.py algorithms.py main.py run_report.py   expts/<epochs>/<setting>/
dynamic/     config.py game.py algorithms.py main.py run_report.py README.md  expts/<epochs>/<setting>/
cross_play/  run_crossplay.py evaluate_greedy.py evaluate_static_defender_vs_rl_attacker.py  <direction>/<epochs>/*.csv
```

## Authors

Bhaumik Malaviya, Laxmidhar Panda, Bhakti Prasad Swain, and Mateen Sharief.
