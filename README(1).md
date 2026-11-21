# Adaptive Defense in a Repeated Attacker–Defender Security Game

A tabular reinforcement-learning study of how to allocate a limited defense budget across vulnerable targets when an attacker may be either **fixed and greedy** or **adaptive**.

The project compares two increasingly interactive settings:

- **Static attacker:** a Q-learning defender faces a deterministic, myopic attacker. Fixing the attacker makes the defender's problem a stationary Markov decision process (MDP), which can be solved exactly with dynamic programming (DP).
- **Adaptive attacker:** the defender and attacker are independent tabular Q-learners. The attacker observes the defender's allocation before selecting a target, so both policies adapt to one another in a general-sum stochastic game.

The mathematical formulation, experimental protocol, plots, and discussion are in the [project report](project_report_and_figures/revised_mathematical_formulation_with_experiments_and_discussion.md).

## Game model

There are $K$ targets. Target $i$ has value $V_i$, baseline attack-success probability $p_i$, attack cost $c_i$, and per-turn recovery probability $\rho_i$. The defender has an integer budget $B$, defense effectiveness $\eta$, unit defense cost $\lambda$, and discount factor $\gamma$.

The infection state is $x\in\{0,1\}^K$, where $x_i=1$ denotes an infected target. The defender chooses an allocation $d$ satisfying

$$d_i\in\mathbb{Z}_{\geq 0},\qquad \sum_{i=1}^{K}d_i\leq B.$$

The probability that an attack on target $i$ succeeds is

$$q_i(d_i)=p_i(1-\eta)^{d_i}.$$

Each turn proceeds in the following order: the defender observes the current infection state and allocates defense; the attacker observes both the state and allocation and selects one target; the attack is sampled; then targets infected at the start of the turn may recover. A target infected during the current turn cannot recover in that same turn.

The defender's reward is the negative weighted infection loss and defense expenditure,

$$r_D(x,d)=-\sum_i V_i x_i-\lambda\sum_i d_i.$$

The attacker receives the target's value only when it causes a new infection, minus the attack cost. These rewards are not zero-sum.

## Learning architectures

| | `static` | `dynamic` |
|---|---|---|
| Learning agents | Defender only | Defender and attacker |
| Attacker | Greedy myopic best response to the current allocation | Tabular Q-learner observing state and allocation |
| Defender Q-table | $Q_D(x,d)$ | $Q_D(x,d)$ |
| Attacker Q-table | — | $Q_A(x,d,a)$ |
| Benchmark | Exact value iteration for the fixed-attacker MDP | No equilibrium benchmark; optional best-response DP is available in code |
| Training environment | Stationary for the defender | Non-stationary while both agents learn |

Both learners use epsilon-greedy action selection and tabular Q-learning. In `dynamic`, the attacker bootstraps over the defender's next allocation before maximizing over its own next target. Since both policies change during joint training, the usual single-agent Q-learning convergence guarantees do not apply to that process. The project does not claim to compute a Nash or Stackelberg equilibrium.

## Experimental findings

The report's main testbed uses $K=5$, $B=2$, $\eta=0.20$, $\lambda=0.75$, and $\gamma=0.95$, with four training seeds. Each episode contains 100 turns.

### Static: learning against a fixed attacker

At $\eta=0.20$, Q-learning improves with longer training and outperforms the fixed highest-value allocation, while remaining below the exact DP benchmark.

| Training episodes | DP return | Q-learning return | Highest-value return |
|---:|---:|---:|---:|
| 10,000 | $-262.81\pm1.36$ | $-294.89\pm7.66$ | $-299.83\pm1.18$ |
| 50,000 | $-262.81\pm1.36$ | $-288.96\pm11.06$ | $-299.83\pm1.18$ |

*Mean finite-horizon discounted defender return ± standard deviation across four seeds. The exact infinite-horizon DP value at the all-healthy state is $-265.19$. The 500-episode reference run uses $\eta=0.10$, so it is not directly comparable to these rows.*

![Static policy comparison](project_report_and_figures/report_figures/01_static_policy_comparison.png)

### Dynamic: joint learning

Against the same seed's frozen learned attacker, the learned defender achieves a higher discounted return than the fixed highest-value baseline at every reported training length. It also reduces cumulative infection loss and defense expenditure in each comparison.

| Training episodes | Learned defender return | Highest-value return |
|---:|---:|---:|
| 500 | $-230.57\pm16.32$ | $-260.67\pm25.67$ |
| 10,000 | $-271.48\pm11.04$ | $-281.24\pm10.06$ |
| 50,000 | $-273.75\pm24.44$ | $-287.11\pm7.54$ |

*Mean discounted defender return ± standard deviation across four seeds. The 500-episode attacker is weaker on average than the longer-trained attackers, so these rows compare different stages of joint adaptation rather than isolating the effect of training duration.*

![Dynamic policy comparison](project_report_and_figures/report_figures/09_dynamic_policy_returns.png)

Cross-play tests show that performance depends on the opponent: for example, the 10,000-episode dynamic-trained defender's return changes from $-271.48$ against its paired learned attacker to $-290.12$ against the greedy attacker. The learned policies should therefore not be interpreted as robust against arbitrary attack strategies.

![Cross-play results](project_report_and_figures/report_figures/13_crossplay_defender_returns.png)

## Repository layout

```text
.
├── README.md
├── proj/
│   ├── static/
│   │   ├── config.py       # Game, training, evaluation, and sweep settings
│   │   ├── game.py         # Environment and fixed greedy attacker
│   │   ├── algorithms.py   # Value iteration and defender Q-learning
│   │   ├── main.py         # Experiments, evaluation, plots, and CSVs
│   │   └── expts/          # Archived static experiment outputs
│   ├── dynamic/
│   │   ├── config.py       # Game and two-agent training settings
│   │   ├── game.py         # Environment with explicit attacker actions
│   │   ├── algorithms.py   # Independent defender/attacker Q-learning
│   │   ├── main.py         # Joint experiments, evaluation, plots, and CSVs
│   │   ├── README.md
│   │   └── expts/          # Archived dynamic experiment outputs
│   └── cross_play/
│       ├── evaluate_greedy.py
│       ├── evaluate_static_defender_vs_rl_attacker.py
│       └── ...             # Saved cross-play result tables
└── project_report_and_figures/
    ├── revised_mathematical_formulation_with_experiments_and_discussion.md
    └── report_figures/     # Figures used in the report
```

## Setup

**Requirements:** Python 3.10 or newer and NumPy. Matplotlib is optional; it is used to generate plots, while the simulations and CSV outputs can run without it.

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install numpy matplotlib
```

There is no repository-level `requirements.txt`; the dependencies above are the direct runtime dependencies used by the experiment code.

## Run the experiments

Run each architecture from its own directory so its local `config.py`, `game.py`, and `algorithms.py` are imported.

**Static attacker / single-agent defender**

```bash
cd proj/static
python main.py
```

This computes the exact DP solution, trains the defender for the configured seeds, evaluates DP/Q-learning/heuristic policies, and writes result tables, checkpoints, and plots under `proj/static/outputs/`.

**Dynamic attacker / two-agent learning**

```bash
cd proj/dynamic
python main.py
```

This trains both agents, evaluates the learned defender against its paired frozen attacker and fixed-allocation baselines, and writes outputs under `proj/dynamic/outputs/`. An exact defender best-response calculation against the frozen learned attacker can be enabled with `run_best_response_dp=True` in `proj/dynamic/config.py`; this is a best response, not an equilibrium computation.

To enable the parameter sweeps defined in a configuration file, set `RUN_PARAMETER_SWEEPS = True` in that architecture's `config.py` and adjust `PARAMETER_SWEEP` as needed.

## Configuration and reproducibility

Edit `config.py` in the relevant architecture to change the number of targets (`K`), budget (`B`), target values, attack and recovery probabilities, learning rates, exploration schedules, episode lengths, evaluation settings, and random seeds. The per-target arrays (`values`, `attack_probs`, `attack_costs`, and `recovery_probs`) must each contain exactly `K` entries.

**The source defaults are not the report defaults.** The current `static/config.py` and `dynamic/config.py` start with `K=10`, `B=5`, 500 training episodes, and five seeds. The report's principal experiments instead use `K=5`, `B=2`, four seeds (`42`, `169`, `6969`, `696969`), and training lengths of 500, 10,000, or 50,000 episodes. Some static runs also use different defense-effectiveness and evaluation settings. The archived `config.json` files under each `expts/` run directory record the settings used for those saved experiments.

The state space has $2^K$ states and the number of legal defense allocations is $\binom{K+B}{B}$. Consequently, the tabular action spaces grow quickly as `K` and `B` increase, especially for the attacker table in `dynamic`.

## Outputs

Each fresh run writes:

- `outputs/config.json` — exact configuration used for the run.
- `outputs/results/` — per-seed and aggregate CSV summaries.
- `outputs/checkpoints/` — saved Q-tables; `static` also saves the DP solution.
- `outputs/figures/` — learning curves and policy-comparison plots (when Matplotlib is installed).

The repository also includes archived sweeps and plots under `proj/static/expts/` and `proj/dynamic/expts/`, plus cross-play CSVs under `proj/cross_play/`.

### Cross-play helper

The `evaluate_greedy.py` helper evaluates a trained dynamic defender against the original greedy attacker. After training the dynamic agents, run it from the dynamic directory:

```bash
cd proj/dynamic
python ../cross_play/evaluate_greedy.py
```

The `evaluate_static_defender_vs_rl_attacker.py` helper is also included. Its current implementation hard-codes checkpoint paths (the command-line options shown in its docstring are not implemented) and contains a duplicated script entry point; adjust these before running it with separate static and dynamic checkpoint directories. Archived cross-play results are included in the repository.

## Scope and limitations

The experiments are tabular and use four training seeds, a 100-turn evaluation horizon, and an all-healthy initial state. The state/action spaces become expensive as the number of targets or defense budget grows. In the dynamic setting, simultaneous adaptation makes the environment non-stationary during training; no equilibrium is computed, and cross-play covers only a small set of opponent swaps. Results should be read as empirical findings for the tested configurations, not general guarantees about optimal or robust defense.
