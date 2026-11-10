# main.py

import csv
import json
import warnings

from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import numpy as np

from config import (
    CONFIG,
    OUTPUT_DIR,
    RESULTS_DIR,
    CHECKPOINTS_DIR,
    FIGURES_DIR,
    RUN_PARAMETER_SWEEPS,
    PARAMETER_SWEEP,
    GameConfig,
)

from game import (
    SecurityGame,
    State,
    DefenseAction,
)

from algorithms import (
    solve_value_iteration,
    train_q_learning,
    highest_value_policy,
    uniform_policy,
    dp_policy,
    q_learning_policy,
)


# Matplotlib is optional; the simulation and CSV outputs
# still work if it is not installed.

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


PolicyFunction = Callable[[State], DefenseAction]


# =========================================================
# 1. FILE OUTPUT UTILITIES
# =========================================================

def save_csv(path: Path, rows: list[dict]) -> None:
    """Save a list of dictionaries as a CSV file."""

    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def save_config(config: GameConfig, output_dir: Path) -> None:
    """Save the exact configuration used in an experiment."""

    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "config.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(asdict(config), file, indent=2)


# =========================================================
# 2. EVALUATE A SINGLE POLICY
# =========================================================

def evaluate_policy(
    config: GameConfig,
    policy: PolicyFunction,
    seed: int,
) -> dict:
    """
    Evaluate one policy over several independent episodes.

    Each episode starts from config.initial_state, or from
    the all-healthy state when initial_state is None.

    Returns average return, cumulative loss, defense cost,
    attack statistics, and infection/recovery counts.
    """

    env = SecurityGame(config, seed=seed)

    episode_returns = []
    episode_rewards = []
    episode_losses = []
    episode_defense_costs = []

    attacks_per_target = np.zeros(
        config.K,
        dtype=np.int64,
    )

    successful_attacks = 0
    eligible_attack_attempts = 0
    total_new_infections = 0
    total_recoveries = 0

    for episode in range(config.eval_episodes):

        state = env.reset()

        discounted_return = 0.0
        total_reward = 0.0
        cumulative_loss = 0.0
        defense_cost = 0.0

        for step in range(config.eval_horizon):

            # -----------------------------------------
            # Select a defense allocation
            # -----------------------------------------

            defense = tuple(policy(state))

            if not env.is_valid_action(defense):
                raise ValueError(
                    f"Policy returned an invalid action: {defense}"
                )

            # -----------------------------------------
            # Simulate one turn
            # -----------------------------------------

            next_state, reward, info = env.step(defense)

            discounted_return += (
                config.gamma ** step
            ) * reward

            total_reward += reward

            cumulative_loss += info["infected_value"]
            defense_cost += info["defense_cost"]

            # -----------------------------------------
            # Attack statistics
            # -----------------------------------------

            target = info["attack_target"]

            attacks_per_target[target] += 1

            # Only an attack on a healthy target can
            # produce a new infection.
            if info["state_before"][target] == 0:

                eligible_attack_attempts += 1

                if info["attack_roll_success"]:
                    successful_attacks += 1

            total_new_infections += int(
                info["new_infection"]
            )

            total_recoveries += len(
                info["recovered_targets"]
            )

            state = next_state

        episode_returns.append(discounted_return)
        episode_rewards.append(total_reward)
        episode_losses.append(cumulative_loss)
        episode_defense_costs.append(defense_cost)

    total_attack_attempts = (
        config.eval_episodes * config.eval_horizon
    )

    attack_success_rate = (
        successful_attacks / eligible_attack_attempts
        if eligible_attack_attempts > 0
        else float("nan")
    )

    metrics = {
        "mean_discounted_return": float(
            np.mean(episode_returns)
        ),

        "std_discounted_return": float(
            np.std(episode_returns)
        ),

        "mean_episode_reward": float(
            np.mean(episode_rewards)
        ),

        "mean_cumulative_loss": float(
            np.mean(episode_losses)
        ),

        "mean_defense_cost": float(
            np.mean(episode_defense_costs)
        ),

        "attack_success_rate": float(
            attack_success_rate
        ),

        "new_infections_per_episode": float(
            total_new_infections / config.eval_episodes
        ),

        "recoveries_per_episode": float(
            total_recoveries / config.eval_episodes
        ),
    }

    # Fraction of all attack attempts directed at each target.
    for i in range(config.K):
        metrics[f"attack_frequency_{i}"] = float(
            attacks_per_target[i] / total_attack_attempts
        )

    return metrics


# =========================================================
# 3. AGGREGATE RESULTS ACROSS SEEDS
# =========================================================

def aggregate_results(results: list[dict]) -> list[dict]:
    """
    Aggregate per-seed evaluation results by policy.

    The resulting mean and standard deviation are calculated
    across seeds, not across individual time steps.
    """

    if not results:
        return []

    policy_names = list(dict.fromkeys(
        row["policy"] for row in results
    ))

    metadata = {
        "policy",
        "training_seed",
        "evaluation_seed",
    }

    metric_names = [
        key
        for key in results[0]
        if key not in metadata
    ]

    summaries = []

    for policy_name in policy_names:

        rows = [
            row for row in results
            if row["policy"] == policy_name
        ]

        summary = {
            "policy": policy_name,
            "num_seeds": len(rows),
        }

        for metric in metric_names:

            values = np.asarray(
                [row[metric] for row in rows],
                dtype=float,
            )

            valid = values[~np.isnan(values)]

            if len(valid) == 0:
                mean = float("nan")
                std = float("nan")
            else:
                mean = float(np.mean(valid))
                std = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0

            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std_across_seeds"] = std

        summaries.append(summary)

    return summaries


# =========================================================
# 4. PLOTTING
# =========================================================

def plot_learning_curves(
    histories: list[dict],
    output_dir: Path,
) -> None:
    """Plot mean training return across Q-learning seeds."""

    if plt is None:
        warnings.warn(
            "Matplotlib is unavailable; skipping plots."
        )
        return

    if not histories:
        return

    curves = np.asarray([
        history["episode_return"]
        for history in histories
    ])

    mean_curve = np.mean(curves, axis=0)

    window = min(20, len(mean_curve))

    moving_average = np.convolve(
        mean_curve,
        np.ones(window) / window,
        mode="valid",
    )

    episodes = np.arange(
        window,
        len(mean_curve) + 1,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(episodes, moving_average)
    plt.xlabel("Training episode")
    plt.ylabel("Discounted episode return")
    plt.title("Q-learning training curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "learning_curve.png", dpi=160)
    plt.close()


def plot_policy_comparison(
    summary: list[dict],
    output_dir: Path,
) -> None:
    """Create separate plots for the main evaluation metrics."""

    if plt is None or not summary:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    plots = [
        (
            "mean_discounted_return",
            "Mean discounted return",
            "policy_returns.png",
        ),
        (
            "mean_cumulative_loss",
            "Mean cumulative infected-target loss",
            "policy_infected_loss.png",
        ),
        (
            "mean_defense_cost",
            "Mean cumulative defense cost",
            "policy_defense_cost.png",
        ),
        (
            "policy_loss",
            "Empirical performance loss relative to DP",
            "policy_performance_loss.png",
        ),
    ]

    for metric, ylabel, filename in plots:

        means = np.asarray([
            row[f"{metric}_mean"]
            for row in summary
        ], dtype=float)

        stds = np.asarray([
            row[f"{metric}_std_across_seeds"]
            for row in summary
        ], dtype=float)

        # Avoid passing NaN error-bar sizes to Matplotlib.
        stds = np.nan_to_num(stds, nan=0.0)

        labels = [
            row["policy"].replace("_", " ")
            for row in summary
        ]

        plt.figure(figsize=(9, 5))

        plt.bar(
            labels,
            means,
            yerr=stds,
            capsize=4,
        )

        plt.ylabel(ylabel)
        plt.xticks(rotation=15, ha="right")
        plt.title(ylabel)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=160)
        plt.close()


# =========================================================
# 5. RUN THE COMPLETE EXPERIMENT
# =========================================================

def run_experiments(
    config: GameConfig,
    output_dir: Path = OUTPUT_DIR,
    make_plots: bool = True,
    save_checkpoints: bool = True,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Run:
        1. Exact value iteration.
        2. Q-learning for every configured training seed.
        3. Evaluation of four policies.
        4. CSV output, checkpoints, and plots.

    Returns:
        results:
            Per-policy, per-seed evaluation records.

        summary:
            Aggregated results across seeds.

        training_histories:
            Q-learning training histories.
    """

    if not config.seeds:
        raise ValueError("At least one seed is required.")

    if config.eval_episodes <= 0 or config.eval_horizon <= 0:
        raise ValueError(
            "Evaluation episodes and horizon must be positive."
        )

    output_dir = Path(output_dir)

    results_dir = output_dir / "results"
    checkpoints_dir = output_dir / "checkpoints"
    figures_dir = output_dir / "figures"

    results_dir.mkdir(parents=True, exist_ok=True)

    if save_checkpoints:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

    save_config(config, output_dir)

    # -----------------------------------------------------
    # A. Create the state/action spaces
    # -----------------------------------------------------

    base_env = SecurityGame(
        config,
        seed=config.seeds[0],
    )

    states = base_env.states
    actions = base_env.actions

    state_to_index = {
        state: i
        for i, state in enumerate(states)
    }

    print("\nConfiguration")
    print("-------------")
    print(f"Targets:            {config.K}")
    print(f"Defense budget:     {config.B}")
    print(f"States:             {len(states)}")
    print(f"Defense actions:    {len(actions)}")
    print(f"Training seeds:     {len(config.seeds)}")

    # -----------------------------------------------------
    # B. Solve the exact MDP with value iteration
    # -----------------------------------------------------

    print("\nComputing the exact DP solution...")

    V_star, Q_dp, optimal_policy = solve_value_iteration(
        config,
        states,
        actions,
        base_env,
    )

    initial_state = (
        tuple(config.initial_state)
        if config.initial_state is not None
        else (0,) * config.K
    )

    initial_state_idx = state_to_index[initial_state]

    print(
        "Optimal infinite-horizon value at initial state:",
        float(V_star[initial_state_idx]),
    )

    if save_checkpoints:
        np.savez(
            checkpoints_dir / "dp_solution.npz",
            V=V_star,
            Q_dp=Q_dp,
            optimal_policy=optimal_policy,
            states=np.asarray(states, dtype=int),
            actions=np.asarray(actions, dtype=int),
        )

    # -----------------------------------------------------
    # C. Define the two heuristic allocations
    # -----------------------------------------------------

    high_value_action = highest_value_policy(
        config,
        actions,
    )

    even_action = uniform_policy(config)

    # -----------------------------------------------------
    # D. Train Q-learning and evaluate all policies
    # -----------------------------------------------------

    all_results = []
    training_histories = []

    for seed_idx, training_seed in enumerate(config.seeds):

        print(
            f"\nSeed {seed_idx + 1}/{len(config.seeds)}: "
            f"{training_seed}"
        )

        train_env = SecurityGame(
            config,
            seed=training_seed,
        )

        # Train the Q-learning defender.
        Q, history = train_q_learning(
            config=config,
            env=train_env,
            states=states,
            actions=actions,
            state_to_index=state_to_index,
            seed=training_seed,
        )

        training_histories.append(history)

        if save_checkpoints:
            np.savez(
                checkpoints_dir / f"q_seed_{training_seed}.npz",
                Q=Q,
                states=np.asarray(states, dtype=int),
                actions=np.asarray(actions, dtype=int),
            )

        # Policies all receive a state and return a defense tuple.
        policies: dict[str, PolicyFunction] = {
            "dynamic_programming": (
                lambda state: dp_policy(
                    state,
                    optimal_policy,
                    state_to_index,
                    actions,
                )
            ),

            "q_learning": (
                lambda state, q=Q: q_learning_policy(
                    state,
                    q,
                    state_to_index,
                    actions,
                )
            ),

            "highest_value": (
                lambda state, action=high_value_action: action
            ),

            "uniform": (
                lambda state, action=even_action: action
            ),
        }

        # Use separate evaluation randomness from training.
        # The same evaluation seed is used for all policies
        # within this comparison.
        evaluation_seed = int(training_seed) + 100000

        seed_metrics = {}

        for policy_name, policy in policies.items():

            print(f"  Evaluating {policy_name}...")

            seed_metrics[policy_name] = evaluate_policy(
                config=config,
                policy=policy,
                seed=evaluation_seed,
            )

        # Use finite-horizon simulated DP return as the empirical
        # reference for this seed's policy comparison.
        dp_return = seed_metrics[
            "dynamic_programming"
        ]["mean_discounted_return"]

        denominator = abs(dp_return)

        for policy_name, metrics in seed_metrics.items():

            if policy_name == "dynamic_programming":
                performance_loss = 0.0

            elif denominator > 1e-12:
                performance_loss = (
                    dp_return
                    - metrics["mean_discounted_return"]
                ) / denominator

            else:
                performance_loss = float("nan")

            row = {
                "policy": policy_name,
                "training_seed": int(training_seed),
                "evaluation_seed": int(evaluation_seed),
                **metrics,
                "policy_loss": float(performance_loss),
            }

            all_results.append(row)

    # -----------------------------------------------------
    # E. Aggregate and save results
    # -----------------------------------------------------

    summary = aggregate_results(all_results)

    save_csv(
        results_dir / "results_by_seed.csv",
        all_results,
    )

    save_csv(
        results_dir / "summary.csv",
        summary,
    )

    # -----------------------------------------------------
    # F. Save training and evaluation plots
    # -----------------------------------------------------

    if make_plots:

        plot_learning_curves(
            training_histories,
            figures_dir,
        )

        plot_policy_comparison(
            summary,
            figures_dir,
        )

    # -----------------------------------------------------
    # G. Print a compact comparison table
    # -----------------------------------------------------

    print("\nFinal results")
    print("=" * 83)

    print(
        f"{'Policy':<24}"
        f"{'Mean return':>17}"
        f"{'Infected loss':>19}"
        f"{'Defense cost':>16}"
    )

    print("-" * 83)

    for row in summary:

        print(
            f"{row['policy']:<24}"
            f"{row['mean_discounted_return_mean']:>17.4f}"
            f"{row['mean_cumulative_loss_mean']:>19.4f}"
            f"{row['mean_defense_cost_mean']:>16.4f}"
        )

    print(f"\nResults saved to: {results_dir}")
    print(f"Figures saved to: {figures_dir}")

    return all_results, summary, training_histories


# =========================================================
# 6. OPTIONAL PARAMETER SWEEPS
# =========================================================

def run_parameter_sweep(
    config: GameConfig,
    parameter_grid: dict,
    output_dir: Path = OUTPUT_DIR,
) -> list[dict]:
    """
    Run complete experiments for each parameter setting.

    Scalar example:
        {"B": (1, 2, 3)}

    For K sweeps, supply complete override dictionaries
    containing K and matching target parameter tuples.
    """

    sweep_output_dir = Path(output_dir) / "parameter_sweeps"

    sweep_results = []

    for parameter, settings in parameter_grid.items():

        for setting_idx, setting in enumerate(settings):

            if isinstance(setting, dict):
                overrides = setting

                displayed_value = overrides.get(
                    parameter,
                    f"configuration_{setting_idx}",
                )

            else:
                overrides = {parameter: setting}
                displayed_value = setting

            variant_config = replace(
                config,
                **overrides,
            )

            label = (
                f"{parameter}_{setting_idx:02d}_"
                f"{str(displayed_value).replace('.', 'p')}"
            )

            variant_output_dir = (
                sweep_output_dir / parameter / label
            )

            print(
                f"\nParameter sweep: {parameter} = "
                f"{displayed_value}"
            )

            _, variant_summary, _ = run_experiments(
                config=variant_config,
                output_dir=variant_output_dir,
                make_plots=False,
                save_checkpoints=False,
            )

            if (
                isinstance(displayed_value, (int, float))
                and not isinstance(displayed_value, bool)
            ):
                numeric_value = float(displayed_value)
            else:
                numeric_value = float("nan")

            for row in variant_summary:

                sweep_results.append({
                    "sweep_parameter": parameter,
                    "sweep_value": str(displayed_value),
                    "sweep_value_numeric": numeric_value,
                    "policy": row["policy"],

                    "mean_discounted_return":
                        row["mean_discounted_return_mean"],

                    "mean_cumulative_loss":
                        row["mean_cumulative_loss_mean"],

                    "mean_defense_cost":
                        row["mean_defense_cost_mean"],

                    "policy_loss":
                        row["policy_loss_mean"],
                })

    save_csv(
        sweep_output_dir / "parameter_sweep_summary.csv",
        sweep_results,
    )

    plot_parameter_sweep(
        sweep_results,
        sweep_output_dir,
    )

    return sweep_results


def plot_parameter_sweep(
    sweep_results: list[dict],
    output_dir: Path,
) -> None:
    """Plot discounted return against numeric sweep parameters."""

    if plt is None or not sweep_results:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    parameters = list(dict.fromkeys(
        row["sweep_parameter"]
        for row in sweep_results
    ))

    for parameter in parameters:

        rows = [
            row for row in sweep_results
            if row["sweep_parameter"] == parameter
            and np.isfinite(row["sweep_value_numeric"])
        ]

        if not rows:
            continue

        plt.figure(figsize=(9, 5))

        policies = list(dict.fromkeys(
            row["policy"] for row in rows
        ))

        for policy in policies:

            policy_rows = sorted(
                (
                    row for row in rows
                    if row["policy"] == policy
                ),
                key=lambda row: row["sweep_value_numeric"],
            )

            x = [
                row["sweep_value_numeric"]
                for row in policy_rows
            ]

            y = [
                row["mean_discounted_return"]
                for row in policy_rows
            ]

            plt.plot(
                x,
                y,
                marker="o",
                label=policy.replace("_", " "),
            )

        plt.xlabel(parameter)
        plt.ylabel("Mean discounted return")
        plt.title(f"Parameter sweep: {parameter}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            output_dir / f"{parameter}_return.png",
            dpi=160,
        )

        plt.close()


# =========================================================
# 7. ENTRY POINT
# =========================================================

def main() -> None:

    # Run the primary experiment.
    run_experiments(CONFIG)

    # Only run sweeps when explicitly enabled in config.py.
    if RUN_PARAMETER_SWEEPS and PARAMETER_SWEEP:

        run_parameter_sweep(
            CONFIG,
            PARAMETER_SWEEP,
        )


if __name__ == "__main__":
    main()