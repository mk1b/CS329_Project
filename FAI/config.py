# config.py

from dataclasses import dataclass
from pathlib import Path


@dataclass
class GameConfig:
    # =============================================
    # Game parameters
    # =============================================

    K: int = 10
    B: int = 5

    eta: float = 0.20
    defense_cost_lambda: float = 0.75
    gamma: float = 0.95

    # Per-target parameters: all must have length K.
    values: tuple[float, ...] = (
        10.0, 8.0, 6.0, 4.0, 3.0,
        2.5, 2.0, 1.5, 1.0, 0.5,
    )

    attack_probs: tuple[float, ...] = (
        0.60, 0.55, 0.45, 0.35, 0.30,
        0.25, 0.20, 0.15, 0.10, 0.05,
    )

    attack_costs: tuple[float, ...] = (
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    )

    recovery_probs: tuple[float, ...] = (
        0.20, 0.15, 0.25, 0.30, 0.20,
        0.25, 0.30, 0.35, 0.40, 0.45,
    )

    # None means all targets start healthy.
    initial_state: tuple[int, ...] | None = None

    # =============================================
    # Q-learning parameters
    # =============================================

    num_episodes: int = 500
    steps_per_episode: int = 100

    alpha: float = 0.10

    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay_steps: int = 20000

    # =============================================
    # Dynamic programming parameters
    # =============================================

    dp_tolerance: float = 1e-8
    dp_max_iterations: int = 10000

    # =============================================
    # Evaluation parameters
    # =============================================

    eval_episodes: int = 100
    eval_horizon: int = 100

    seeds: tuple[int, ...] = (
        42,
        169,
        6969,
        4267,
        6742,
    )


# =============================================
# Main configuration
# =============================================

CONFIG = GameConfig()


# =============================================
# Output directories
# =============================================

OUTPUT_DIR = Path("outputs")

RESULTS_DIR = OUTPUT_DIR / "results"
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
FIGURES_DIR = OUTPUT_DIR / "figures"


# =============================================
# Optional parameter sweeps
# =============================================

RUN_PARAMETER_SWEEPS = False

PARAMETER_SWEEP = {
    "B": (1, 2, 3),
    "eta": (0.10, 0.20, 0.30),
    "defense_cost_lambda": (0.25, 0.75, 1.50),
}