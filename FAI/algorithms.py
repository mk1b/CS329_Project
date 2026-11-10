# algorithms.py

import warnings
import numpy as np

from config import GameConfig
from game import SecurityGame, State, DefenseAction


# =========================================================
# 1. EXACT DYNAMIC PROGRAMMING: VALUE ITERATION
# =========================================================

def solve_value_iteration(
    config: GameConfig,
    states: list[State],
    actions: list[DefenseAction],
    env: SecurityGame | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        V:
            Optimal state values, shape (num_states,).

        Q_dp:
            Bellman action values, shape (num_states, num_actions).

        optimal_policy:
            Best action index for every state, shape (num_states,).
    """

    if env is None:
        env = SecurityGame(config, seed=42)

    num_states = len(states)
    num_actions = len(actions)

    if num_states == 0 or num_actions == 0:
        raise ValueError("States and actions must be nonempty.")

    state_to_index = {
        state: i for i, state in enumerate(states)
    }

    # R[s, a] = immediate reward for state s and action a.
    R = np.zeros((num_states, num_actions), dtype=float)

    # P[s][a] contains (next_state_index, probability) pairs.
    # Precompute these once because the model is stationary.
    P = [
        [[] for _ in range(num_actions)]
        for _ in range(num_states)
    ]

    for s_idx, state in enumerate(states):
        for a_idx, action in enumerate(actions):

            R[s_idx, a_idx] = env.defender_reward(
                state, action
            )

            distribution = env.transition_distribution(
                state, action
            )

            total_probability = sum(distribution.values())

            if not np.isclose(total_probability, 1.0):
                raise ValueError(
                    f"Transition probabilities sum to "
                    f"{total_probability}, not 1."
                )

            P[s_idx][a_idx] = [
                (state_to_index[next_state], probability)
                for next_state, probability in distribution.items()
            ]

    gamma = config.gamma
    tolerance = config.dp_tolerance
    max_iterations = config.dp_max_iterations

    # Initial value estimates.
    V = np.zeros(num_states, dtype=float)

    converged = False

    for iteration in range(max_iterations):

        V_new = np.zeros(num_states, dtype=float)

        for s_idx in range(num_states):

            best_value = -np.inf

            for a_idx in range(num_actions):

                expected_future_value = sum(
                    probability * V[next_idx]
                    for next_idx, probability in P[s_idx][a_idx]
                )

                q_value = (
                    R[s_idx, a_idx]
                    + gamma * expected_future_value
                )

                if q_value > best_value:
                    best_value = q_value

            V_new[s_idx] = best_value

        delta = np.max(np.abs(V_new - V))
        V = V_new

        if delta < tolerance:
            converged = True
            break

    if not converged:
        warnings.warn(
            "Value iteration reached dp_max_iterations "
            "without meeting dp_tolerance."
        )

    # Recompute Q using the final value estimates.
    Q_dp = np.zeros(
        (num_states, num_actions),
        dtype=float,
    )

    for s_idx in range(num_states):
        for a_idx in range(num_actions):

            expected_future_value = sum(
                probability * V[next_idx]
                for next_idx, probability in P[s_idx][a_idx]
            )

            Q_dp[s_idx, a_idx] = (
                R[s_idx, a_idx]
                + gamma * expected_future_value
            )

    # np.argmax selects the first action in case of ties.
    optimal_policy = np.argmax(Q_dp, axis=1)

    return V, Q_dp, optimal_policy


# =========================================================
# 2. EPSILON-GREEDY EXPLORATION
# =========================================================

def epsilon_at_step(
    global_step: int,
    config: GameConfig,
) -> float:
    """Linearly decay epsilon from epsilon_start to epsilon_min."""

    start = config.epsilon_start
    minimum = config.epsilon_min
    decay_steps = config.epsilon_decay_steps

    if not 0 <= minimum <= start <= 1:
        raise ValueError(
            "Require 0 <= epsilon_min <= epsilon_start <= 1."
        )

    if decay_steps <= 0:
        raise ValueError(
            "epsilon_decay_steps must be positive."
        )

    fraction = min(global_step / decay_steps, 1.0)

    return start - fraction * (start - minimum)


def select_action(
    state: State,
    Q: np.ndarray,
    state_to_index: dict[State, int],
    actions: list[DefenseAction],
    epsilon: float,
    rng: np.random.Generator,
) -> tuple[int, DefenseAction]:
    """
    Select an action using epsilon-greedy exploration.

    Returns:
        action_idx: Index into actions and Q[state_idx].
        defense: Actual defense allocation tuple.
    """

    state_idx = state_to_index[state]

    if rng.random() < epsilon:
        # Explore: select any legal action uniformly.
        action_idx = int(rng.integers(len(actions)))
    else:
        # Exploit: choose the highest estimated Q-value.
        action_idx = int(np.argmax(Q[state_idx]))

    return action_idx, actions[action_idx]


# =========================================================
# 3. TABULAR Q-LEARNING
# =========================================================

def train_q_learning(
    config: GameConfig,
    env: SecurityGame,
    states: list[State],
    actions: list[DefenseAction],
    state_to_index: dict[State, int] | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """
    Train a tabular Q-learning defender.

    Returns:
        Q:
            Learned action-value table.

        training_history:
            Episode-level training metrics.
    """

    if not states or not actions:
        raise ValueError("States and actions must be nonempty.")

    if not 0 < config.alpha <= 1:
        raise ValueError("alpha must lie in (0, 1].")

    if config.num_episodes <= 0:
        raise ValueError("num_episodes must be positive.")

    if config.steps_per_episode <= 0:
        raise ValueError("steps_per_episode must be positive.")

    if state_to_index is None:
        state_to_index = {
            state: i for i, state in enumerate(states)
        }

    num_states = len(states)
    num_actions = len(actions)

    # Independent random generators for exploration and environment.
    rng = np.random.default_rng(seed + 1)
    env.rng = np.random.default_rng(seed)

    # Q[s, a] estimates the long-term return from (s, a).
    Q = np.zeros(
        (num_states, num_actions),
        dtype=float,
    )

    # Counts the visits to each state-action pair.
    visits = np.zeros(
        (num_states, num_actions),
        dtype=np.int64,
    )

    gamma = config.gamma
    alpha = config.alpha

    training_history = {
        "episode": [],
        "episode_return": [],
        "episode_reward": [],
        "cumulative_loss": [],
        "defense_cost": [],
        "mean_epsilon": [],
        "mean_abs_td_error": [],
    }

    global_step = 0

    for episode in range(config.num_episodes):

        state = env.reset()

        discounted_return = 0.0
        total_reward = 0.0
        cumulative_loss = 0.0
        total_defense_cost = 0.0

        epsilon_sum = 0.0
        td_error_sum = 0.0

        for step in range(config.steps_per_episode):

            # ---------------------------------------------
            # A. Select defense allocation
            # ---------------------------------------------

            epsilon = epsilon_at_step(
                global_step, config
            )

            action_idx, defense = select_action(
                state=state,
                Q=Q,
                state_to_index=state_to_index,
                actions=actions,
                epsilon=epsilon,
                rng=rng,
            )

            state_idx = state_to_index[state]

            # ---------------------------------------------
            # B. Simulate one game turn
            # ---------------------------------------------

            next_state, reward, info = env.step(defense)

            next_state_idx = state_to_index[next_state]

            # ---------------------------------------------
            # C. Update visit counts
            # ---------------------------------------------

            visits[state_idx, action_idx] += 1

            # ---------------------------------------------
            # D. Q-learning update
            # ---------------------------------------------

            td_target = (
                reward
                + gamma * np.max(Q[next_state_idx])
            )

            td_error = (
                td_target - Q[state_idx, action_idx]
            )

            Q[state_idx, action_idx] += (
                alpha * td_error
            )

            # ---------------------------------------------
            # E. Record episode metrics
            # ---------------------------------------------

            discounted_return += (
                gamma ** step
            ) * reward

            total_reward += reward

            cumulative_loss += info["infected_value"]
            total_defense_cost += info["defense_cost"]

            epsilon_sum += epsilon
            td_error_sum += abs(td_error)

            state = next_state
            global_step += 1

        # Store episode-level aggregates.
        training_history["episode"].append(episode + 1)

        training_history["episode_return"].append(
            discounted_return
        )

        training_history["episode_reward"].append(
            total_reward
        )

        training_history["cumulative_loss"].append(
            cumulative_loss
        )

        training_history["defense_cost"].append(
            total_defense_cost
        )

        training_history["mean_epsilon"].append(
            epsilon_sum / config.steps_per_episode
        )

        training_history["mean_abs_td_error"].append(
            td_error_sum / config.steps_per_episode
        )

    training_history["visits"] = visits
    training_history["seed"] = seed

    return Q, training_history


# =========================================================
# 4. EXTRACT LEARNED POLICY
# =========================================================

def extract_q_policy(Q: np.ndarray) -> np.ndarray:
    """
    Return the best action index for every state.
    Shape: (num_states,).
    """
    if Q.ndim != 2 or Q.shape[1] == 0:
        raise ValueError(
            "Q must be a 2D array with at least one action."
        )

    return np.argmax(Q, axis=1)


def q_learning_policy(
    state: State,
    Q: np.ndarray,
    state_to_index: dict[State, int],
    actions: list[DefenseAction],
) -> DefenseAction:
    """Return the greedy defense action learned by Q-learning."""

    state_idx = state_to_index[state]
    action_idx = int(np.argmax(Q[state_idx]))

    return actions[action_idx]


# =========================================================
# 5. DYNAMIC-PROGRAMMING POLICY
# =========================================================

def dp_policy(
    state: State,
    optimal_policy: np.ndarray,
    state_to_index: dict[State, int],
    actions: list[DefenseAction],
) -> DefenseAction:
    """Return the optimal defense allocation for a state."""

    state_idx = state_to_index[state]
    action_idx = int(optimal_policy[state_idx])

    return actions[action_idx]


# =========================================================
# 6. HIGHEST-VALUE HEURISTIC
# =========================================================

def highest_value_policy(
    config: GameConfig,
    actions: list[DefenseAction] | None = None,
) -> DefenseAction:
    """
    Allocate all B units in rounds ordered by descending target value.

    Ties in target value are broken by target index.
    """

    order = sorted(
        range(config.K),
        key=lambda i: (-config.values[i], i),
    )

    defense = [0] * config.K

    for unit in range(config.B):
        target = order[unit % config.K]
        defense[target] += 1

    return tuple(defense)


# =========================================================
# 7. UNIFORM HEURISTIC
# =========================================================

def uniform_policy(
    config: GameConfig,
) -> DefenseAction:
    """
    Distribute B units as evenly as possible across K targets.
    """

    base = config.B // config.K
    remainder = config.B % config.K

    defense = tuple(
        base + (1 if i < remainder else 0)
        for i in range(config.K)
    )

    return defense