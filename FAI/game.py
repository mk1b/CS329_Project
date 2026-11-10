# game.py

from itertools import product
import numpy as np

from config import GameConfig

State = tuple[int, ...]
DefenseAction = tuple[int, ...]


def enumerate_states(K: int) -> list[State]:
    """Generate all 2^K possible states."""
    return list(product((0, 1), repeat=K))


def generate_actions(K: int, B: int) -> list[DefenseAction]:
    """Generate all integer allocations with total budget <= B."""
    actions = []

    def generate(i, remaining, current):
        if i == K:
            actions.append(tuple(current))
            return

        for amount in range(remaining + 1):
            current.append(amount)
            generate(i + 1, remaining - amount, current)
            current.pop()

    generate(0, B, [])
    return actions


class SecurityGame:
    def __init__(self, config: GameConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.default_rng(seed)

        self.K = config.K
        self.B = config.B

        self.states = enumerate_states(self.K)
        self.actions = generate_actions(self.K, self.B)

        self.state = tuple(
            config.initial_state
            if config.initial_state is not None
            else (0,) * self.K
        )

        self._validate_config()
        self._validate_state(self.state)

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    def _validate_config(self):
        c = self.config

        if self.K < 1 or self.B < 0:
            raise ValueError("K must be positive and B nonnegative.")

        if len(c.values) != self.K:
            raise ValueError("values must have length K.")

        if len(c.attack_probs) != self.K:
            raise ValueError("attack_probs must have length K.")

        if len(c.attack_costs) != self.K:
            raise ValueError("attack_costs must have length K.")

        if len(c.recovery_probs) != self.K:
            raise ValueError("recovery_probs must have length K.")

        if not 0 < c.eta < 1:
            raise ValueError("eta must lie in (0, 1).")

        if not 0 < c.gamma < 1:
            raise ValueError("gamma must lie in (0, 1).")

        if any(v <= 0 for v in c.values):
            raise ValueError("Target values must be positive.")

        if any(cost <= 0 for cost in c.attack_costs):
            raise ValueError("Attack costs must be positive.")

        if any(not 0 <= p <= 1 for p in c.attack_probs):
            raise ValueError("Attack probabilities must lie in [0, 1].")

        if any(not 0 <= r <= 1 for r in c.recovery_probs):
            raise ValueError("Recovery probabilities must lie in [0, 1].")

        if c.defense_cost_lambda < 0:
            raise ValueError("Defense cost must be nonnegative.")

    def _validate_state(self, state):
        if (
            len(state) != self.K
            or any(x not in (0, 1) for x in state)
        ):
            raise ValueError("Invalid state.")

    def is_valid_action(self, defense: DefenseAction) -> bool:
        return (
            len(defense) == self.K
            and all(
                isinstance(d, (int, np.integer)) and 0 <= d <= self.B
                for d in defense
            )
            and sum(defense) <= self.B
        )

    # --------------------------------------------------
    # Game model
    # --------------------------------------------------

    def infection_probability(self, i: int, defense: DefenseAction) -> float:
        """q_i(d_i) = p_i * (1 - eta)^d_i"""
        return (
            self.config.attack_probs[i]
            * (1 - self.config.eta) ** defense[i]
        )

    def attacker_utility(
        self,
        state: State,
        defense: DefenseAction,
        i: int,
    ) -> float:
        """Utility of attacking target i."""
        q = self.infection_probability(i, defense)

        return (
            self.config.values[i]
            * (1 - state[i])
            * q
            - self.config.attack_costs[i]
        )

    def choose_attack_target(
        self,
        state: State,
        defense: DefenseAction,
    ) -> int:
        """Strategic attacker chooses maximum utility.
        Ties are broken by smallest target index.
        """
        utilities = [
            self.attacker_utility(state, defense, i)
            for i in range(self.K)
        ]

        return max(range(self.K), key=lambda i: utilities[i])

    def defender_loss(self, state: State) -> float:
        """Weighted value of infected targets, following the report's equation."""
        return sum(
            self.config.values[i] * state[i]
            for i in range(self.K)
        )

    def defense_cost(self, defense: DefenseAction) -> float:
        return (
            self.config.defense_cost_lambda
            * sum(defense)
        )

    def defender_reward(
        self,
        state: State,
        defense: DefenseAction,
    ) -> float:
        """R(x,d) = -L(x) - C_D(d)."""
        return -self.defender_loss(state) - self.defense_cost(defense)

    # --------------------------------------------------
    # Exact transition distribution: used by DP
    # --------------------------------------------------

    def transition_distribution(
        self,
        state: State,
        defense: DefenseAction,
    ) -> dict[State, float]:
        """Return {next_state: probability} for every possible successor."""

        self._validate_state(state)

        if not self.is_valid_action(defense):
            raise ValueError("Invalid defense allocation.")

        attack_target = self.choose_attack_target(state, defense)
        distribution = {}

        for next_state in self.states:
            probability = 1.0

            for i in range(self.K):
                current = state[i]
                nxt = next_state[i]

                if current == 1:
                    # Previously infected target may recover.
                    rho = self.config.recovery_probs[i]

                    if nxt == 0:
                        p_i = rho
                    else:
                        p_i = 1 - rho

                elif i == attack_target:
                    # Selected target was healthy.
                    q = self.infection_probability(i, defense)

                    if nxt == 1:
                        p_i = q
                    else:
                        p_i = 1 - q

                else:
                    # Unattacked healthy target stays healthy.
                    p_i = 1.0 if nxt == 0 else 0.0

                probability *= p_i

                if probability == 0:
                    break

            if probability > 0:
                distribution[next_state] = probability

        return distribution

    # --------------------------------------------------
    # Reset and simulation: used by Q-learning
    # --------------------------------------------------

    def reset(self, initial_state: State | None = None) -> State:
        """Reset the game state without resetting the random generator."""

        if initial_state is None:
            initial_state = (
                self.config.initial_state
                if self.config.initial_state is not None
                else (0,) * self.K
            )

        initial_state = tuple(initial_state)
        self._validate_state(initial_state)

        self.state = initial_state
        return self.state

    def step(
        self,
        defense: DefenseAction,
    ) -> tuple[State, float, dict]:
        """Simulate one turn and return (next_state, reward, info)."""

        defense = tuple(defense)

        if not self.is_valid_action(defense):
            raise ValueError("Invalid defense allocation.")

        # Current state is the basis for reward and transitions.
        state_before = self.state

        reward = self.defender_reward(state_before, defense)

        attack_target = self.choose_attack_target(
            state_before, defense
        )

        attack_probability = self.infection_probability(
            attack_target, defense
        )

        # Sample the attack outcome.
        attack_roll_success = (
            self.rng.random() < attack_probability
        )

        next_state = list(state_before)
        new_infection = False

        # Attack can infect only a previously healthy target.
        if state_before[attack_target] == 0 and attack_roll_success:
            next_state[attack_target] = 1
            new_infection = True

        # Recover targets infected at the BEGINNING of this turn.
        recovered_targets = []

        for i in range(self.K):
            if state_before[i] == 1:
                if self.rng.random() < self.config.recovery_probs[i]:
                    next_state[i] = 0
                    recovered_targets.append(i)

        self.state = tuple(next_state)

        info = {
            "state_before": state_before,
            "defense": defense,
            "attack_target": attack_target,
            "attack_probability": attack_probability,
            "attack_roll_success": attack_roll_success,
            "new_infection": new_infection,
            "recovered_targets": recovered_targets,
            "state_after": self.state,
            "infected_value": self.defender_loss(state_before),
            "defense_cost": self.defense_cost(defense),
        }

        return self.state, reward, info