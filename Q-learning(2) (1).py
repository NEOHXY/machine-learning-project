"""
Dynamic Grid World (6x6) - Plan B Temporary Blocks
- No weather
- Temporary blocks appear/disappear during an episode
- Q-learning with partial observability (blocks NOT in state)

Outputs:
1) Training progress
2) Final visualization: * = path, ! = temporary block history, # = wall
3) COMPLETE Q-table: for every cell (row,col), show Q for UP/RIGHT/DOWN/LEFT
4) Policy map: best action per cell (arrows)

"""

from __future__ import annotations
import random
from typing import Dict, List, Tuple, Optional

# -----------------------------
# 1) Hard-coded 6x6 base map
# -----------------------------
# Legend:
# 'S' = Start, 'G' = Goal, '#' = Wall, '.' = Empty
GRID: List[List[str]] = [
    ['S', '.', '.', '#', '.', '.'],
    ['.', '#', '.', '#', '.', '.'],
    ['.', '#', '.', '.', '.', '#'],
    ['.', '.', '#', '#', '.', '.'],
    ['#', '.', '.', '.', '#', '.'],
    ['.', '.', '#', '.', '.', 'G'],
]
ROWS, COLS = 6, 6

# Rewards (Plan 3: stronger goal reward)
STEP_REWARD = -0.1
INVALID_REWARD = -1.0
GOAL_REWARD = 5.0  # <- Plan 3: increased goal reward

# Dynamic blocks settings (Plan B)
P_SPAWN = 0.05
BLOCK_DURATION = 12
MAX_TEMP_BLOCKS = 5

# Q-learning hyperparameters
ALPHA = 0.2
GAMMA = 0.95
EPSILON = 1.0
EPS_MIN = 0.05
EPS_DECAY = 0.995

EPISODES = 5000
MAX_STEPS = 120

# Actions: UP, RIGHT, DOWN, LEFT
ACTIONS: Dict[int, Tuple[int, int]] = {
    0: (-1, 0),  # UP
    1: (0, 1),   # RIGHT
    2: (1, 0),   # DOWN
    3: (0, -1),  # LEFT
}
ACTION_NAMES = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
ARROWS = {0: "↑", 1: "→", 2: "↓", 3: "←"}


# -----------------------------
# 2) Utilities
# -----------------------------
def find_cell(target: str) -> Tuple[int, int]:
    for r in range(ROWS):
        for c in range(COLS):
            if GRID[r][c] == target:
                return (r, c)
    raise ValueError(f"Cell '{target}' not found in grid.")


START = find_cell('S')
GOAL = find_cell('G')


def in_bounds(r: int, c: int) -> bool:
    return 0 <= r < ROWS and 0 <= c < COLS


def is_wall(r: int, c: int) -> bool:
    return GRID[r][c] == '#'


def is_empty_cell(r: int, c: int) -> bool:
    return GRID[r][c] == '.'


# -----------------------------
# 3) Dynamic block manager
# -----------------------------
def decay_blocks(temp_blocks: Dict[Tuple[int, int], int]) -> None:
    expired = []
    for pos in list(temp_blocks.keys()):
        temp_blocks[pos] -= 1
        if temp_blocks[pos] <= 0:
            expired.append(pos)
    for pos in expired:
        del temp_blocks[pos]


def maybe_spawn_block(
    temp_blocks: Dict[Tuple[int, int], int],
    rng: random.Random,
) -> None:
    if len(temp_blocks) >= MAX_TEMP_BLOCKS:
        return
    if rng.random() >= P_SPAWN:
        return

    candidates = []
    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) == START or (r, c) == GOAL:
                continue
            # only spawn on empty cells (.)
            if is_empty_cell(r, c) and (r, c) not in temp_blocks:
                candidates.append((r, c))

    if not candidates:
        return

    pos = rng.choice(candidates)
    temp_blocks[pos] = BLOCK_DURATION


# -----------------------------
# 4) Environment step (includes dynamic blocks)
# -----------------------------
def env_step(
    state: Tuple[int, int],
    action: int,
    temp_blocks: Dict[Tuple[int, int], int],
    rng: random.Random,
) -> Tuple[Tuple[int, int], float, bool]:
    # Update blocks each step
    decay_blocks(temp_blocks)
    maybe_spawn_block(temp_blocks, rng)

    r, c = state
    dr, dc = ACTIONS[action]
    nr, nc = r + dr, c + dc

    # Invalid: outside, wall, or temporary block -> stay, penalty
    if (not in_bounds(nr, nc)) or is_wall(nr, nc) or ((nr, nc) in temp_blocks):
        return (r, c), INVALID_REWARD, False

    next_state = (nr, nc)

    if next_state == GOAL:
        return next_state, GOAL_REWARD, True

    return next_state, STEP_REWARD, False


# -----------------------------
# 5) Q-table (complete for all valid cells)
# -----------------------------
# Q[(r,c)][a] = value
Q: Dict[Tuple[int, int], List[float]] = {}
for r in range(ROWS):
    for c in range(COLS):
        if not is_wall(r, c):
            Q[(r, c)] = [0.0, 0.0, 0.0, 0.0]


def epsilon_greedy_action(state: Tuple[int, int], epsilon: float, rng: random.Random) -> int:
    if rng.random() < epsilon:
        return rng.choice(list(ACTIONS.keys()))
    qs = Q[state]
    max_q = max(qs)
    best_actions = [a for a, v in enumerate(qs) if v == max_q]
    return rng.choice(best_actions)


# -----------------------------
# 6) Training
# -----------------------------
def train(seed: int = 42) -> None:
    global EPSILON
    rng = random.Random(seed)

    for ep in range(1, EPISODES + 1):
        state = START
        total_reward = 0.0
        temp_blocks: Dict[Tuple[int, int], int] = {}

        for _ in range(MAX_STEPS):
            action = epsilon_greedy_action(state, EPSILON, rng)
            next_state, reward, done = env_step(state, action, temp_blocks, rng)

            old_q = Q[state][action]
            next_max = max(Q[next_state]) if next_state in Q else 0.0
            Q[state][action] = old_q + ALPHA * (reward + GAMMA * next_max - old_q)

            state = next_state
            total_reward += reward

            if done:
                break

        EPSILON = max(EPS_MIN, EPSILON * EPS_DECAY)

        if ep % 500 == 0:
            print(f"Episode {ep:4d} | epsilon={EPSILON:.3f} | total_reward={total_reward:.2f}")


# -----------------------------
# 7) Evaluation (Plan 1: small exploration during eval)
# -----------------------------
def run_eval_episode(seed: int = 123, eval_epsilon: float = 0.05):
    rng = random.Random(seed)
    state = START
    path = [state]

    temp_blocks: Dict[Tuple[int, int], int] = {}
    blocked_history = set()
    blocks_count = []

    for _ in range(MAX_STEPS):
        blocks_count.append(len(temp_blocks))
        for pos in temp_blocks:
            blocked_history.add(pos)

        if state == GOAL:
            break

        # Plan 1: keep small exploration in evaluation to escape loops
        action = epsilon_greedy_action(state, eval_epsilon, rng)
        next_state, _, done = env_step(state, action, temp_blocks, rng)

        path.append(next_state)
        state = next_state

        if done:
            break

    return path, blocked_history, blocks_count


# -----------------------------
# 8) Rendering
# -----------------------------
def render_final(path: List[Tuple[int, int]], blocked_history: set) -> None:
    view = [row[:] for row in GRID]

    # mark block history first
    for (r, c) in blocked_history:
        if view[r][c] == '.':
            view[r][c] = '!'

    # mark path, override ! if visited
    for (r, c) in path:
        if view[r][c] in ['.', '!']:
            view[r][c] = '*'

    print("\nFinal result:")
    print("Legend: * = path, ! = temporary block history, # = wall\n")
    for row in view:
        print(" ".join(row))


# -----------------------------
# 9) Print COMPLETE Q-table
# -----------------------------
def print_complete_q_table() -> None:
    """
    Print Q-values for every cell and every action.
    Walls are printed as N/A.
    """
    print("\nCOMPLETE Q-TABLE (all states and all actions)")
    print("Format: (row,col) | UP | RIGHT | DOWN | LEFT")
    print("-" * 70)

    for r in range(ROWS):
        for c in range(COLS):
            if is_wall(r, c):
                print(f"({r},{c}) WALL |  N/A   |  N/A   |  N/A   |  N/A")
            else:
                q = Q[(r, c)]
                print(f"({r},{c})      | {q[0]:7.3f} | {q[1]:7.3f} | {q[2]:7.3f} | {q[3]:7.3f}")


def best_action(state: Tuple[int, int]) -> Optional[int]:
    if state not in Q:
        return None
    qs = Q[state]
    m = max(qs)
    bests = [a for a, v in enumerate(qs) if v == m]
    return random.choice(bests)


def print_policy_map() -> None:
    """
    Print best action arrow for each cell (policy).
    """
    print("\nPOLICY MAP (best action per cell)")
    print("Legend: arrows = best move, # = wall, S/G kept\n")
    for r in range(ROWS):
        row_out = []
        for c in range(COLS):
            cell = GRID[r][c]
            if cell in ['S', 'G', '#']:
                row_out.append(cell)
            else:
                a = best_action((r, c))
                row_out.append(ARROWS[a] if a is not None else '?')
        print(" ".join(row_out))


if __name__ == "__main__":
    print("Training Q-learning agent on Dynamic 6x6 Grid World (temporary blocks)...")
    train(seed=42)

    path, blocked_history, blocks_count = run_eval_episode(seed=123, eval_epsilon=0.05)
    render_final(path, blocked_history)

    print("\nReached goal?", path[-1] == GOAL)
    print("Path length:", len(path))
    print("Avg active temp blocks during eval:", sum(blocks_count) / max(1, len(blocks_count)))

    print_complete_q_table()
    print_policy_map()
