import numpy as np
import random
import json
import os
import pandas as pd
import glob
import time
from collections import deque

# --- Config ---
GRID_ROWS = 3
GRID_COLS = 3
EPISODES = 10000
ALPHA = 0.1
GAMMA = 0.9
SCORES = {'E': 10, 'G': 10, 'H': 15, 'R': 5, '0': 0}
Q_TABLE_FILE = "q_table.json"
BEST_LAYOUT_FILE = "best_layout.json"
E_START_POSITION = (1, 1)
csv_folder = "data/maps/CSV/goodcsv"

HOUSE_PRICES = {
    'H1': {'cost': 2_000_000, 'sale': 2_800_000, 'size': 120, 'weight': 1.0},
    'H2': {'cost': 1_800_000, 'sale': 2_500_000, 'size': 100, 'weight': 1.0},
    'H3': {'cost': 2_200_000, 'sale': 2_900_000, 'size': 140, 'weight': 1.1},
    'H4': {'cost': 2_500_000, 'sale': 3_200_000, 'size': 160, 'weight': 0.9},
}

# --- Utilities ---
def optimize_ratios():
    total = sum((v['sale'] - v['cost']) * v['weight'] for v in HOUSE_PRICES.values())
    return {k: (v['sale'] - v['cost']) * v['weight'] / total for k, v in HOUSE_PRICES.items()}

H_TYPE_RATIOS = optimize_ratios()

# --- Grid Functions ---
def initialize_grid(rows, cols, e_position):
    grid = [['0' for _ in range(cols)] for _ in range(rows)]
    r, c = e_position
    grid[r-1][c-1] = 'E'
    return grid, (r, c)

def load_or_initialize_grid(csv_folder, rows, cols, e_position):
    csv_files = glob.glob(f"{csv_folder}/**/*.csv", recursive=True)
    if not csv_files:
        return initialize_grid(rows, cols, e_position)

    best_grid = None
    best_score = float('-inf')

    for file in csv_files:
        df = pd.read_csv(file, header=None)
        if df.shape != (rows, cols):
            continue
        grid = df.astype(str).values.tolist()
        e_found = [(r+1, c+1) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] == 'E']
        if e_found and e_found[0] == e_position:
            score = calculate_reward(grid)
            if score > best_score:
                best_score = score
                best_grid = grid

    if best_grid:
        return best_grid, e_position
    return initialize_grid(rows, cols, e_position)

# --- Reward Functions ---
def calculate_reward(grid):
    base = sum(SCORES.get(cell, 0) for row in grid for cell in row)
    bonus = 0
    penalty = 0

    grid = np.array(grid)

    bonus += np.sum((grid[:, :-2] == 'H') & (grid[:, 1:-1] == 'H') & (grid[:, 2:] == 'H')) * 100
    bonus += np.sum((grid[:, :-2] == 'R') & (grid[:, 1:-1] == 'R') & (grid[:, 2:] == 'R')) * 100
    
    total_cells = grid.size
    if np.sum(grid == 'G') / total_cells < 0.1:
        penalty -= 500
    if np.sum(grid == 'R') == 0:
        penalty -= 500

    return base + bonus + penalty

# --- AI Functions ---
def choose_action(grid):
    empty = [(r, c) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] == '0']
    if not empty:
        return None
    r, c = random.choice(empty)
    char = random.choice(['H', 'R', 'G'])
    return r, c, char

def update_q_table(state, action, reward, next_state):
    global q_table
    state_str = json.dumps(state)
    action_str = str(tuple(action))
    next_state_str = json.dumps(next_state)
    
    if state_str not in q_table:
        q_table[state_str] = {}
    if action_str not in q_table[state_str]:
        q_table[state_str][action_str] = 0

    max_future = max(q_table.get(next_state_str, {}).values() or [0])
    q_table[state_str][action_str] = (1 - ALPHA) * q_table[state_str][action_str] + ALPHA * (reward + GAMMA * max_future)

# --- Training ---
def train_ai(episodes, grid):
    global q_table
    best_grid = None
    best_score = float('-inf')
    
    for _ in range(episodes):
        state = [row[:] for row in grid]
        steps = len(state) * len(state[0])

        for _ in range(steps):
            action = choose_action(state)
            if action is None:
                break
            r, c, char = action
            prev = state[r][c]
            state[r][c] = char
            reward = calculate_reward(state)
            update_q_table(state, action, reward, state)

        total_reward = calculate_reward(state)
        if total_reward > best_score:
            best_score = total_reward
            best_grid = [row[:] for row in state]

    return best_grid, best_score

# --- Layout Functions ---
def apply_house_types(grid):
    h_positions = [(r, c) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] == 'H']
    total_h = len(h_positions)
    if total_h == 0:
        return grid

    house_sequence = []
    for htype, ratio in H_TYPE_RATIOS.items():
        house_sequence.extend([htype] * int(ratio * total_h))

    while len(house_sequence) < total_h:
        house_sequence.append('H1')

    for (r, c), htype in zip(h_positions, house_sequence):
        grid[r][c] = htype

    return grid

# --- Save / Load ---
def save_q_table(filename=Q_TABLE_FILE):
    with open(filename, 'w') as f:
        json.dump(q_table, f)

def save_best_layout(layout, filename=BEST_LAYOUT_FILE):
    with open(filename, 'w') as f:
        json.dump(layout, f)

def load_q_table(filename=Q_TABLE_FILE):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return {}

# --- Initialize ---
q_table = load_q_table()
