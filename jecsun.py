# jecsun.py

import numpy as np
import random
import json
import os
import pandas as pd
import glob
import time
from collections import deque

# --- Global Settings ---
GRID_ROWS = 3
GRID_COLS = 3
EPISODES = 10000
ALPHA = 0.1
GAMMA = 0.9
E_START_POSITION = (1, 1)
Q_TABLE_FILE = "q_table.json"
csv_folder = "data/maps/CSV/goodcsv"
save_folder = "data/saved_layouts/"

HOUSE_PRICES = {
    'H1': {'cost': 2_000_000, 'sale': 2_800_000, 'size': 120, 'weight': 1.0},
    'H2': {'cost': 1_800_000, 'sale': 2_500_000, 'size': 100, 'weight': 1.0},
    'H3': {'cost': 2_200_000, 'sale': 2_900_000, 'size': 140, 'weight': 1.1},
    'H4': {'cost': 2_500_000, 'sale': 3_200_000, 'size': 160, 'weight': 0.9},
}

os.makedirs(save_folder, exist_ok=True)

def optimize_ratios():
    ratios = {}
    total = sum((p['sale'] - p['cost']) * p['weight'] for p in HOUSE_PRICES.values())
    for htype, data in HOUSE_PRICES.items():
        ratios[htype] = ((data['sale'] - data['cost']) * data['weight']) / total
    return ratios

H_TYPE_RATIOS = optimize_ratios()

def save_q_table():
    with open(Q_TABLE_FILE, "w") as f:
        json.dump(q_table, f)

def load_q_table():
    if os.path.exists(Q_TABLE_FILE):
        with open(Q_TABLE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_grid(grid, filename):
    df = pd.DataFrame(grid)
    df.to_csv(os.path.join(save_folder, filename), index=False, header=False)

def initialize_grid(rows, cols, e_position):
    grid = [['0' for _ in range(cols)] for _ in range(rows)]
    r, c = e_position
    if r == 1 or r == rows or c == 1 or c == cols:
        grid[r-1][c-1] = 'E'
    else:
        if r - 1 <= rows - r:
            r = 1
        else:
            r = rows
        if c - 1 <= cols - c:
            c = 1
        else:
            c = cols
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
        if not e_found or e_found[0] != e_position:
            continue
        score = calculate_reward_verbose(grid)
        if score > best_score:
            best_score = score
            best_grid = grid

    if best_grid is not None:
        return best_grid, e_position
    return initialize_grid(rows, cols, e_position)

def calculate_reward_verbose(grid):
    grid = np.array(grid)
    base_score = sum({'E':10,'G':10,'H':15,'R':5,'0':0}.get(cell, 0) for row in grid for cell in row)

    bonus, penalty = 0, 0
    rows, cols = grid.shape

    bonus += np.sum((grid[:, :-2] == 'H') & (grid[:, 1:-1] == 'H') & (grid[:, 2:] == 'H')) * 100
    bonus += np.sum((grid[:, :-2] == 'R') & (grid[:, 1:-1] == 'R') & (grid[:, 2:] == 'R')) * 100

    if np.sum(grid == 'G') / (rows * cols) < 0.1:
        penalty -= 500
    if np.sum(grid == 'R') == 0:
        penalty -= 500

    return base_score + bonus + penalty

def apply_house_types(grid):
    h_positions = [(r, c) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] == 'H']
    total_h = len(h_positions)
    if total_h == 0:
        return grid

    house_types = []
    counts = {k: int(v * total_h) for k,v in H_TYPE_RATIOS.items()}
    while sum(counts.values()) < total_h:
        for k in counts:
            counts[k] += 1
            if sum(counts.values()) == total_h:
                break
    for k,v in counts.items():
        house_types.extend([k]*v)

    for (r,c), h in zip(h_positions, house_types):
        grid[r][c] = h
    return grid

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
    q_table[state_str][action_str] = (1-ALPHA)*q_table[state_str][action_str] + ALPHA*(reward + GAMMA*max_future)

def train_ai(episodes, grid):
    global q_table
    top_layouts = []
    start_time = time.time()

    for _ in range(episodes):
        state = [row[:] for row in grid]
        for _ in range(len(state)*len(state[0])):
            action = choose_action(state)
            if action is None:
                break
            r,c,char = action
            state[r][c] = char
            reward = calculate_reward_verbose(state)
            update_q_table(state, action, reward, state)

        total_reward = calculate_reward_verbose(state)

        if len(top_layouts) < 3:
            top_layouts.append((total_reward, [row[:] for row in state]))
        else:
            top_layouts = sorted(top_layouts, key=lambda x: x[0], reverse=True)
            if total_reward > top_layouts[-1][0]:
                top_layouts[-1] = (total_reward, [row[:] for row in state])

    top_layouts = sorted(top_layouts, key=lambda x: x[0], reverse=True)

    # Save best grid
    save_grid(top_layouts[0][1], f"best_layout_{int(time.time())}.csv")

    elapsed_sec = time.time() - start_time
    return top_layouts, elapsed_sec

# Load q_table when start
q_table = load_q_table()
