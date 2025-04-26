# jecsun.py

import numpy as np
import random
import json
import os
import glob
import pandas as pd
from collections import deque

# --- Grid settings ---
GRID_ROWS = 3
GRID_COLS = 3
EPISODES = 10000
ALPHA = 0.1
GAMMA = 0.9
SCORES = {'E': 10, 'G': 10, 'H': 15, 'R': 5, '0': 0}
Q_TABLE_FILE = "q_table.json"
E_START_POSITION = (1, 1)
csv_folder = "data/maps/CSV/goodcsv"

# --- House type data ---
HOUSE_PRICES = {
    'H1': {'cost': 2_000_000, 'sale': 2_800_000, 'size': 120, 'weight': 1.0},
    'H2': {'cost': 1_800_000, 'sale': 2_500_000, 'size': 100, 'weight': 1.0},
    'H3': {'cost': 2_200_000, 'sale': 2_900_000, 'size': 140, 'weight': 1.1},
    'H4': {'cost': 2_500_000, 'sale': 3_200_000, 'size': 160, 'weight': 0.9},
}

# --- Functions only ---
def optimize_ratios():
    ratios = {}
    total_score = sum((p['sale'] - p['cost']) * p['weight'] for p in HOUSE_PRICES.values())
    for htype, data in HOUSE_PRICES.items():
        score = (data['sale'] - data['cost']) * data['weight']
        ratios[htype] = score / total_score
    return ratios

H_TYPE_RATIOS = optimize_ratios()

def save_q_table(q_table, filename=Q_TABLE_FILE):
    with open(filename, "w") as f:
        json.dump(q_table, f)

def load_q_table(filename=Q_TABLE_FILE):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return {}

def initialize_grid(rows, cols, e_position):
    grid = [['0' for _ in range(cols)] for _ in range(rows)]
    r, c = e_position
    if r == 1 or r == rows or c == 1 or c == cols:
        grid[r-1][c-1] = 'E'
    else:
        distances = {"top": r-1, "bottom": rows-r, "left": c-1, "right": cols-c}
        edge = min(distances, key=distances.get)
        if edge == "top": r = 1
        elif edge == "bottom": r = rows
        elif edge == "left": c = 1
        elif edge == "right": c = cols
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
        if df.shape != (rows, cols): continue
        grid = df.astype(str).values.tolist()
        e_found = [(r+1, c+1) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] == 'E']
        if not e_found or e_found[0] != e_position: continue
        score = calculate_reward_verbose(grid)
        if score > best_score:
            best_score = score
            best_grid = grid
    if best_grid is not None:
        return best_grid, e_position
    return initialize_grid(rows, cols, e_position)

def calculate_reward_verbose(grid):
    base_score = sum(SCORES.get(cell, 0) for row in grid for cell in row)
    return base_score  # (สั้นลง เวอร์ชันง่าย)

def choose_action(grid):
    rows, cols = len(grid), len(grid[0])
    empty_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == '0']
    if not empty_cells:
        return None
    r, c = random.choice(empty_cells)
    char = random.choice(['H', 'R', 'G'])
    return r, c, char

def update_q_table(q_table, state, action, reward, next_state):
    state_str = json.dumps(state)
    action_str = str(tuple(action))
    next_state_str = json.dumps(next_state)
    if state_str not in q_table:
        q_table[state_str] = {}
    if action_str not in q_table[state_str]:
        q_table[state_str][action_str] = 0
    max_future_q = max(q_table.get(next_state_str, {}).values() or [0])
    q_table[state_str][action_str] = (1-ALPHA)*q_table[state_str][action_str] + ALPHA*(reward + GAMMA*max_future_q)

def train_ai(episodes, grid):
    q_table = load_q_table()
    top_layouts = []
    action_log = []
    for episode in range(episodes):
        state = [row[:] for row in grid]
        for step in range(len(state)*len(state[0])):
            action = choose_action(state)
            if action is None: break
            r, c, char = action
            prev = state[r][c]
            state[r][c] = char
            reward = calculate_reward_verbose(state)
            update_q_table(q_table, state, action, reward, state)
            action_log.append(f"EP:{episode+1} STEP:{step+1} ➤ '{char}' at ({r+1},{c+1}) [was '{prev}'] → Reward: {reward}")
        total_reward = calculate_reward_verbose(state)
        if len(top_layouts) < 3:
            top_layouts.append((total_reward, [row[:] for row in state]))
        else:
            min_score = min(top_layouts, key=lambda x: x[0])[0]
            if total_reward > min_score:
                top_layouts = sorted(top_layouts, key=lambda x: x[0], reverse=True)
                top_layouts[-1] = (total_reward, [row[:] for row in state])
    save_q_table(q_table)
    return sorted(top_layouts, key=lambda x: x[0], reverse=True), action_log

def apply_house_types(grid):
    h_positions = [(r, c) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] == 'H']
    total_h = len(h_positions)
    if total_h == 0:
        return grid
    house_types_count = {htype: int(ratio * total_h) for htype, ratio in H_TYPE_RATIOS.items()}
    current_total = sum(house_types_count.values())
    while current_total < total_h:
        for htype in house_types_count:
            house_types_count[htype] += 1
            current_total += 1
            if current_total == total_h:
                break
    house_sequence = []
    for htype, count in house_types_count.items():
        house_sequence.extend([htype] * count)
    for (r, c), htype in zip(h_positions, house_sequence):
        grid[r][c] = htype
    return grid

# No code runs here directly!! (only functions)
