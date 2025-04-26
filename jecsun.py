import numpy as np
import random
import json
import os
import glob
import pandas as pd
from collections import deque
import time
from datetime import datetime

# --- Grid settings ---
GRID_ROWS = 3
GRID_COLS = 3
EPISODES = 10000
ALPHA = 0.1
GAMMA = 0.9
SCORES = {'E': 10, 'G': 10, 'H': 15, 'R': 5, '0': 0}
Q_TABLE_FILE = "q_table.json"
E_START_POSITION = (1, 1)

# --- House type data ---
HOUSE_PRICES = {
    'H1': {'cost': 2_000_000, 'sale': 2_800_000, 'size': 120, 'weight': 1.0},
    'H2': {'cost': 1_800_000, 'sale': 2_500_000, 'size': 100, 'weight': 1.0},
    'H3': {'cost': 2_200_000, 'sale': 2_900_000, 'size': 140, 'weight': 1.1},
    'H4': {'cost': 2_500_000, 'sale': 3_200_000, 'size': 160, 'weight': 0.9},
}

csv_folder = "data/maps/CSV/goodcsv"

# --- Helper functions ---
def optimize_ratios():
    ratios = {}
    total_score = sum((p['sale'] - p['cost']) * p['weight'] for p in HOUSE_PRICES.values())
    for htype, data in HOUSE_PRICES.items():
        score = (data['sale'] - data['cost']) * data['weight']
        ratios[htype] = score / total_score
    return ratios

H_TYPE_RATIOS = optimize_ratios()

def save_q_table(filename=Q_TABLE_FILE):
    with open(filename, "w") as f:
        json.dump(q_table, f)
    print(f"💾 Q-table saved to: {filename}")

def load_q_table(filename=Q_TABLE_FILE):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    else:
        print(f"📁 Q-table file not found: {filename}")
        return {}

def calculate_reward_verbose(grid):
    if grid is None or len(grid) == 0 or len(grid[0]) == 0:
        grid = np.full((5, 5), '0')
    grid = np.array(grid)
    rows, cols = grid.shape
    base_score = sum(SCORES.get(cell, 0) for row in grid for cell in row)

    bonus = 0
    bonus += np.sum((grid[:, :-2] == 'H') & (grid[:, 1:-1] == 'H') & (grid[:, 2:] == 'H')) * 100
    bonus += np.sum((grid[:, :-2] == 'R') & (grid[:, 1:-1] == 'R') & (grid[:, 2:] == 'R')) * 100
    bonus += np.sum((grid[:-2, :] == 'H') & (grid[1:-1, :] == 'R') & (grid[2:, :] == 'H')) * 100
    bonus += np.sum((grid[:-1, :-1] == 'H') & (grid[:-1, 1:] == 'H') &
                    (grid[1:, :-1] == 'R') & (grid[1:, 1:] == 'R')) * 100
    bonus += np.sum((grid[:-1, :-1] == 'R') & (grid[:-1, 1:] == 'R') &
                    (grid[1:, :-1] == 'H') & (grid[1:, 1:] == 'H')) * 100

    penalty = 0
    h_positions = np.argwhere(grid == 'H')
    e_positions = np.argwhere(grid == 'E')

    if len(e_positions) > 0:
        e_neighbors_r = np.any([
            np.roll(grid == 'R', shift, axis=axis)[e_positions[:, 0], e_positions[:, 1]]
            for shift, axis in [(1, 0), (-1, 0), (1, 1), (-1, 1)]
        ], axis=0)
        if not np.any(e_neighbors_r):
            penalty -= 200

    r_clusters = count_r_clusters(grid) if np.any(grid == 'R') else 0
    if r_clusters > 1:
        penalty -= 500 * (r_clusters - 1)

    if len(h_positions) > 0:
        h_neighbors_r = np.any([
            np.roll(grid == 'R', shift, axis=axis)[h_positions[:, 0], h_positions[:, 1]]
            for shift, axis in [(1, 0), (-1, 0), (1, 1), (-1, 1)]
        ], axis=0)
        penalty -= 200 * np.sum(~h_neighbors_r)
        if np.all(h_neighbors_r):
            bonus += 100

    green_ratio = np.sum(grid == 'G') / (rows * cols)
    if green_ratio < 0.1:
        penalty -= 500
    if np.sum(grid == 'R') == 0:
        penalty -= 500

    return base_score + bonus + penalty

def count_r_clusters(grid, use_dfs=False):
    GRID_ROWS, GRID_COLS = len(grid), len(grid[0])
    visited = [[False]*GRID_COLS for _ in range(GRID_ROWS)]
    clusters = 0

    def bfs(r, c):
        queue = deque([(r, c)])
        visited[r][c] = True
        while queue:
            x, y = queue.popleft()
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_ROWS and 0 <= ny < GRID_COLS and not visited[nx][ny] and grid[nx][ny] == 'R':
                    visited[nx][ny] = True
                    queue.append((nx, ny))

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if grid[r][c] == 'R' and not visited[r][c]:
                clusters += 1
                bfs(r, c)
    return clusters

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

def choose_action(grid):
    rows, cols = len(grid), len(grid[0])
    empty_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == '0']
    if not empty_cells:
        return None
    r, c = random.choice(empty_cells)
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
    max_future_q = max(q_table.get(next_state_str, {}).values() or [0])
    q_table[state_str][action_str] = (1-ALPHA)*q_table[state_str][action_str] + ALPHA*(reward + GAMMA*max_future_q)

def train_ai(episodes, grid):
    global q_table
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
            update_q_table(state, action, reward, state)
            action_log.append(f"EP:{episode+1} STEP:{step+1} ➤ '{char}' at ({r+1},{c+1}) [was '{prev}'] → Reward: {reward}")

        total_reward = calculate_reward_verbose(state)
        if len(top_layouts) < 3:
            top_layouts.append((total_reward, [row[:] for row in state]))
        else:
            min_score = min(top_layouts, key=lambda x: x[0])[0]
            if total_reward > min_score:
                top_layouts = sorted(top_layouts, key=lambda x: x[0], reverse=True)
                top_layouts[-1] = (total_reward, [row[:] for row in state])
    return sorted(top_layouts, key=lambda x: x[0], reverse=True), action_log

def save_training_results(final_layouts, action_log, duration_sec):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = "data/results"
    os.makedirs(folder, exist_ok=True)

    for i, (score, layout) in enumerate(final_layouts, 1):
        pd.DataFrame(layout).to_csv(f"{folder}/layout_{i}_{now}.csv", index=False, header=False)

    summary = {
        "timestamp": now,
        "duration_sec": duration_sec,
        "duration_min": duration_sec/60,
        "layouts": [{"layout_num": i+1, "score": score} for i, (score, _) in enumerate(final_layouts)],
        "action_log_sample": action_log[:50]
    }
    with open(f"{folder}/summary_{now}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

# --- Execute ---
q_table = load_q_table()

grid, new_e_position = initialize_grid(GRID_ROWS, GRID_COLS, E_START_POSITION)
grid, _ = load_or_initialize_grid(csv_folder, GRID_ROWS, GRID_COLS, new_e_position)

print(f"\n✅ Grid size: {len(grid)} rows x {len(grid[0])} cols | E Position: {new_e_position}")

start_time = time.time()
top_layouts, action_log = train_ai(EPISODES, grid)
duration_sec = time.time() - start_time

best_score, best_grid = top_layouts[0]

# Show all layouts
final_layouts = []
for i, (score, layout) in enumerate(top_layouts, 1):
    print(f"\n🏆 Layout #{i} — Raw Score: {score}")
    for row in layout:
        print(" ".join(row))
    final_layout = apply_house_types([row[:] for row in layout])
    final_layouts.append((score, final_layout))
    print(f"\n📌 Layout #{i} with H1–H4:")
    for row in final_layout:
        print(" ".join(row))
    print("\n" + "-"*50)

# Save
save_q_table()
save_training_results(final_layouts, action_log, duration_sec)
