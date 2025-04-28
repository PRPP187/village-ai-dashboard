import numpy as np
import random
import json
import os
import glob
import pandas as pd
from collections import deque
import time

# Grid Settings
GRID_ROWS = 3
GRID_COLS = 3
EPISODES = 10000
ALPHA = 0.1
GAMMA = 0.9
SCORES = {'E': 10, 'G': 10, 'H': 15, 'R': 5, '0': 0}
Q_TABLE_FILE = "q_table.json"
E_START_POSITION = (1, 1)

# House Price Settings
HOUSE_PRICES = {
    'H1': {'cost': 1_800_000, 'sale': 3_300_000, 'size': 110, 'weight': 1.3},
    'H2': {'cost': 1_900_000, 'sale': 3_700_000, 'size': 125, 'weight': 1.1},
    'H3': {'cost': 2_300_000, 'sale': 4_300_000, 'size': 160, 'weight': 1.0},
    'H4': {'cost': 3_000_000, 'sale': 5_200_000, 'size': 200, 'weight': 0.9},
}

# CSV Loading
csv_folder = "data/maps/CSV/goodcsv"
csv_files = glob.glob(f"{csv_folder}/**/*.csv", recursive=True)

# Optimize house type ratios
def optimize_ratios():
    ratios = {}
    total_score = sum((p['sale'] - p['cost']) * p['weight'] for p in HOUSE_PRICES.values())
    for htype, data in HOUSE_PRICES.items():
        score = (data['sale'] - data['cost']) * data['weight']
        ratios[htype] = score / total_score
    return ratios

H_TYPE_RATIOS = optimize_ratios()

# Calculate reward score
def calculate_reward_verbose(grid):
    """ Calculate total score based on house-road-green pattern and connectivity rules (upgraded version). """

    if grid is None or len(grid) == 0 or len(grid[0]) == 0:
        print("⚠️ Error: Grid is None or empty! Using blank grid instead.")
        grid_size = 5
        grid = np.full((grid_size, grid_size), '0')

    grid = np.array(grid)
    rows, cols = grid.shape

    # Base Scores (fixed)
    base_score = sum(SCORES.get(cell, 0) for row in grid for cell in row)

    # Bonus
    bonus = 0

    # Bonus: H at edge
    h_positions = np.argwhere(grid == 'H')
    edge_houses = np.sum((h_positions[:, 0] == 0) | (h_positions[:, 0] == rows - 1) |
                         (h_positions[:, 1] == 0) | (h_positions[:, 1] == cols - 1))
    bonus += edge_houses * 50

    # Bonus: Pattern bonuses (Face to North)
    bonus += np.sum((grid[:, :-2] == 'H') & (grid[:, 1:-1] == 'H') & (grid[:, 2:] == 'H')) * 100  # H-H-H แนวนอน
    bonus += np.sum((grid[:, :-2] == 'R') & (grid[:, 1:-1] == 'R') & (grid[:, 2:] == 'R')) * 100  # R-R-R แนวนอน
    bonus += np.sum((grid[:-2, :] == 'H') & (grid[1:-1, :] == 'R') & (grid[2:, :] == 'H')) * 100  # H-R-H แนวตั้ง
    bonus += np.sum((grid[:-1, :] == 'R') & (grid[1:, :] == 'R') & (grid[:-1, :] == 'H') & (grid[1:, :] == 'H')) * 100  # RR-HH
    bonus += np.sum((grid[:, :-1] == 'H') & (grid[:, 1:] == 'H') & (grid[:, :-1] == 'R') & (grid[:, 1:] == 'R')) * 100  # HR-HR
    
    # Bonus: Pattern bonuses (Face to South)
    bonus += np.sum((grid[:, :-2] == 'H') & (grid[:, 1:-1] == 'R') & (grid[:, 2:] == 'H')) * 50   # H R H แนวนอน
    bonus += np.sum((grid[:-2, :] == 'H') & (grid[1:-1, :] == 'H') & (grid[2:, :] == 'H')) * 50    # H-H-H แนวตั้ง
    bonus += np.sum((grid[:-2, :] == 'R') & (grid[1:-1, :] == 'R') & (grid[2:, :] == 'R')) * 50    # R-R-R แนวตั้ง


    # Penalty
    penalty = 0

    # ✅ Penalty ใหม่: ถ้า H น้อยกว่า R
    num_houses = np.sum(grid == 'H')
    num_roads = np.sum(grid == 'R')
    if num_houses < num_roads:
        penalty -= 1000
    
    # Penalty: H not connected to R
    for r, c in h_positions:
        has_road_neighbor = any(
            0 <= r+dr < rows and 0 <= c+dc < cols and grid[r+dr, c+dc] == 'R'
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]
        )
        if not has_road_neighbor:
            penalty -= 1000

    # Penalty: E not connected to R
    e_positions = np.argwhere(grid == 'E')
    for r, c in e_positions:
        has_road_neighbor = any(
            0 <= r+dr < rows and 0 <= c+dc < cols and grid[r+dr, c+dc] == 'R'
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]
        )
        if not has_road_neighbor:
            penalty -= 1000

    # Penalty: R not connected (count clusters)
    r_clusters = count_r_clusters(grid) if np.any(grid == 'R') else 0
    if r_clusters > 1:
        penalty -= 500 * r_clusters
    elif r_clusters == 0:
        penalty -= 1000

    # Penalty: Wrong adjacent patterns
    penalty -= np.sum((grid[:-1, :] == 'H') & (grid[1:, :] == 'H') & (grid[:-1, :] == 'R') & (grid[1:, :] == 'R')) * 50
    penalty -= np.sum((grid[:, :-1] == 'R') & (grid[:, 1:] == 'R') & (grid[:, :-1] == 'H') & (grid[:, 1:] == 'H')) * 50

    # Penalty: Green ratio too low or too high
    total_cells = rows * cols
    green_cells = np.sum(grid == 'G')
    green_ratio = green_cells / total_cells
    if green_ratio < 0.05:
        penalty -= 1000
    if green_ratio > 0.20:
        penalty -= 500

    # Final score
    total_score = base_score + bonus + penalty

    print(f"🎯 Debug: Total Grid Score = {total_score} (Base: {base_score}, Bonus: {bonus}, Penalty: {penalty})")
    return total_score

def count_r_clusters(grid, use_dfs=False):
    rows, cols = len(grid), len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    clusters = 0

    def bfs(r, c):
        queue = deque([(r, c)])
        visited[r][c] = True
        while queue:
            x, y = queue.popleft()
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and grid[nx][ny] == 'R':
                    visited[nx][ny] = True
                    queue.append((nx, ny))

    def dfs(r, c):
        stack = [(r, c)]
        visited[r][c] = True
        while stack:
            x, y = stack.pop()
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and grid[nx][ny] == 'R':
                    visited[nx][ny] = True
                    stack.append((nx, ny))

    search = dfs if use_dfs else bfs

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 'R' and not visited[r][c]:
                clusters += 1
                search(r, c)

    return clusters

def apply_house_types(grid):
    h_positions = [(r, c) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] == 'H']
    total_h = len(h_positions)
    if total_h == 0:
        return grid

    house_counts = {htype: int(ratio * total_h) for htype, ratio in H_TYPE_RATIOS.items()}
    while sum(house_counts.values()) < total_h:
        for htype in house_counts:
            house_counts[htype] += 1
            if sum(house_counts.values()) == total_h:
                break

    house_sequence = []
    for htype, count in house_counts.items():
        house_sequence.extend([htype] * count)

    for (r, c), htype in zip(h_positions, house_sequence):
        grid[r][c] = htype

    return grid

def initialize_grid(rows, cols, e_position):
    print(f"Creating blank {rows}x{cols} grid, placing E at {e_position} (1-based index)")
    grid = [['0' for _ in range(cols)] for _ in range(rows)]
    r, c = e_position

    if r == 1 or r == rows or c == 1 or c == cols:
        grid[r-1][c-1] = 'E'
    else:
        distances = {
            "top": r - 1,
            "bottom": rows - r,
            "left": c - 1,
            "right": cols - c
        }
        sorted_edges = sorted(distances.items(), key=lambda x: x[1])
        for edge, _ in sorted_edges:
            if edge == "top":
                r = 1
                break
            elif edge == "bottom":
                r = rows
                break
            elif edge == "left":
                c = 1
                break
            elif edge == "right":
                c = cols
                break
        grid[r-1][c-1] = 'E'

    return grid, (r, c)

def load_or_initialize_grid(csv_folder, rows, cols, e_position):
    print(f"Searching CSVs {rows}x{cols} with E at {e_position}")
    csv_files = glob.glob(f"{csv_folder}/**/*.csv", recursive=True)

    if not csv_files:
        print("No CSVs found! Creating new grid.")
        return initialize_grid(rows, cols, e_position)

    best_grid = None
    best_score = float('-inf')

    for file in csv_files:
        df = pd.read_csv(file, header=None)
        if df.shape != (rows, cols):
            continue
        grid_from_csv = df.astype(str).values.tolist()
        e_found = [(r+1, c+1) for r in range(len(grid_from_csv)) for c in range(len(grid_from_csv[0])) if grid_from_csv[r][c] == 'E']
        if not e_found or e_found[0] != e_position:
            continue

        score = calculate_reward_verbose(grid_from_csv)
        if score > best_score:
            best_score = score
            best_grid = grid_from_csv

    if best_grid is not None:
        return best_grid, e_position
    return initialize_grid(rows, cols, e_position)

def choose_action(grid, e_position=(1,1), epsilon=0.1):
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid)

    rows, cols = grid.shape
    visited = set()
    queue = deque([(e_position[0]-1, e_position[1]-1)])
    build_order = []

    while queue:
        r, c = queue.popleft()
        if (r, c) in visited:
            continue
        visited.add((r, c))

        if grid[r, c] == '0':
            build_order.append((r, c))

        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                queue.append((nr, nc))

    if not build_order:
        return None

    # Simplify: สุ่มเลือก cell แล้วสุ่ม H, R, G
    r, c = random.choice(build_order)
    char = random.choices(['H', 'R', 'G'], weights=[0.5, 0.4, 0.1])[0]
    return r, c, char


def update_q_table(state, action, reward, next_state):
    state_str = json.dumps(state)
    action_str = str(tuple(action))
    next_state_str = json.dumps(next_state)

    if state_str not in q_table:
        q_table[state_str] = {}
    if action_str not in q_table[state_str]:
        q_table[state_str][action_str] = 0

    max_future_q = max(q_table.get(next_state_str, {}).values() or [0])
    q_table[state_str][action_str] = (1 - ALPHA) * q_table[state_str][action_str] + ALPHA * (reward + GAMMA * max_future_q)

def train_ai(episodes, grid):
    best_grid = None
    best_score = float('-inf')

    for episode in range(episodes):
        state = [row[:] for row in grid]
        rows = len(state)
        cols = len(state[0]) if state else 0
        max_steps = rows * cols

        for _ in range(max_steps):
            action = choose_action(state, new_e_position)
            if action is None:
                break
            r, c, char = action
            state[r][c] = char
            reward = calculate_reward_verbose(state)
            update_q_table(state, action, reward, state)

        total_reward = calculate_reward_verbose(state)

        if total_reward > best_score:
            best_score = total_reward
            best_grid = [row[:] for row in state]

    return best_grid, best_score

def measure_execution_time(function, *args, **kwargs):
    start_time = time.time()
    result = function(*args, **kwargs)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return *result, elapsed_time

def analyze_profit(grid):
    summary = {k: 0 for k in HOUSE_PRICES}
    total_cost = total_sale = total_size = weighted_profit = 0

    for row in grid:
        for cell in row:
            if cell in HOUSE_PRICES:
                info = HOUSE_PRICES[cell]
                summary[cell] += 1
                total_cost += info['cost']
                total_sale += info['sale']
                total_size += info['size']
                weighted_profit += (info['sale'] - info['cost']) * info['weight']

    total_profit = total_sale - total_cost
    avg_profit_per_sqm = total_profit / total_size if total_size else 0

    print("\n📋 House Type Summary:\n")
    print("House Type | Number of Units | Cost/Unit | Sale/Unit | Profit/Unit | Total Cost | Total Profit")
    print("-" * 120)

    total_units = sum(summary.values())
    for htype, count in summary.items():
        if count:
            info = HOUSE_PRICES[htype]
            cost_per_unit = info['cost']
            sale_per_unit = info['sale']
            profit_per_unit = sale_per_unit - cost_per_unit
            total_cost_type = cost_per_unit * count
            total_profit_type = profit_per_unit * count

            print(f"🏠 {htype} | {count} units | {cost_per_unit:,.0f} Baht | {sale_per_unit:,.0f} Baht | {profit_per_unit:,.0f} Baht | {total_cost_type:,.0f} Baht | {total_profit_type:,.0f} Baht")

    print("\n💸 Total Construction Cost:", f"{total_cost:,} Baht")
    print("💰 Total Revenue:", f"{total_sale:,} Baht")
    print("📈 Total Profit:", f"{total_profit:,} Baht")
    print("📐 Average Profit per sqm:", f"{avg_profit_per_sqm:,.2f} Baht/sqm")
    print("🎯 Weighted Profit (Market Preference):", f"{weighted_profit:,.2f} Baht")

# Start execution
q_table = {}

grid, new_e_position = initialize_grid(GRID_ROWS, GRID_COLS, E_START_POSITION)
grid, _ = load_or_initialize_grid(csv_folder, GRID_ROWS, GRID_COLS, new_e_position)

best_grid, best_score, execution_time = measure_execution_time(train_ai, EPISODES, grid)
final_layout = apply_house_types([row[:] for row in best_grid])

print("\n🏆 Best Layout Found:")
for row in best_grid:
    print(" ".join(row))
print(f"\n✅ Best Score Achieved: {best_score}")

print("\n📌 Final Layout with House Types:")
for row in final_layout:
    print(" ".join(row))

analyze_profit(final_layout)
