# ... (import เหมือนเดิม)

# ✅ เพิ่ม argument ให้รองรับ green_ratio_min

def calculate_reward_verbose(grid, green_ratio_min=0.1):
    if grid is None or len(grid) == 0 or len(grid[0]) == 0:
        grid_size = 5
        grid = np.full((grid_size, grid_size), '0')

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

    total_cells = rows * cols
    num_green = np.sum(grid == 'G')
    green_ratio = num_green / total_cells

    if green_ratio < green_ratio_min:
        penalty -= 500

    roads_exist = np.sum(grid == 'R') > 0
    if not roads_exist:
        penalty -= 500

    total_score = base_score + bonus + penalty
    return total_score

# ✅ อัปเดต train_ai ให้ส่งต่อ green_ratio_min

def train_ai(episodes, grid, green_ratio_min=0.1):
    best_grid = None
    best_score = float('-inf')
    rewards = []  # คะแนนทุกรอบ
    top_layouts = []  # [(score, grid), ...]

    for episode in range(episodes):
        state = [row[:] for row in grid]

        for _ in range(GRID_ROWS * GRID_COLS):
            action = choose_action(state)
            if action is None:
                if any('0' in row for row in state):
                    continue
                else:
                    break

            r, c, char = action
            state[r][c] = char

            reward = calculate_reward_verbose(state, green_ratio_min)
            update_q_table(state, action, reward, state)

        total_reward = calculate_reward_verbose(state, green_ratio_min)
        rewards.append(total_reward)

        # 🥇 เก็บ Top 3
        top_layouts.append((total_reward, [row[:] for row in state]))
        top_layouts = sorted(top_layouts, key=lambda x: x[0], reverse=True)[:3]

        # 🎯 เก็บ Best Layout
        if total_reward > best_score:
            best_score = total_reward
            best_grid = [row[:] for row in state]

    return best_grid, best_score, rewards, top_layouts

