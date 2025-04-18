import numpy as np
import random
import json
import os
import ast
import glob
import pandas as pd
from collections import deque
import time

# ตั้งค่าขนาด Grid
GRID_ROWS = 4
GRID_COLS = 4
EPISODES = 1
ALPHA = 0.1
GAMMA = 0.9
SCORES = {'E': 10, 'G': 10, 'H': 15, 'R': 5, '0': 0}
Q_TABLE_FILE = "q_table.json"
E_START_POSITION = (1, 1)

# กำหนดราคาบ้านแต่ละแบบ (ต้นทุน, ราคาขาย, ขนาดตร.ม., market weight)
HOUSE_PRICES = {
    'H1': {'cost': 2_000_000, 'sale': 2_800_000, 'size': 120, 'weight': 1.0},
    'H2': {'cost': 1_800_000, 'sale': 2_500_000, 'size': 100, 'weight': 1.0},
    'H3': {'cost': 2_200_000, 'sale': 2_900_000, 'size': 140, 'weight': 1.1},
    'H4': {'cost': 2_500_000, 'sale': 3_200_000, 'size': 160, 'weight': 0.9},
}

# ✅ โหลดหรือสร้าง Grid จาก CSV ครั้งเดียว
csv_folder = "data/maps/CSV/goodcsv"
csv_files = glob.glob(f"{csv_folder}/**/*.csv", recursive=True)

# ✅ ฟังก์ชันการให้คะแนน

def optimize_ratios():
    ratios = {}
    total_score = sum((p['sale'] - p['cost']) * p['weight'] for p in HOUSE_PRICES.values())
    for htype, data in HOUSE_PRICES.items():
        score = (data['sale'] - data['cost']) * data['weight']
        ratios[htype] = score / total_score
    return ratios

H_TYPE_RATIOS = optimize_ratios()

def calculate_reward_verbose(grid):
    # ✅ กรณี grid เป็น None หรือว่าง
    if grid is None or len(grid) == 0 or len(grid[0]) == 0:
        print(f"⚠️ Error: grid เป็น None หรือว่าง! กำลังใช้ Grid เปล่าแทน...")
        grid_size = 5
        grid = np.full((grid_size, grid_size), '0')  # ใช้ NumPy แทน List

    grid = np.array(grid)
    rows, cols = grid.shape

    # ✅ คำนวณคะแนนพื้นฐาน
    base_score = sum(SCORES.get(cell, 0) for row in grid for cell in row)

    # ✅ ตรวจสอบโบนัสจาก Pattern
    bonus = 0
    bonus += np.sum((grid[:, :-2] == 'H') & (grid[:, 1:-1] == 'H') & (grid[:, 2:] == 'H')) * 100
    bonus += np.sum((grid[:, :-2] == 'R') & (grid[:, 1:-1] == 'R') & (grid[:, 2:] == 'R')) * 100
    bonus += np.sum((grid[:-2, :] == 'H') & (grid[1:-1, :] == 'R') & (grid[2:, :] == 'H')) * 100
    bonus += np.sum((grid[:-1, :-1] == 'H') & (grid[:-1, 1:] == 'H') &
                    (grid[1:, :-1] == 'R') & (grid[1:, 1:] == 'R')) * 100
    bonus += np.sum((grid[:-1, :-1] == 'R') & (grid[:-1, 1:] == 'R') &
                    (grid[1:, :-1] == 'H') & (grid[1:, 1:] == 'H')) * 100

    # ✅ ค่าปรับเกี่ยวกับ E และ R
    penalty = 0
    h_positions = np.argwhere(grid == 'H')
    e_positions = np.argwhere(grid == 'E')

    # ✅ 1) ตรวจสอบว่ามี H ที่ติดกับ E หรือไม่
    #if len(h_positions) > 0 and len(e_positions) > 0:
        #h_neighbors_e = np.sum([
            #np.roll(grid == 'E', shift, axis=axis)[h_positions[:, 0], h_positions[:, 1]]
            #for shift, axis in [(1, 0), (-1, 0), (1, 1), (-1, 1)]
        #], axis=0)
        #penalty -= 100 * np.sum(h_neighbors_e)  # ✅ ปรับค่าให้ลงโทษตามจำนวน H ที่ติด E

    # ✅ 2) ตรวจสอบว่ามี E ที่ไม่ได้ติด R หรือไม่
    if len(e_positions) > 0:
        e_neighbors_r = np.any([
            np.roll(grid == 'R', shift, axis=axis)[e_positions[:, 0], e_positions[:, 1]]
            for shift, axis in [(1, 0), (-1, 0), (1, 1), (-1, 1)]
        ], axis=0)

        if not np.any(e_neighbors_r):
            penalty -= 200

    # ✅ 3) ตรวจสอบจำนวนกลุ่ม R (ถนน)
    r_clusters = count_r_clusters(grid) if np.any(grid == 'R') else 0
    if r_clusters > 1:
        penalty -= 500 * (r_clusters - 1)

    # ✅ 4) ตรวจสอบว่า H ทุกหลังติด R หรือไม่
    if len(h_positions) > 0:
        h_neighbors_r = np.any([
            np.roll(grid == 'R', shift, axis=axis)[h_positions[:, 0], h_positions[:, 1]]
            for shift, axis in [(1, 0), (-1, 0), (1, 1), (-1, 1)]
        ], axis=0)

        penalty -= 200 * np.sum(~h_neighbors_r)  # ลงโทษบ้านที่ไม่ได้ติดถนน

        if np.all(h_neighbors_r):  # ✅ ถ้าบ้านทุกหลังติดถนน ให้โบนัส
            bonus += 100

    # ✅ ตรวจสอบสัดส่วนพื้นที่สีเขียว (G)
    total_cells = rows * cols
    num_green = np.sum(grid == 'G')
    green_ratio = num_green / total_cells

    if green_ratio < 0.1:
        penalty -= 500  # ✅ ปรับลดคะแนนหากพื้นที่สีเขียวน้อยเกินไป

    # ✅ ตรวจสอบว่ามีถนนหรือไม่
    roads_exist = np.sum(grid == 'R') > 0
    if not roads_exist:
        penalty -= 500  # ลงโทษหนักถ้าไม่มีถนน

    # ✅ คำนวณคะแนนรวม
    total_score = base_score + bonus + penalty
    #print(f"🎯 Debug: คะแนน Grid = {total_score} (Base: {base_score}, Bonus: {bonus}, Penalty: {penalty})")

    return total_score

def count_r_clusters(grid, use_dfs=False):
    """ ✅ นับจำนวนกลุ่มของ 'R' ที่แยกกัน พร้อมเลือกโหมด BFS หรือ DFS """
    GRID_ROWS, GRID_COLS = len(grid), len(grid[0])  # ✅ ดึงขนาดของ Grid อัตโนมัติ
    visited = [[False] * GRID_COLS for _ in range(GRID_ROWS)]  # ✅ ใช้ List แทน Set
    clusters = 0

    def bfs(r, c):
        """ ✅ BFS สำหรับหา Cluster """
        queue = deque([(r, c)])
        visited[r][c] = True
        while queue:
            x, y = queue.popleft()
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:  # 4 ทิศทาง
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_ROWS and 0 <= ny < GRID_COLS and not visited[nx][ny] and grid[nx][ny] == 'R':
                    visited[nx][ny] = True
                    queue.append((nx, ny))

    def dfs(r, c):
        """ ✅ DFS สำหรับหา Cluster """
        stack = [(r, c)]
        visited[r][c] = True
        while stack:
            x, y = stack.pop()
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_ROWS and 0 <= ny < GRID_COLS and not visited[nx][ny] and grid[nx][ny] == 'R':
                    visited[nx][ny] = True
                    stack.append((nx, ny))

    search_func = dfs if use_dfs else bfs  # ✅ เลือก BFS หรือ DFS

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if grid[r][c] == 'R' and not visited[r][c]:
                clusters += 1
                search_func(r, c)  # ✅ เรียกใช้ BFS หรือ DFS

    return clusters

def apply_house_types(grid):
    # 1. หาเฉพาะตำแหน่งที่เป็น 'H' (ตามลำดับบนซ้ายไปล่างขวา)
    h_positions = [(r, c) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] == 'H']
    total_h = len(h_positions)
    if total_h == 0:
        return grid

    # 2. คำนวณจำนวนบ้านแต่ละแบบจาก H_TYPE_RATIOS
    house_types_count = {htype: int(ratio * total_h) for htype, ratio in H_TYPE_RATIOS.items()}

    # 3. ตรวจสอบว่าจำนวนรวมครบหรือไม่
    current_total = sum(house_types_count.values())
    while current_total < total_h:
        for htype in house_types_count:
            house_types_count[htype] += 1
            current_total += 1
            if current_total == total_h:
                break

    # 4. สร้างลำดับชื่อบ้านทั้งหมด เช่น ['H1', 'H1', 'H2', 'H3', ...]
    house_sequence = []
    for htype, count in house_types_count.items():
        house_sequence.extend([htype] * count)

    # 5. วาง H1–H4 ทีละตำแหน่งตามลำดับ
    for (r, c), htype in zip(h_positions, house_sequence):
        grid[r][c] = htype

    return grid

# ✅ ฟังก์ชันสรา้างกริด และวาง E จุดแรก

def initialize_grid(rows, cols, e_position):
    """ ✅ สร้าง Grid ว่าง และเลื่อน `E` ไปขอบที่ใกล้ที่สุด (ใช้ 1-based index ตลอด) """
    print(f"📌 กำลังสร้าง Grid ขนาด {rows}x{cols} และวาง E ที่ {e_position} (1-based index)")

    grid = [['0' for _ in range(cols)] for _ in range(rows)]  # ✅ สร้าง Grid ว่าง
    r, c = e_position  # ✅ ใช้ค่าพิกัดที่รับมาเป็น 1-based index โดยตรง

    # ✅ ตรวจสอบว่า `E` อยู่ที่ขอบแล้วหรือไม่
    if r == 1 or r == rows or c == 1 or c == cols:
        grid[r-1][c-1] = 'E'  # ✅ ถ้าอยู่ขอบแล้ว ใช้ตำแหน่งเดิม
    else:
        print(f"⚠️ E ไม่อยู่ที่ขอบ กำลังเลื่อนไปขอบที่ใกล้ที่สุด ({e_position}) ...")

        # ✅ ตรวจสอบระยะทางไปแต่ละขอบ (Top, Bottom, Left, Right)
        distances = {
            "top": r - 1,
            "bottom": rows - r,
            "left": c - 1,
            "right": cols - c
        }

        # ✅ เรียงลำดับขอบที่ใกล้ที่สุดจากน้อยไปมาก
        sorted_edges = sorted(distances.items(), key=lambda x: x[1])

        # ✅ เลือกขอบที่ใกล้ที่สุด
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

        grid[r-1][c-1] = 'E'  # ✅ วาง `E` ในตำแหน่งที่ถูกต้อง

    print(f"✅ ตำแหน่งใหม่ของ E: {r}, {c} (1-based index)")
    return grid, (r, c)  # ✅ คืน Grid และตำแหน่งใหม่ของ E (1-based)

def load_or_initialize_grid(csv_folder, rows, cols, e_position):
    """
    ✅ โหลด Grid จาก CSV โดยใช้ตำแหน่ง E ใหม่
    ✅ ถ้าไม่มีไฟล์ CSV ที่ตรงกัน ให้สร้าง Grid ใหม่
    """
    print(f"📌 ค้นหาไฟล์ CSV {rows}x{cols} ที่มีตำแหน่ง E: {e_position} (1-based index)")

    csv_files = glob.glob(f"{csv_folder}/**/*.csv", recursive=True)

    if not csv_files:
        print("⚠️ ไม่พบไฟล์ CSV! ใช้ Grid ที่สร้างขึ้นแทน...")
        return initialize_grid(rows, cols, e_position)  # ✅ ใช้ฟังก์ชันใหม่

    best_grid = None
    best_score = float('-inf')

    for file in csv_files:
        df = pd.read_csv(file, header=None)

        # ✅ ตรวจสอบขนาด Grid
        if df.shape != (rows, cols):
            print(f"⚠️ ข้ามไฟล์ {file} เพราะขนาดไม่ตรง {df.shape}")
            continue

        grid_from_csv = df.astype(str).values.tolist()
        e_found = [(r+1, c+1) for r in range(len(grid_from_csv)) for c in range(len(grid_from_csv[0])) if grid_from_csv[r][c] == 'E']

        if not e_found or e_found[0] != e_position:
            print(f"⚠️ ข้ามไฟล์ {file} เพราะตำแหน่ง `E` ไม่ตรง {e_found} (คาดหวัง: {e_position})")
            continue

        score = calculate_reward_verbose(grid_from_csv)  # ✅ คำนวณคะแนน

        if score > best_score:
            best_score = score
            best_grid = grid_from_csv

    if best_grid is not None:
        print(f"🏆 ใช้ไฟล์ที่ดีที่สุด: คะแนน {best_score}")
        return best_grid, e_position  # ✅ คืนค่าเป็น 1-based index

    print("⚠️ ไม่มีไฟล์ที่เหมาะสม! ใช้ Grid ที่สร้างขึ้นแทน...")
    return initialize_grid(rows, cols, e_position)  # ✅ ใช้ฟังก์ชันใหม่

# ✅ ฟังก์ชันเลือก Action แบบเป่ายิงฉุบ (เลือกแบบการเดิน)

def choose_action(grid):
    if not grid or not isinstance(grid, list):
        print("⚠️ ERROR: Grid ไม่ถูกต้อง (Empty or Invalid Format)")
        return None

    rows = len(grid)
    cols = max(len(row) for row in grid)

    empty_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == '0']

    if not empty_cells:
        #print("⚠️ ไม่มีที่ว่างให้วางอาคาร!")
        return None  # ไม่มีตำแหน่งที่วางได้

    r, c = random.choice(empty_cells)
    char = random.choice(['H', 'R', 'G'])
    return r, c, char

# ✅ ฟังก์ชันอัปเดต Q-Table

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

# ✅ ฟังก์ชัน Train AI (ปรับใหม่ให้ใช้ขนาด grid จริง)
def train_ai(episodes, grid):
    """
    ฝึก AI ด้วย Reinforcement Learning (Q-Learning) พร้อมบันทึกขั้นตอนสำหรับ debug
    """
    best_grid = None
    best_score = float('-inf')
    action_log = []

    for episode in range(episodes):
        state = [row[:] for row in grid]  # ใช้ Grid เดิมทุก Episode
        rows = len(state)
        cols = len(state[0]) if state else 0
        max_steps = rows * cols

        for step in range(max_steps):
            action = choose_action(state)

            if action is None:
                break

            r, c, char = action
            prev = state[r][c]
            state[r][c] = char
            reward = calculate_reward_verbose(state)
            update_q_table(state, action, reward, state)

            action_log.append(f"EP:{episode+1} STEP:{step+1} ➤ วาง '{char}' ที่ ({r+1},{c+1}) [จาก '{prev}'] → Reward: {reward}")

        total_reward = calculate_reward_verbose(state)
        if total_reward > best_score:
            best_score = total_reward
            best_grid = [row[:] for row in state]

        if (episode + 1) % 10000 == 0:
            print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}")

    return best_grid, best_score, action_log

# ✅ ฟังก์ชันจับเวลาการรัน

def measure_execution_time(function, *args, **kwargs):
    """
    ฟังก์ชันวัดเวลาการรันของฟังก์ชันที่กำหนด
    คืนค่า: (ผลลัพธ์จากฟังก์ชัน, เวลาที่ใช้)
    """
    start_time = time.time()  # ⏳ เริ่มจับเวลา
    result = function(*args, **kwargs)  # ✅ เรียกใช้ฟังก์ชันที่ต้องการวัดเวลา
    end_time = time.time()  # ⏳ หยุดจับเวลา

    elapsed_time = end_time - start_time  # คำนวณเวลาที่ใช้
    elapsed_minutes = elapsed_time / 60  # แปลงเป็นนาที

    print(f"\n⏳ ใช้เวลาทั้งหมด: {elapsed_time:.2f} วินาที ({elapsed_minutes:.2f} นาที)")

    return *result, elapsed_time  # ✅ ใช้ `*result` เพื่อแยกค่าที่คืนจากฟังก์ชันหลัก

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

    total_units = sum(summary.values())
    for htype, count in summary.items():
        if count:
            info = HOUSE_PRICES[htype]
            ratio_percent = (count / total_units) * 100
            print(f"🏠 {htype}: {count} หลัง | {ratio_percent:.1f}% | ต้นทุนต่อหลัง: {info['cost']:,} บาท | รายได้ต่อหลัง: {info['sale']:,} บาท | รายได้รวม: {info['sale'] * count:,} บาท")

    print(f"\n💸 ต้นทุนรวมทั้งหมด: {total_cost:,} บาท")
    print(f"💰 รายได้รวมทั้งหมด: {total_sale:,} บาท")
    print(f"📈 กำไรรวมทั้งหมด: {total_profit:,} บาท")
    print(f"📐 กำไรเฉลี่ยต่อ ตร.ม.: {avg_profit_per_sqm:,.2f} บาท/ตร.ม.")
    print(f"🎯 กำไรปรับตาม market weight: {weighted_profit:,.2f} บาท")

# ✅ ฝึก AI และบันทึกผลลัพธ์
q_table = {}
grid, new_e_position = initialize_grid(GRID_ROWS, GRID_COLS, E_START_POSITION)
grid, _ = load_or_initialize_grid(csv_folder, GRID_ROWS, GRID_COLS, new_e_position)

print(f"✅ ขนาดของ Grid หลังโหลด: {len(grid)} rows x {len(grid[0]) if grid else 0} cols | ตำแหน่ง E: {new_e_position}")

best_grid, best_score, action_log = train_ai(EPISODES, grid)
final_layout = apply_house_types([row[:] for row in best_grid])

print(f"\n🏆 Best Layout Found:")
for row in best_grid:
    print(" ".join(row))
print(f"\n✅ Best Score Achieved: {best_score}")

print("\n📌 Final Layout with H1–H4:")
for row in final_layout:
    print(" ".join(row))

analyze_profit(final_layout)

print("\n📜 ACTION LOG (AI Placement):")
for log in action_log[-20:]:  # แสดงท้าย ๆ พอประมาณ
    print(log)
