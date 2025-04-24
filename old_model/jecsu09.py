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
GRID_ROWS = 3
GRID_COLS = 3
EPISODES = 100
ALPHA = 0.1  
GAMMA = 0.9  
SCORES = {'E': 10, 'G': 10, 'H': 15, 'R': 5, '0': 0}
Q_TABLE_FILE = "q_table.json"
E_START_POSITION = (2, 1)

# ✅ โหลดหรือสร้าง Grid จาก CSV ครั้งเดียว
csv_folder = "data/maps/CSV/goodcsv"
csv_files = glob.glob(f"{csv_folder}/**/*.csv", recursive=True)

def count_r_clusters(grid):
    """ นับจำนวนกลุ่มของ 'R' ที่แยกกัน """
    visited = set()
    clusters = 0

    def bfs(r, c):
        queue = deque([(r, c)])
        visited.add((r, c))
        while queue:
            x, y = queue.popleft()
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:  # 4 ทิศทาง
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_ROWS and 0 <= ny < GRID_COLS and (nx, ny) not in visited and grid[nx][ny] == 'R':
                    visited.add((nx, ny))
                    queue.append((nx, ny))

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if grid[r][c] == 'R' and (r, c) not in visited:
                clusters += 1
                bfs(r, c)
    
    return clusters

def calculate_reward_verbose(grid):
    total_reward = 0  # ✅ กำหนดค่าเริ่มต้น

    # ✅ ค้นหาตำแหน่งของ E ที่แท้จริงใน Grid
    e_row, e_col = None, None
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if grid[r][c] == 'E':
                e_row, e_col = r + 1, c + 1  # บันทึกตำแหน่งจริง (1-based index)
                break
        if e_row is not None:
            break  # หยุดค้นหาเมื่อเจอ E ตัวแรก

    # ❌ ถ้าไม่พบ E ใน Grid ให้แจ้งเตือน และใช้ค่าเริ่มต้น
    if e_row is None or e_col is None:
        print("⚠️ WARNING: ไม่พบ E ใน Grid! ใช้ค่าเริ่มต้น (1,1)")
        e_row, e_col = 1, 1

    # 1️⃣ คำนวณคะแนนพื้นฐาน
    base_score = sum(SCORES[grid[r][c]] for r in range(GRID_ROWS) for c in range(GRID_COLS) if grid[r][c] in SCORES)

    # 2️⃣ คำนวณโบนัสแพทเทิร์น
    bonus = 0  

    # ✅ ค้นหาทุกแพทเทิร์น "HHH" และ "RRR" (แนวนอน)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS - 2):
            if grid[r][c] == 'H' and grid[r][c+1] == 'H' and grid[r][c+2] == 'H':
                bonus += 20
            if grid[r][c] == 'R' and grid[r][c+1] == 'R' and grid[r][c+2] == 'R':
                bonus += 20

    # ✅ ค้นหาแพทเทิร์น "H R H" แนวตั้ง
    for c in range(GRID_COLS):
        for r in range(GRID_ROWS - 2):
            if grid[r][c] == 'H' and grid[r+1][c] == 'R' and grid[r+2][c] == 'H':
                bonus += 20

    # ✅ ค้นหาแพทเทิร์น HH บน RR และ RR บน HH
    for r in range(GRID_ROWS - 1):
        for c in range(GRID_COLS - 1):
            if (grid[r][c] == 'H' and grid[r][c+1] == 'H' and grid[r+1][c] == 'R' and grid[r+1][c+1] == 'R'):
                bonus += 20
            if (grid[r][c] == 'R' and grid[r][c+1] == 'R' and grid[r+1][c] == 'H' and grid[r+1][c+1] == 'H'):
                bonus += 20

    # 3️⃣ ตรวจสอบเงื่อนไขผิดกฎ (Penalty)
    penalty = 0

    # ❌ เช็ค H ห้ามติด E (หัก -50 ต่อตัวที่ติด E)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if grid[r][c] == 'H':
                if (r > 0 and grid[r-1][c] == 'E') or (r < GRID_ROWS-1 and grid[r+1][c] == 'E') or \
                   (c > 0 and grid[r][c-1] == 'E') or (c < GRID_COLS-1 and grid[r][c+1] == 'E'):
                    penalty -= 50
    
    # ❌ เช็ค E ไม่ติด R เลย = -100
    e_has_r_neighbor = False
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if grid[r][c] == 'E':
                if (r > 0 and grid[r-1][c] == 'R') or (r < GRID_ROWS-1 and grid[r+1][c] == 'R') or \
                   (c > 0 and grid[r][c-1] == 'R') or (c < GRID_COLS-1 and grid[r][c+1] == 'R'):
                    e_has_r_neighbor = True
    if not e_has_r_neighbor:
        penalty -= 100

    # ✅ ตรวจสอบ H ที่ไม่ติด R (หัก -100 ต่อตัวที่ไม่ติด R)
    h_not_touching_r = 0  # ตัวนับจำนวน H ที่ไม่ติด R
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if grid[r][c] == 'H':
                # ✅ ตรวจสอบว่า H มี R ติดกันหรือไม่
                has_r_neighbor = (
                    (r > 0 and grid[r-1][c] == 'R') or  # บน
                    (r < GRID_ROWS-1 and grid[r+1][c] == 'R') or  # ล่าง
                    (c > 0 and grid[r][c-1] == 'R') or  # ซ้าย
                    (c < GRID_COLS-1 and grid[r][c+1] == 'R')  # ขวา
                )
                if not has_r_neighbor:  # ❌ ไม่มี R ติดเลย
                    h_not_touching_r += 1
                    penalty -= 100  # หัก -100 ต่อ H ที่ไม่ติด R

    # ✅ ตรวจสอบว่ามีกลุ่ม `R` แยกกันหรือไม่
    r_clusters = count_r_clusters(grid)
    if r_clusters > 1:
        penalty -= 100 * (r_clusters - 1)  

    # ✅ เพิ่มโบนัส G และ R ติด E
    additional_bonus = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if grid[r][c] == 'G':
                if (r > 0 and grid[r-1][c] == 'E') or (r < GRID_ROWS-1 and grid[r+1][c] == 'E') or \
                (c > 0 and grid[r][c-1] == 'E') or (c < GRID_COLS-1 and grid[r][c+1] == 'E'):
                    additional_bonus += 10
            if grid[r][c] == 'R':
                if (r > 0 and grid[r-1][c] == 'E') or (r < GRID_ROWS-1 and grid[r+1][c] == 'E') or \
                (c > 0 and grid[r][c-1] == 'E') or (c < GRID_COLS-1 and grid[r][c+1] == 'E'):
                    additional_bonus += 5
              
    if all(cell != '0' for row in grid for cell in row):
        total_reward += 50  # โบนัสพิเศษถ้า Grid เต็ม

    # รวมคะแนนทั้งหมด
    final_score = base_score + bonus + penalty + additional_bonus

    return final_score

def load_or_initialize_grid(csv_folder, rows, cols, e_start_position):
    """
    โหลด Grid จากไฟล์ CSV ที่มีขนาดตรงกัน, `E` ตรงกัน และเลือก Grid ที่มีคะแนนดีที่สุด
    ถ้าไม่พบไฟล์ที่ใช้ได้ จะสร้าง Grid ขนาดที่กำหนดขึ้นมาใหม่
    """
    print("📌 กำลังค้นหาไฟล์ CSV...")
    csv_files = glob.glob(f"{csv_folder}/**/*.csv", recursive=True)

    if not csv_files:
        print("⚠️ ไม่พบไฟล์ CSV เลย! กำลังสร้าง Grid เริ่มต้นแทน...")
        return initialize_grid(rows, cols, e_start_position)

    best_grid = None
    best_score = float('-inf')

    for file in csv_files:
        df = pd.read_csv(file, header=None)

        # ตรวจสอบขนาดของ Grid
        if df.shape != (rows, cols):
            print(f"⚠️ ข้ามไฟล์ {file} เพราะขนาดไม่ตรง ({df.shape})")
            continue  

        grid = df.astype(str).values.tolist()

        # ค้นหาตำแหน่ง `E` ใน Grid ที่โหลดมา
        e_found = [(r, c) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] == 'E']

        # ถ้าพบ `E` แต่ตำแหน่งไม่ตรงกันให้ข้าม
        if not e_found or e_found[0] != (e_start_position[0] - 1, e_start_position[1] - 1):
            print(f"⚠️ ข้ามไฟล์ {file} เพราะตำแหน่ง `E` ไม่ตรง ({e_found})")
            continue  

        # คำนวณคะแนนของ Grid นี้
        score = calculate_reward_verbose(grid)

        print(f"✅ ตรวจสอบไฟล์ {file} | ขนาด: {df.shape} | ตำแหน่ง E: {e_found} | คะแนน: {score}")

        # เลือก Grid ที่มีคะแนนดีที่สุด
        if score > best_score:
            best_score = score
            best_grid = grid  

    if best_grid is not None:
        print(f"🏆 ใช้ไฟล์ที่ดีที่สุดจาก CSV ด้วยคะแนน: {best_score}")
        return best_grid
    
    # ถ้าไม่มี Grid ที่เหมาะสมในไฟล์ CSV -> สร้างใหม่
    print("⚠️ ไม่พบ Grid ที่มีขนาดและตำแหน่ง E ตรงกัน! กำลังสร้าง Grid เริ่มต้นแทน...")
    return initialize_grid(rows, cols, e_start_position)

# ✅ ฟังก์ชันสร้าง Grid เริ่มต้น
def initialize_grid(rows, cols, e_position=E_START_POSITION):
    """ สร้าง Grid ขนาดกำหนด และวาง E ตามตำแหน่งที่ผู้ใช้ต้องการ (1-based index) """
    grid = [['0' for _ in range(cols)] for _ in range(rows)]  # สร้าง Grid ว่าง

    # แปลงจาก 1-based index เป็น 0-based index
    r, c = e_position
    r, c = r - 1, c - 1  # ✅ แปลงตำแหน่งให้ตรงกับ index ใน Python

    # ตรวจสอบว่า E อยู่ที่ขอบหรือไม่
    if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
        grid[r][c] = 'E'  # ✅ อยู่ขอบแล้ว ใช้ตำแหน่งเดิม
    else:
        print("⚠️ E ไม่อยู่ที่ขอบ กำลังเลือกตำแหน่งที่ใกล้ขอบที่สุด...")

        # กรณีอยู่ระหว่างขอบและกึ่งกลาง (อยู่ในส่วนในของ Grid แต่ไม่ตรงกลางเป๊ะ)
        candidate_positions = []
        if r < rows - 1:
            candidate_positions.append((rows - 1, c))  # ขอบล่างของคอลัมน์เดิม
        if c < cols - 1:
            candidate_positions.append((r, cols - 1))  # ขอบขวาของแถวเดิม

        # เลือกตำแหน่งใหม่แบบสุ่มจากขอบที่ใกล้ที่สุด
        if candidate_positions:
            r, c = random.choice(candidate_positions)
        
        grid[r][c] = 'E'

    return grid

# ✅ โหลด Grid ครั้งเดียวและใช้ตลอดการรัน Episode
grid = load_or_initialize_grid(csv_folder, GRID_ROWS, GRID_COLS, E_START_POSITION)
print(f"✅ ใช้ Grid ที่โหลดมา: {len(grid)} rows x {max(len(row) for row in grid)} cols")

# ✅ กำหนดขนาดใหม่ให้ตรงกับข้อมูลจริง
GRID_ROWS = len(grid)
GRID_COLS = max(len(row) for row in grid) if grid else 0
print(f"✅ ขนาด Grid ที่ใช้: {GRID_ROWS} rows x {GRID_COLS} cols")

# ✅ ฟังก์ชันเลือก Action แบบเป่ายิงฉุบ
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

# ✅ ฟังก์ชัน Train AI
def train_ai(episodes, grid):
    """
    ฝึก AI ด้วย Reinforcement Learning (Q-Learning)
    """
    best_grid = None
    best_score = float('-inf')

    for episode in range(episodes):
        state = [row[:] for row in grid]  # ใช้ Grid เดิมทุก Episode
        total_reward = 0

        for _ in range(GRID_ROWS * GRID_COLS):
            action = choose_action(state)

            if action is None:
                if any('0' in row for row in state):  # ถ้ายังมี 0 ห้ามหยุด
                    continue  
                else:
                    break  

            r, c, char = action  
            state[r][c] = char  

            reward = calculate_reward_verbose(state)
            update_q_table(state, action, reward, state)

        total_reward = calculate_reward_verbose(state)

        if total_reward > best_score:
            best_score = total_reward
            best_grid = [row[:] for row in state]

        if (episode + 1) % 10000 == 0:
            print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}")

    return best_grid, best_score

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

# ✅ ฝึก AI และบันทึกผลลัพธ์
q_table = {}

print(f"📂 CSV Files Found: {csv_files}")
print(f"📂 Searching CSV in: {csv_folder}")
print(f"🔍 ตรวจสอบขนาดของ Grid ก่อนเลือก Action: {len(grid)} rows x {max(len(row) for row in grid)} cols")
print(f"✅ ขนาด Grid ที่โหลดมา: {len(grid)} rows x {max(len(row) for row in grid)} cols")

best_grid, best_score, execution_time = measure_execution_time(train_ai, EPISODES, grid)

print("\n🏆 Best Layout Found:")
for row in best_grid:
    print(" ".join(row))
print(f"\n✅ Best Score Achieved: {best_score}")
