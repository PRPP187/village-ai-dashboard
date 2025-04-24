import numpy as np
import random
import json
import os
import ast
import glob
import pandas as pd
from collections import deque
import time
import threading
import shutil
from filelock import FileLock
import requests


# ตั้งค่าขนาด Grid
GRID_ROWS = 5
GRID_COLS = 5
EPISODES = 10000
ALPHA = 0.1  
GAMMA = 0.9  
SCORES = {'E': 10, 'G': 10, 'H': 15, 'R': 5, '0': 0}
E_START_POSITION = (4, 4)

Q_TABLE_FILE = "q_table.json"
LOCK_FILE = Q_TABLE_FILE + ".lock"

# ✅ โหลดหรือสร้าง Grid จาก CSV ครั้งเดียว
csv_folder = "data/maps/CSV/goodcsv"
csv_files = glob.glob(f"{csv_folder}/**/*.csv", recursive=True)

# ✅ ใช้ Lock ป้องกันไฟล์เสียหายจากการเขียนพร้อมกัน
save_lock = threading.Lock()

# ✅ ฟังก์ชันการให้คะแนน

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

    base_score = sum(SCORES[grid[r][c]] for r in range(GRID_ROWS) for c in range(GRID_COLS) if grid[r][c] in SCORES)

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

    # ✅ เพิ่มเงื่อนไข: ถ้า G มากเกินไป หักคะแนน
    g_count = sum(row.count('G') for row in grid)
    if g_count > (GRID_ROWS * GRID_COLS) * 0.3:  # ถ้า G มากกว่า 30% ของ Grid
        penalty -= 100  # หักคะแนน

    additional_bonus = 0

    # ✅ เพิ่มโบนัส G และ R ติด E
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
        additional_bonus += 50  # โบนัสพิเศษถ้า Grid เต็ม

    # รวมคะแนนทั้งหมด
    total_reward = base_score + bonus + penalty + additional_bonus

    return total_reward

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

# ✅ ฟังก์ชัน Q-Table

def load_q_table():
    """ ✅ โหลด Q-Table จากไฟล์ JSON และแก้ไขปัญหาข้อมูลเสียหาย """
    try:
        with open(Q_TABLE_FILE, "r") as f:
            data = json.load(f)

        # ✅ ตรวจสอบว่าข้อมูลสามารถใช้ได้หรือไม่
        q_table = {}
        for k, v in data.items():
            try:
                key = ast.literal_eval(k)  # ✅ แปลง key ให้เป็น tuple
                q_table[key] = v
            except (SyntaxError, ValueError):
                print(f"⚠️ Warning: ไม่สามารถแปลง Key {k} ได้ ข้ามไป")

        print(f"✅ Q-Table Loaded: {len(q_table)} states")  # ✅ แจ้งจำนวน states ใน Q-Table
        return q_table

    except (FileNotFoundError, json.JSONDecodeError):
        print("⚠️ Warning: ไม่พบไฟล์ Q-Table หรือไฟล์เสียหาย สร้างใหม่")
        return {}  # ✅ ถ้าไฟล์ไม่มี หรือเสียหาย ให้คืนค่าเป็น Q-Table ว่าง

def send_q_table_to_server():
    url = "http://127.0.0.1:5000/update_q_table"

    # ✅ ตรวจสอบว่า Q-Table มีข้อมูลก่อนส่ง
    if not q_table:
        print("⚠️ Warning: Q-Table ว่างเปล่า ไม่ส่งไปเซิร์ฟเวอร์")
        return

    try:
        response = requests.post(url, json=q_table)
        response.raise_for_status()
        print(f"✅ Q-Table Sent: {len(q_table)} states | Server Response: {response.json()}")  # ✅ แจ้งจำนวน states ที่ส่งไป

        time.sleep(0.1)  # ✅ เพิ่มหน่วงเวลาป้องกันการส่งถี่เกินไป

    except requests.exceptions.RequestException as e:
        print(f"⚠️ ERROR: ไม่สามารถส่ง Q-Table ไปเซิร์ฟเวอร์: {e}")

def update_q_table(state, action, reward, next_state):
    """
    ✅ อัปเดตค่า Q-Table ตาม Q-Learning Algorithm
    ✅ บันทึก Q-Table ลงไฟล์ทุกครั้งที่อัปเดต
    """
    state_key = str(tuple(tuple(row) for row in state))
    next_state_key = str(tuple(tuple(row) for row in next_state))
    action_key = str(tuple(action))

    if state_key not in q_table:
        q_table[state_key] = {}
    if action_key not in q_table[state_key]:
        q_table[state_key][action_key] = 0  

    max_future_q = max(q_table.get(next_state_key, {}).values(), default=0)
    q_table[state_key][action_key] = (1 - ALPHA) * q_table[state_key][action_key] + ALPHA * (reward + GAMMA * max_future_q)

    send_q_table_to_server()  # ✅ ส่งค่าไปเซิร์ฟเวอร์ทุกครั้งที่อัปเดต

# ✅ ฟังก์ชันเลือก Action แบบเป่ายิงฉุบ (เลือกแบบการเดิน)

def choose_action(grid, step):
    rows = len(grid)
    cols = len(grid[0])

    e_position = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 'E'][0]
    e_row, e_col = e_position

    empty_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == '0']
    empty_cells.sort(key=lambda pos: abs(pos[0] - e_row) + abs(pos[1] - e_col))

    best_options = []

    for r, c in empty_cells:
        for char in ['H', 'R', 'G']:
            grid[r][c] = char
            score = calculate_reward_verbose(grid)
            grid[r][c] = '0'

            best_options.append((score, (r, c), char))

    best_options.sort(reverse=True, key=lambda x: x[0])  # เรียงคะแนนจากมากไปน้อย
    top_choices = best_options[:3]  # ✅ เลือก Top 3 ตัวเลือกที่ดีที่สุด

    if top_choices:
        chosen = random.choice(top_choices)  # ✅ สุ่มจากตัวเลือกที่ดีที่สุด 3 อันดับแรก
        r, c = chosen[1]
        char = chosen[2]
        grid[r][c] = char
        #print(f"Step {step}: วาง '{char}' ที่ตำแหน่ง ({r+1}, {c+1}), ได้คะแนน {chosen[0]}")

        #for row in grid:
            #print(" ".join(row))
        #print("\n" + "="*20 + "\n")

        return r, c, char
    else:
        #print(f"Step {step}: ❌ ไม่สามารถหาตำแหน่งที่วางได้")
        return None

# ✅ ฟังก์ชัน Train AI

def train_ai(episodes, grid):
    best_grid = None
    best_score = float('-inf')

    for episode in range(episodes):
        state = [row[:] for row in grid]
        total_reward = 0
        step = 0  # ✅ เริ่มต้น step ที่ 0

        for _ in range(GRID_ROWS * GRID_COLS):
            step += 1  # ✅ เพิ่ม step ทีละ 1
            action = choose_action(state, step)

            if action is None:
                if any('0' in row for row in state):  
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

        if (episode + 1) % 1 == 0:
            print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}")

    send_q_table_to_server()

    return best_grid, best_score

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

# ✅ ใช้ Lock ป้องกันไฟล์เสียหายจากการเขียนพร้อมกัน
save_lock = threading.Lock()

# ✅ สร้างตัวแปร Q-Table (ต้องมาก่อน load_q_table)
q_table = {}  # ต้องกำหนดตัวแปรก่อนใช้งาน

# ✅ โหลด Q-Table ตอนเริ่มต้น (ถ้ามีไฟล์อยู่)
load_q_table()

# ✅ เลื่อน `E` ก่อน
grid, new_e_position = initialize_grid(GRID_ROWS, GRID_COLS, E_START_POSITION)

# ✅ โหลด Grid จาก CSV โดยใช้ตำแหน่ง `E` ใหม่
grid, _ = load_or_initialize_grid(csv_folder, GRID_ROWS, GRID_COLS, new_e_position)

print(f"✅ ขนาดของ Grid หลังโหลด: {len(grid)} rows x {len(grid[0]) if grid else 0} cols | ตำแหน่ง E: {new_e_position}")

best_grid, best_score, execution_time = measure_execution_time(train_ai, EPISODES, grid)

#print(f"📂 CSV Files Found: {csv_files}")
#print(f"📂 Searching CSV in: {csv_folder}")

print("\n🏆 Best Layout Found:")
for row in best_grid:
    print(" ".join(row))
print(f"\n✅ Best Score Achieved: {best_score}")
