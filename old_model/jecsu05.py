import numpy as np
import random
import json
import os
import ast  # ✅ ใช้ ast.literal_eval แทน eval เพื่อความปลอดภัย
import glob
import pandas as pd
from collections import deque

# ตั้งค่าขนาด Grid (สามารถเปลี่ยนได้ในอนาคต)
GRID_ROWS = 3
GRID_COLS = 3
EPISODES = 10000  # จำนวนรอบการเรียนรู้
ALPHA = 0.1  # Learning Rate
GAMMA = 0.9  # Discount Factor

# ค่าคะแนนของแต่ละอาคาร
SCORES = {'E': 10, 'G': 10, 'H': 15, 'R': 5, '0': 0}  # ✅ เปลี่ยน '?' เป็น '0'

Q_TABLE_FILE = "q_table.json"

# ตำแหน่งเริ่มต้นของ E (1-based index)
E_START_POSITION = (3, 1)


# ค้นหาไฟล์ .csv ทั้งหมดในทุกโฟลเดอร์ย่อยของ goodcsv
csv_files = glob.glob(r"C:\Users\USER\Desktop\AI_Housing_Project\data\maps\CSV\goodcsv\**\*.csv", recursive=True)

# อ่านไฟล์ทั้งหมดและรวมข้อมูล
all_dataframes = [pd.read_csv(file, header=None) for file in csv_files]

# รวมข้อมูลทั้งหมดเป็น DataFrame เดียว
if all_dataframes:
    combined_data = pd.concat(all_dataframes, ignore_index=True)
    print("✅ อ่านไฟล์ CSV สำเร็จ! ข้อมูลรวมทั้งหมด:")
    print(combined_data.head())  # แสดงตัวอย่างข้อมูล
else:
    print("⚠️ ไม่พบไฟล์ CSV ในโฟลเดอร์ที่กำหนด!")

def csv_to_grid(df):
    """ แปลง DataFrame เป็น Grid สำหรับ AI """
    return df.astype(str).values.tolist()

# ตัวอย่าง: แปลงไฟล์แรกที่อ่านได้เป็น Grid
if all_dataframes:
    grid = csv_to_grid(all_dataframes[0])
    print("✅ ตัวอย่าง Grid ที่แปลงจาก CSV:")
    for row in grid:
        print(" ".join(row))


# ✅ ฟังก์ชันโหลด Q-Table อย่างปลอดภัย
def load_q_table(filepath=Q_TABLE_FILE):
    """ โหลด Q-Table จากไฟล์ JSON พร้อมจัดการ Key Format """
    if not os.path.exists(filepath):
        print("⚠️ Q-Table file not found. Creating a new Q-Table...")
        return {}  # คืนค่า Q-Table ว่าง ๆ
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print("⚠️ JSON Decode Error: The file may be corrupted. Resetting Q-Table...")
        return {}
    except Exception as e:
        print(f"⚠️ Unexpected Error while loading JSON: {e}")
        return {}

# ✅ ฟังก์ชันบันทึก Q-Table
def save_q_table(q_table, filepath=Q_TABLE_FILE):
    """ บันทึก Q-Table ลงไฟล์ JSON """
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(q_table, f, indent=4)
    except Exception as e:
        print(f"⚠️ Error while saving Q-Table: {e}")

# โหลด Q-Table
q_table = load_q_table()

# ฟังก์ชันสร้าง Grid เริ่มต้น (รองรับการย้าย E ไปขอบอัตโนมัติ)
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

def calculate_reward(grid):
    # 1️⃣ คำนวณคะแนนพื้นฐาน
    score = sum(SCORES[grid[r][c]] for r in range(GRID_ROWS) for c in range(GRID_COLS) if grid[r][c] in SCORES)

    # 2️⃣ คำนวณโบนัสแพทเทิร์น
    bonus = 0  

    # ✅ ค้นหาทุกแพทเทิร์น "HHH" และ "RRR" (แนวนอน) แบบ 3 ตัวติดกัน
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS - 2):  # ตรวจสอบ 3 ตัวติดกัน
            if grid[r][c] == 'H' and grid[r][c+1] == 'H' and grid[r][c+2] == 'H':
                bonus += 20  # HHH
            if grid[r][c] == 'R' and grid[r][c+1] == 'R' and grid[r][c+2] == 'R':
                bonus += 20  # RRR

    # ✅ ค้นหาทุกแพทเทิร์น H R H แนวตั้ง
    for c in range(GRID_COLS):
        count_hrh = sum(1 for r in range(GRID_ROWS - 2) if grid[r][c] == 'H' and grid[r+1][c] == 'R' and grid[r+2][c] == 'H')
        bonus += count_hrh * 20

    # ✅ ค้นหาทุกแพทเทิร์น HH บน RR และ RR บน HH (นับได้หลายครั้ง)
    for r in range(GRID_ROWS - 1):
        for c in range(GRID_COLS - 1):
            if (grid[r][c] == 'H' and grid[r][c+1] == 'H' and grid[r+1][c] == 'R' and grid[r+1][c+1] == 'R'):
                bonus += 20  # HH บน RR
            if (grid[r][c] == 'R' and grid[r][c+1] == 'R' and grid[r+1][c] == 'H' and grid[r+1][c+1] == 'H'):
                bonus += 20  # RR บน HH

    # 3️⃣ ตรวจสอบเงื่อนไขผิดกฎ (Penalty)
    penalty = 0

    # ❌ เช็ค H ห้ามติด E (หัก -50 ต่อตัวที่ติด E)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if grid[r][c] == 'H':
                if (r > 0 and grid[r-1][c] == 'E') or (r < GRID_ROWS-1 and grid[r+1][c] == 'E') or \
                   (c > 0 and grid[r][c-1] == 'E') or (c < GRID_COLS-1 and grid[r][c+1] == 'E'):
                    penalty -= 50  # หักคะแนนหาก H ติด E

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

    # ❌ เช็ค H ไม่ติด R (หัก -50 ต่อตัวที่ไม่ติด R)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if grid[r][c] == 'H':
                if not ((r > 0 and grid[r-1][c] == 'R') or (r < GRID_ROWS-1 and grid[r+1][c] == 'R') or \
                        (c > 0 and grid[r][c-1] == 'R') or (c < GRID_COLS-1 and grid[r][c+1] == 'R')):
                    penalty -= 50  # หัก -50 ต่อตัว H ที่ไม่ติด R

    # ❌ เช็ค R แยกกันเป็นกี่สาย แล้วหัก -100 ต่อสาย
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

    # รวมคะแนนทั้งหมด
    final_score = score + bonus + penalty + additional_bonus
    return final_score

# ฟังก์ชันเลือกการกระทำ (วางอาคารใน Grid)
def choose_action(grid):
    empty_cells = [(r, c) for r in range(GRID_ROWS) for c in range(GRID_COLS) if grid[r][c] == '0']  # ✅ เปลี่ยน '?' เป็น '0'
    if not empty_cells:
        return None  # ไม่มีที่วางแล้ว
    r, c = random.choice(empty_cells)
    char = random.choice(['H', 'R', 'G'])  # เลือกอาคารแบบสุ่ม
    return r, c, char

# ฟังก์ชันอัปเดต Q-Table
def update_q_table(state, action, reward, next_state):
    state_str = json.dumps(state)
    action_str = str(tuple(action))
    next_state_str = json.dumps(next_state)
    
    if state_str not in q_table:
        q_table[state_str] = {}
    if action_str not in q_table[state_str]:
        q_table[state_str][action_str] = 0
    
    max_future_q = max(q_table[next_state_str].values(), default=0)
    q_table[state_str][action_str] = (1 - ALPHA) * q_table[state_str][action_str] + ALPHA * (reward + GAMMA * max_future_q)

# ฝึก AI ด้วย Reinforcement Learning
best_grid = None
best_score = float('-inf')

for episode in range(EPISODES):
    grid = initialize_grid(GRID_ROWS, GRID_COLS)
    state = grid
    total_reward = 0

    reward = 0  # ✅ กำหนดค่าเริ่มต้นให้ reward เป็น 0

    for _ in range(GRID_ROWS * GRID_COLS):
        action = choose_action(grid)
        if action is None:
            reward = calculate_reward(grid)  # ✅ คำนวณคะแนนเฉพาะเมื่อกริดเต็ม
            break
        r, c, char = action
        grid[r][c] = char
        next_state = grid
        update_q_table(state, action, 0, next_state)  # ✅ ให้ reward เป็น 0 จนกว่ากริดจะเต็ม
        state = next_state

    reward = calculate_reward(grid)  # ✅ คำนวณคะแนนครั้งสุดท้าย
    total_reward = reward

    if total_reward > best_score:
        best_score = total_reward
        best_grid = [row[:] for row in grid]

    if (episode + 1) % 1000 == 0:
        print(f"Episode {episode + 1}/{EPISODES}, Reward: {total_reward}")

# บันทึก Q-Table
save_q_table(q_table)

print("🎯 AI Training Completed! Q-Table Saved.")
print(f"🏆 Best Score: {best_score}")
print("🏆 Best Grid Layout:")
for row in best_grid:
    print(" ".join(row))
