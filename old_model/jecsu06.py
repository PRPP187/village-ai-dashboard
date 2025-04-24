import numpy as np
import random
import json
import os
import ast  # ✅ ใช้ ast.literal_eval แทน eval เพื่อความปลอดภัย
import glob
import pandas as pd
from collections import deque

# ตั้งค่าขนาด Grid (สามารถเปลี่ยนได้ในอนาคต)
GRID_ROWS = 4
GRID_COLS = 4
EPISODES = 10000  # จำนวนรอบการเรียนรู้
ALPHA = 0.3  # Learning Rate
GAMMA = 0.9  # Discount Factor

# ค่าคะแนนของแต่ละอาคาร
SCORES = {'E': 10, 'G': 10, 'H': 15, 'R': 5, '0': 0}

Q_TABLE_FILE = "q_table.json"

# ตำแหน่งเริ่มต้นของ E (1-based index)
E_START_POSITION = (1, 1)

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

# ✅ ค้นหาไฟล์ CSV ในโฟลเดอร์ goodcsv
csv_files = glob.glob(r"C:\Users\USER\Desktop\AI_Housing_Project\data\maps\goodcsv\**\*.csv", recursive=True)

# ✅ โหลดไฟล์ CSV ทั้งหมด
all_dataframes = []
for file in csv_files:
    df = pd.read_csv(file, header=None)
    
    # ✅ ใช้เฉพาะไฟล์ที่มีขนาดตรงหรือใกล้เคียง
    if df.shape == (GRID_ROWS, GRID_COLS):
        all_dataframes.append(df)
    elif abs(df.shape[0] - GRID_ROWS) <= 1 and abs(df.shape[1] - GRID_COLS) <= 1:
        all_dataframes.append(df)

# ✅ ใช้ DataFrame ที่ขนาดใกล้เคียงที่สุด
if all_dataframes:
    grid_df = all_dataframes[0]  # ใช้ไฟล์แรกที่ตรงขนาด
    print(f"✅ ใช้ไฟล์ CSV ที่มีขนาด {grid_df.shape} เป็นแนวทางการเรียนรู้")
    grid = grid_df.astype(str).values.tolist()
else:
    print("⚠️ ไม่มีไฟล์ CSV ที่ตรงขนาด! กำลังสร้าง Grid เริ่มต้นแทน...")
    grid = initialize_grid(GRID_ROWS, GRID_COLS, E_START_POSITION)  # สร้าง Grid ใหม่

# ✅ แสดง Grid ที่ใช้
print("\n✅ Grid ที่ใช้สำหรับ AI:")
for row in grid:
    print(" | ".join(row))
print("\n")

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

# ✅ ฟังก์ชันแปลง CSV เป็น Grid
def csv_to_grid(df):
    df = df.fillna("0")  # แทน NaN ด้วย "0"
    grid = df.astype(str).values.tolist()
    for r in range(len(grid)):
        grid[r] = [cell if cell in SCORES else '0' for cell in grid[r]]
    return grid

# ✅ ฟังก์ชันโหลด CSV และใช้เป็นแนวทางการเรียนรู้
def load_grid_from_csv(csv_folder, rows, cols, e_position):
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
    best_match = None

    for file in csv_files:
        df = pd.read_csv(os.path.join(csv_folder, file), header=None)

        # ✅ ใช้เฉพาะไฟล์ที่มีขนาดตรงหรือใกล้เคียง
        if df.shape == (rows, cols):
            best_match = csv_to_grid(df)  # แปลง CSV เป็น Grid
            break
        elif abs(df.shape[0] - rows) <= 1 and abs(df.shape[1] - cols) <= 1:
            best_match = csv_to_grid(df)  # ใช้ขนาดใกล้เคียงที่สุด

    # ✅ ถ้ามี Grid จาก CSV ให้ลบ `E` ทั้งหมด และแทนที่ใหม่
    if best_match:
        print(f"✅ ใช้ไฟล์ CSV ที่มีขนาด {df.shape} เป็นแนวทางการเรียนรู้")
        
        # ลบ `E` ทั้งหมดใน Grid
        for r in range(rows):
            for c in range(cols):
                if best_match[r][c] == 'E':
                    best_match[r][c] = '0'  # แทนที่เป็นพื้นที่ว่าง
        
        # ✅ วาง `E` ตาม `E_START_POSITION`
        er, ec = e_position
        best_match[er-1][ec-1] = 'E'
        return best_match

    print("⚠️ ไม่มีไฟล์ CSV ที่ตรงขนาด! กำลังสร้าง Grid เริ่มต้นแทน...")
    return initialize_grid(rows, cols, e_position)

# ✅ โหลด Grid จาก CSV หรือสร้างใหม่
csv_folder = "data/maps"
grid = load_grid_from_csv(csv_folder, GRID_ROWS, GRID_COLS, E_START_POSITION)

# ✅ ตรวจสอบขนาดของ Grid
GRID_ROWS = len(grid)
GRID_COLS = max(len(row) for row in grid)

# ✅ แสดง Grid ที่ใช้
print("\n✅ Grid ที่ใช้สำหรับ AI:")
for row in grid:
    print(" | ".join(row))
print("\n")

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

# 4️⃣ ฟังก์ชันคำนวณคะแนนโดยละเอียด
def calculate_reward_verbose(grid):

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
    print(f"🔹 Base Score: {base_score}")

    # 2️⃣ คำนวณโบนัสแพทเทิร์น
    bonus = 0  

    # ✅ ค้นหาทุกแพทเทิร์น "HHH" และ "RRR" (แนวนอน)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS - 2):
            if grid[r][c] == 'H' and grid[r][c+1] == 'H' and grid[r][c+2] == 'H':
                bonus += 20
                print(f"✅ พบแพทเทิร์น HHH ที่แถว {r}, คอลัมน์ {c}-{c+2} (+20)")
            if grid[r][c] == 'R' and grid[r][c+1] == 'R' and grid[r][c+2] == 'R':
                bonus += 20
                print(f"✅ พบแพทเทิร์น RRR ที่แถว {r}, คอลัมน์ {c}-{c+2} (+20)")

    # ✅ ค้นหาแพทเทิร์น "H R H" แนวตั้ง
    for c in range(GRID_COLS):
        for r in range(GRID_ROWS - 2):
            if grid[r][c] == 'H' and grid[r+1][c] == 'R' and grid[r+2][c] == 'H':
                bonus += 20
                print(f"✅ พบแพทเทิร์น H R H ที่คอลัมน์ {c}, แถว {r}-{r+2} (+20)")

    # ✅ ค้นหาแพทเทิร์น HH บน RR และ RR บน HH
    for r in range(GRID_ROWS - 1):
        for c in range(GRID_COLS - 1):
            if (grid[r][c] == 'H' and grid[r][c+1] == 'H' and grid[r+1][c] == 'R' and grid[r+1][c+1] == 'R'):
                bonus += 20
                print(f"✅ พบแพทเทิร์น HH บน RR ที่แถว {r}-{r+1}, คอลัมน์ {c}-{c+1} (+20)")
            if (grid[r][c] == 'R' and grid[r][c+1] == 'R' and grid[r+1][c] == 'H' and grid[r+1][c+1] == 'H'):
                bonus += 20
                print(f"✅ พบแพทเทิร์น RR บน HH ที่แถว {r}-{r+1}, คอลัมน์ {c}-{c+1} (+20)")

    print(f"🏆 Bonus Score: {bonus}")

    # 3️⃣ ตรวจสอบเงื่อนไขผิดกฎ (Penalty)
    penalty = 0

    # ❌ เช็ค H ห้ามติด E (หัก -50 ต่อตัวที่ติด E)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if grid[r][c] == 'H':
                if (r > 0 and grid[r-1][c] == 'E') or (r < GRID_ROWS-1 and grid[r+1][c] == 'E') or \
                   (c > 0 and grid[r][c-1] == 'E') or (c < GRID_COLS-1 and grid[r][c+1] == 'E'):
                    penalty -= 50
                    print(f"❌ พบ H ติด E ที่แถว {r}, คอลัมน์ {c} (-50)")
    
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
        print(f"❌ พบ R ไม่ติด E ที่แถว {e_row}, คอลัมน์ {e_col} (-100)")

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
                    print(f"❌ H ที่แถว {r+1}, คอลัมน์ {c+1} ไม่ติด R (-100)")
    print(f"🚨 พบ H ที่ไม่ติด R ทั้งหมด {h_not_touching_r} ตัว (-{h_not_touching_r * 100})")

    # ตรวจสอบว่าถนน 'R' ถูกแบ่งออกเป็นหลายกลุ่มหรือไม่
    r_clusters = count_r_clusters(grid)
    if r_clusters > 1:
        penalty -= 100 * (r_clusters - 1)  # หัก -100 ต่อกลุ่ม R ที่เพิ่มขึ้น
        print(f"❌ พบ R แบ่งออกเป็น {r_clusters} กลุ่ม (-100 ต่อกลุ่ม)")

    # ✅ เพิ่มโบนัส G และ R ติด E
    additional_bonus = 0
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if grid[r][c] == 'G':
                if (r > 0 and grid[r-1][c] == 'E') or (r < GRID_ROWS-1 and grid[r+1][c] == 'E') or \
                   (c > 0 and grid[r][c-1] == 'E') or (c < GRID_COLS-1 and grid[r][c+1] == 'E'):
                    additional_bonus += 10
                    print(f"✅ พบ G ติด E ที่แถว {r}, คอลัมน์ {c} (+10)")
            if grid[r][c] == 'R':
                if (r > 0 and grid[r-1][c] == 'E') or (r < GRID_ROWS-1 and grid[r+1][c] == 'E') or \
                   (c > 0 and grid[r][c-1] == 'E') or (c < GRID_COLS-1 and grid[r][c+1] == 'E'):
                    additional_bonus += 5
                    print(f"✅ พบ R ติด E ที่แถว {r}, คอลัมน์ {c} (+5)")

    print(f"💎 Extra Bonus Score: {additional_bonus}")

    # รวมคะแนนทั้งหมด
    final_score = base_score + bonus + penalty + additional_bonus
    print(f"🏁 Final Score: {final_score}")
    return final_score

# 5️⃣ คำนวณคะแนนจาก Grid ที่ได้รับ
final_score = calculate_reward_verbose(grid)

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
            reward = calculate_reward_verbose(grid)  # ✅ คำนวณคะแนนเฉพาะเมื่อกริดเต็ม
            break
        r, c, char = action
        grid[r][c] = char
        next_state = grid
        update_q_table(state, action, 0, next_state)  # ✅ ให้ reward เป็น 0 จนกว่ากริดจะเต็ม
        state = next_state

    reward = calculate_reward_verbose(grid)  # ✅ คำนวณคะแนนครั้งสุดท้าย
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
