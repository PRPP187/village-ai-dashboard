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

# ✅ ค้นหาไฟล์ CSV ในโฟลเดอร์ goodcsv
csv_files = glob.glob(r"C:\Users\USER\Desktop\AI_Housing_Project\data\maps\CSV\goodcsv\**\*.csv")

def load_or_initialize_grid(csv_folder, rows, cols, e_start_position):
    """
    โหลด Grid จากไฟล์ CSV ที่มีขนาดตรงกัน หรือใช้ไฟล์ที่ขนาดใกล้เคียงที่สุด
    ถ้าไม่พบไฟล์ที่ใช้ได้ จะสร้าง Grid ขนาดที่กำหนดขึ้นมาใหม่
    """
    print("📌 กำลังค้นหาไฟล์ CSV...")
    csv_files = glob.glob(f"{csv_folder}/**/*.csv", recursive=True)

    if not csv_files:
        print("⚠️ ไม่พบไฟล์ CSV! กำลังสร้าง Grid เริ่มต้นแทน...")
        return initialize_grid(rows, cols, e_start_position)

    all_dataframes = []  # เก็บไฟล์ที่ขนาดตรงกัน
    best_match_df = None  # ไฟล์ที่ขนาดใกล้เคียงที่สุด

    # ✅ วนลูปหาไฟล์ CSV ที่มีขนาดตรงกัน และ `E` อยู่ที่ตำแหน่งที่ต้องการ
    for file in csv_files:
        df = pd.read_csv(file, header=None, nrows=rows)  # จำกัดจำนวนแถว
        grid = df.astype(str).values.tolist()  # แปลง CSV เป็น Grid

        actual_rows, actual_cols = df.shape  # ขนาดของ Grid จริงใน CSV
        # ✅ ค้นหาตำแหน่ง `E` ทั้งหมดใน Grid ที่โหลดจาก CSV
        e_found = [(r, c) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] == 'E']

        if e_found:
            # ✅ เลือก `E` ที่อยู่ใกล้ `E_START_POSITION` ที่สุด
            closest_e = min(
                e_found, 
                key=lambda pos: abs(pos[0] - (e_start_position[0] - 1)) + abs(pos[1] - (e_start_position[1] - 1))
            )

            # ✅ ถ้าตำแหน่ง `E` ไม่ตรงกับ `E_START_POSITION` → ย้ายไปที่ `E_START_POSITION`
            if closest_e != (e_start_position[0]-1, e_start_position[1]-1):
                print(f"⚠️ `E` อยู่ที่ตำแหน่ง {closest_e} แต่ต้องการย้ายไป {e_start_position}")
                
                # ลบ `E` เดิม
                grid[closest_e[0]][closest_e[1]] = '0'
                
                # วาง `E` ที่ตำแหน่งที่ถูกต้อง
                grid[e_start_position[0]-1][e_start_position[1]-1] = 'E'

            return grid  # คืนค่า Grid ทันทีเพราะตรงตามที่ต้องการ

        # ✅ ถ้าขนาดตรงกันแต่ `E` ไม่ตรงตำแหน่ง ให้เก็บไว้เป็นตัวเลือก
        if (actual_rows, actual_cols) == (rows, cols):
            print(f"⚠️ ไฟล์ {file} มีขนาดตรงกัน แต่ `E` ไม่อยู่ที่ตำแหน่งที่ต้องการ!")
            all_dataframes.append(df)

        # ✅ ถ้าไม่มีไฟล์ขนาดตรงกัน ให้เลือกไฟล์ขนาดใกล้เคียง
        elif abs(actual_rows - rows) <= 1 and abs(actual_cols - cols) <= 1:
            if best_match_df is None or (abs(actual_rows - rows) + abs(actual_cols - cols) <
                             abs(best_match_df.shape[0] - rows) + abs(best_match_df.shape[1] - cols)):
                best_match_df = df

    # ✅ ถ้าไม่มีไฟล์ที่ขนาดตรงและ `E` ตรง แต่มีไฟล์ขนาดตรง → ใช้ไฟล์นั้นแทน
    if all_dataframes:
        grid_df = all_dataframes[0]  # ใช้ไฟล์แรกที่เจอที่ขนาดตรง
        print(f"✅ ใช้ไฟล์ CSV ที่ขนาดตรงกัน ({grid_df.shape}) แต่ `E` อาจไม่ตรงตำแหน่งที่ต้องการ")
        return grid_df.astype(str).values.tolist()

    # ✅ ถ้าไม่มีไฟล์ขนาดตรงกันเลย → ใช้ไฟล์ขนาดใกล้เคียง
    if best_match_df is not None:
        print(f"✅ ใช้ไฟล์ CSV ที่ขนาดใกล้เคียงที่สุด ({best_match_df.shape})")
        return best_match_df.astype(str).values.tolist()

    # ✅ ถ้าไม่มีไฟล์ที่ใช้ได้เลย → สร้าง Grid ใหม่
    print("⚠️ ไม่มีไฟล์ CSV ที่ตรงขนาด! กำลังสร้าง Grid เริ่มต้นแทน...")
    return initialize_grid(rows, cols, e_start_position)

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

    # ✅ ตรวจสอบ H ที่ไม่ติด R (หัก -50 ต่อตัวที่ไม่ติด R)
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
                    penalty -= 50  # หัก -50 ต่อ H ที่ไม่ติด R
                    print(f"❌ H ที่แถว {r+1}, คอลัมน์ {c+1} ไม่ติด R (-50)")
    print(f"🚨 พบ H ที่ไม่ติด R ทั้งหมด {h_not_touching_r} ตัว (-{h_not_touching_r * 50})")

    # ✅ ตรวจสอบว่ามีกลุ่ม `R` แยกกันหรือไม่
    r_clusters = count_r_clusters(grid)
    if r_clusters > 1:
        penalty -= 100 * (r_clusters - 1)  
        print(f"❌ พบ `R` แยกเป็น {r_clusters} กลุ่ม (-100 ต่อกลุ่ม)")

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

    if all(cell != '0' for row in grid for cell in row):
        total_reward += 50  # โบนัสพิเศษถ้า Grid เต็ม

    print(f"💎 Extra Bonus Score: {additional_bonus}")

    # รวมคะแนนทั้งหมด
    final_score = base_score + bonus + penalty + additional_bonus
    print(f"🏁 Final Score: {final_score}")
    return final_score

# ฟังก์ชันเลือกการกระทำ (วางอาคารใน Grid)
def choose_action(grid):
    """ เลือกตำแหน่งและอาคารที่จะวางใน Grid """
    
    # 🔹 สร้าง State ปัจจุบันของ Grid เป็น JSON (แปลงเป็น string เพื่อใช้กับ Q-Table)
    state_str = json.dumps(grid)

    # 🔹 ถ้ามีค่าใน Q-Table → ใช้ค่าที่ดีที่สุดจาก Q-Table
    if state_str in q_table:
        action_str = max(q_table[state_str], key=q_table[state_str].get)  # เลือก action ที่มีค่าสูงสุด
        print(f"📊 AI ใช้ Q-Table: เลือก {action_str} ด้วยคะแนน {q_table[state_str][action_str]}")
        return ast.literal_eval(action_str)  # แปลง string กลับเป็น tuple (r, c, char)

    # 🔹 ถ้าไม่มีค่าใน Q-Table → เลือกแบบสุ่ม
    empty_cells = [(r, c) for r in range(GRID_ROWS) for c in range(GRID_COLS) if grid[r][c] == '0']

    if not empty_cells:
        print("⚠️ ไม่มีที่ว่างให้วางอาคาร!")
        return None  # ไม่มีตำแหน่งที่วางได้

    r, c = random.choice(empty_cells)  # เลือกตำแหน่งว่างแบบสุ่ม
    char = random.choice(['H', 'R', 'G'])  # เลือกประเภทอาคารแบบสุ่ม
    print(f"🛠 AI เลือกวาง {char} ที่ตำแหน่ง ({r},{c})")
    return r, c, char  # ส่งคืนตำแหน่งและตัวอาคารที่เลือก

# ฟังก์ชันอัปเดต Q-Table
def update_q_table(state, action, reward, next_state):
    state_str = json.dumps(state)
    action_str = str(tuple(action))
    next_state_str = json.dumps(next_state)
    
    if state_str not in q_table or not q_table[state_str]:  # ตรวจสอบว่า Q-Table มีค่าหรือไม่
        q_table[state_str] = {}
    if action_str in q_table[state_str]:

        max_future_q = max(q_table.get(next_state_str, {}).values() or [0])
        q_table[state_str][action_str] = (1 - ALPHA) * q_table[state_str][action_str] + ALPHA * (reward + GAMMA * max_future_q)

def train_ai(episodes, grid_rows, grid_cols, csv_files):
    """
    ฝึก AI ด้วย Reinforcement Learning (Q-Learning)
    :param episodes: จำนวนรอบที่ฝึก
    :param grid_rows: จำนวนแถวของ Grid
    :param grid_cols: จำนวนคอลัมน์ของ Grid
    :return: best_grid, best_score
    """
    best_grid = None
    best_score = float('-inf')

    for episode in range(episodes):
        grid = load_or_initialize_grid(csv_files, grid_rows, grid_cols, E_START_POSITION)  # ✅ ดึงข้อมูลจาก CSV
        state = grid
        total_reward = 0
        reward = 0  # ✅ กำหนดค่าเริ่มต้นให้ reward เป็น 0

        for _ in range(grid_rows * grid_cols):
            action = choose_action(grid)
        if action is None and _ > (grid_rows * grid_cols) // 2:
            break  # หยุดฝึกก็ต่อเมื่อผ่านไปแล้วครึ่งหนึ่งของ Grid

            r, c, char = action  # ✅ ตำแหน่งที่ AI เลือกวาง
            grid[r][c] = char  # ✅ วางอาคารใน Grid

            # ✅ คำนวณคะแนนหลังจากวางแต่ละ 5 ครั้ง
            if _ % 5 == 0:  # คำนวณทุก ๆ 5 รอบเพื่อลดภาระการคำนวณ
                reward = calculate_reward_verbose(grid)
                print(f"🏆 คำนวณคะแนน: {reward}")
                update_q_table(state, action, reward, grid)

            state = grid  # ✅ อัปเดตสถานะใหม่ของ Grid

        reward = calculate_reward_verbose(grid)  # ✅ คำนวณคะแนนครั้งสุดท้าย
        total_reward = reward

        # ✅ บันทึก Layout ที่ดีที่สุด
        if total_reward > best_score:
            best_score = total_reward
            best_grid = [row[:] for row in grid]

        # ✅ แสดงผลทุก ๆ 1000 Episodes
        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}")

    return best_grid, best_score

# บันทึก Q-Table
save_q_table(q_table)

best_grid, best_score = train_ai(EPISODES, GRID_ROWS, GRID_COLS, csv_files)

print("\n🏆 Best Layout Found:")
for row in best_grid:
    print(" | ".join(row))

print("🎯 AI Training Completed! Q-Table Saved.")
print(f"\n✅ Best Score Achieved: {best_score}")

print(f"🏆 Best Score: {best_score}")
print("🏆 Best Grid Layout:")
for row in best_grid:
    print(" ".join(row))
