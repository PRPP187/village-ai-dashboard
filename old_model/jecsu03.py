import numpy as np
import random
import json
import os
import ast  # ✅ ใช้ ast.literal_eval แทน eval เพื่อความปลอดภัย

# ตั้งค่าขนาด Grid (สามารถเปลี่ยนได้ในอนาคต)
GRID_ROWS = 3
GRID_COLS = 3
EPISODES = 5000  # จำนวนรอบการเรียนรู้
ALPHA = 0.1  # Learning Rate
GAMMA = 0.9  # Discount Factor

# ค่าคะแนนของแต่ละอาคาร
SCORES = {'E': 10, 'G': 10, 'H': 15, 'R': 5, '0': 0}  # ✅ เปลี่ยน '?' เป็น '0'

Q_TABLE_FILE = "q_table.json"

# ✅ ฟังก์ชันโหลด Q-Table อย่างปลอดภัย
def load_q_table(filepath=Q_TABLE_FILE):
    """ โหลด Q-Table จากไฟล์ JSON พร้อมจัดการ Key Format """
    if not os.path.exists(filepath):
        print("⚠️ Q-Table file not found. Creating a new Q-Table...")
        return {}  # คืนค่า Q-Table ว่าง ๆ
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw_q_table = json.load(f)
            q_table = {}
            for k, v in raw_q_table.items():
                try:
                    if isinstance(k, str):
                        eval_k = ast.literal_eval(k)  # ✅ ใช้ ast.literal_eval แทน eval
                        if isinstance(eval_k, list):
                            new_key = tuple(map(tuple, eval_k))  # ✅ แปลง list of lists เป็น tuple of tuples
                        else:
                            new_key = eval_k
                    else:
                        new_key = k
                    q_table[new_key] = v
                except Exception as e:
                    print(f"⚠️ ข้าม key ที่ไม่สามารถแปลงได้: {k} - Error: {e}")
            return q_table
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
            json.dump({json.dumps(k): v for k, v in q_table.items()}, f, indent=4)  # ✅ ใช้ json.dumps() เพื่อให้ JSON รองรับ
    except Exception as e:
        print(f"⚠️ Error while saving Q-Table: {e}")

# โหลด Q-Table
q_table = load_q_table()

# ฟังก์ชันสร้าง Grid เริ่มต้น
def initialize_grid(rows, cols, e_position):
    """ สร้าง Grid ขนาดกำหนด และวาง E ตามตำแหน่งที่ผู้ใช้ต้องการ """
    grid = [['0' for _ in range(cols)] for _ in range(rows)]  # สร้าง Grid ว่าง

    # แปลงจาก 1-based index เป็น 0-based index
    r, c = e_position
    r, c = r - 1, c - 1  # ✅ แปลงตำแหน่งให้ตรงกับ index ใน Python
    if 0 <= r < rows and 0 <= c < cols:  # ตรวจสอบว่าอยู่ในขอบเขตของ Grid
        grid[r][c] = 'E'  # วาง E ตามตำแหน่งที่กำหนด
    else:
        print("⚠️ ตำแหน่ง E อยู่นอกขอบเขตของ Grid!")

    # แสดงผลลัพธ์ของกริด
    print("📌 Grid ที่สร้างได้:")
    for row in grid:
        print(" ".join(row))

    return grid

# ฟังก์ชันคำนวณคะแนนของ Grid
def calculate_reward(grid):
    score = sum(SCORES[grid[r][c]] for r in range(GRID_ROWS) for c in range(GRID_COLS) if grid[r][c] in SCORES)
    
    # ตรวจสอบโบนัสแพทเทิร์น
    if all(grid[0][c] == 'H' for c in range(GRID_COLS)):
        score += 20  # แพทเทิร์น HHH
    if all(grid[1][c] == 'R' for c in range(GRID_COLS)):
        score += 20  # แพทเทิร์น RRR
    for c in range(GRID_COLS):
        if grid[0][c] == 'H' and grid[1][c] == 'R' and grid[2][c] == 'H':
            score += 20  # แพทเทิร์น H R H
    
    return score

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
    state_str = json.dumps(state)  # ✅ ใช้ json.dumps() แปลง state เป็น string
    action_str = str(tuple(action))  # ✅ แปลง action (tuple) เป็น string
    next_state_str = json.dumps(next_state)
    
    if state_str not in q_table:
        q_table[state_str] = {}
    if action_str not in q_table[state_str]:
        q_table[state_str][action_str] = 0
    
    max_future_q = max(q_table[next_state_str].values(), default=0)
    q_table[state_str][action_str] = (1 - ALPHA) * q_table[state_str][action_str] + ALPHA * (reward + GAMMA * max_future_q)

# ฝึก AI ด้วย Reinforcement Learning
for episode in range(EPISODES):
    grid = initialize_grid(GRID_ROWS, GRID_COLS, (2, 1))  # เปลี่ยนตำแหน่ง E ได้
    state = grid
    total_reward = 0
    
    for _ in range(GRID_ROWS * GRID_COLS):
        action = choose_action(grid)
        if action is None:
            break  # ไม่มีที่วางแล้ว
        r, c, char = action
        grid[r][c] = char  # วางอาคาร
        reward = calculate_reward(grid)
        next_state = grid
        update_q_table(state, action, reward, next_state)
        state = next_state
        total_reward += reward
    
    if (episode + 1) % 1000 == 0:
        print(f"Episode {episode + 1}/{EPISODES}, Reward: {total_reward}")

# บันทึก Q-Table
save_q_table(q_table)

print("🎯 AI Training Completed! Q-Table Saved.")
