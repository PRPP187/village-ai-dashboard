import numpy as np
import random
import json
import os
import ast  # ✅ ใช้ ast.literal_eval แทน eval เพื่อความปลอดภัย

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
E_START_POSITION = (2, 2)

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

# ฟังก์ชันสร้าง Grid เริ่มต้น (แก้ให้รองรับค่าที่ผู้ใช้คาดหวัง)
def initialize_grid(rows, cols, e_position=E_START_POSITION):
    """ สร้าง Grid ขนาดกำหนด และวาง E ตามตำแหน่งที่ผู้ใช้ต้องการ (1-based index) """
    grid = [['0' for _ in range(cols)] for _ in range(rows)]  # สร้าง Grid ว่าง

    # แปลงจาก 1-based index เป็น 0-based index
    r, c = e_position
    r, c = r - 1, c - 1  # ✅ แปลงตำแหน่งให้ตรงกับ index ใน Python
    if 0 <= r < rows and 0 <= c < cols:  # ตรวจสอบว่าอยู่ในขอบเขตของ Grid
        grid[r][c] = 'E'  # วาง E ตามตำแหน่งที่กำหนด
    else:
        print("⚠️ ตำแหน่ง E อยู่นอกขอบเขตของ Grid!")

    return grid

# ฟังก์ชันคำนวณคะแนนของ Grid
def calculate_reward(grid):
    score = sum(SCORES[grid[r][c]] for r in range(GRID_ROWS) for c in range(GRID_COLS) if grid[r][c] in SCORES)
    
    # ตรวจสอบโบนัสแพทเทิร์น
    bonus = 0  
    if all(grid[0][c] == 'H' for c in range(GRID_COLS)):
        bonus += 20  # แพทเทิร์น HHH
    if all(grid[1][c] == 'R' for c in range(GRID_COLS)):
        bonus += 20  # แพทเทิร์น RRR
    # ตรวจสอบแพทเทิร์น H R H ในแต่ละคอลัมน์
    for c in range(GRID_COLS):
        if grid[0][c] == 'H' and grid[1][c] == 'R' and grid[2][c] == 'H':
            bonus += 20  # แพทเทิร์น H R H

    score += bonus  # เพิ่มโบนัสหลังจากตรวจสอบทั้งหมด

    # ตรวจสอบเงื่อนไขผิดกฎ
    penalty = 0  
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if grid[r][c] == 'H':
                # ตรวจสอบว่า H ติด R หรือไม่
                if not ((r > 0 and grid[r-1][c] == 'R') or (r < GRID_ROWS-1 and grid[r+1][c] == 'R') or
                        (c > 0 and grid[r][c-1] == 'R') or (c < GRID_COLS-1 and grid[r][c+1] == 'R')):
                    penalty -= 20  # ❌ หักคะแนนถ้า H ไม่ติด R

                # ตรวจสอบว่า H ติด E หรือไม่ (หัก -30 ต่อคู่)
                h_e_penalty = 0  
                if (r > 0 and grid[r-1][c] == 'E'):
                    h_e_penalty -= 30
                if (r < GRID_ROWS-1 and grid[r+1][c] == 'E'):
                    h_e_penalty -= 30
                if (c > 0 and grid[r][c-1] == 'E'):
                    h_e_penalty -= 30
                if (c < GRID_COLS-1 and grid[r][c+1] == 'E'):
                    h_e_penalty -= 30

                penalty += h_e_penalty  # เพิ่มค่าหักจาก H ติด E
                if h_e_penalty < 0:
                    print(f"❌ H ติด E ที่ตำแหน่ง ({r}, {c}), หัก {h_e_penalty} คะแนน")

    score += penalty  # เพิ่มค่าหักคะแนน

    print(f"🏆 Base Score: {score}, Bonus: {bonus}, Penalty: {penalty}, Final Score: {score}")
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

    if (episode + 1) % 2500 == 0:
        print(f"Episode {episode + 1}/{EPISODES}, Reward: {total_reward}")

# บันทึก Q-Table
save_q_table(q_table)

print("🎯 AI Training Completed! Q-Table Saved.")
print(f"🏆 Best Score: {best_score}")
print("🏆 Best Grid Layout:")
for row in best_grid:
    print(" ".join(row))
