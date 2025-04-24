import json
import os
import csv
import cv2
import random
import numpy as np

# ✅ CONFIGURABLE SETTINGS
GRID_ROWS = 3  # 🟢 เปลี่ยนขนาด Grid ได้ง่ายที่นี่
GRID_COLS = 3  # 🟢 เปลี่ยนขนาด Grid ได้ง่ายที่นี่
EPISODES = 10000  # 🟢 เปลี่ยนจำนวนรอบของการเทรน AI ได้ง่ายที่นี่
RUNS = 10000  # 🟢 เปลี่ยนจำนวนรอบของการสร้าง Layout ที่ดีที่สุด

# ✅ FILE PATHS (แยกโฟลเดอร์สำหรับแต่ละประเภทของไฟล์)
DATA_PATH = "data/"
Q_TABLE_FILE = os.path.join(DATA_PATH, "q_table.json")
BEST_LAYOUT_FILE = os.path.join(DATA_PATH, "best_layouts.json")
CSV_PATH = os.path.join(DATA_PATH, "maps/csv/")
JSON_PATH = os.path.join(DATA_PATH, "maps/json/")
IMAGE_PATH = os.path.join(DATA_PATH, "maps/images/")

# ✅ Helper Functions
def neighbors(r, c, rows, cols):
    """ คืนค่าตำแหน่งรอบๆ (บน, ล่าง, ซ้าย, ขวา) """
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    return [(r+dr, c+dc) for dr, dc in directions if 0 <= r+dr < rows and 0 <= c+dc < cols]

# ✅ Decision Making for AI Exploration
def should_explore(epsilon=0.2):
    """ 20% ของเวลาให้ AI ทดลอง Layout ใหม่ """
    return random.uniform(0, 1) < epsilon

# ✅ FUNCTION TO LOAD MAP FILES
def load_map_from_csv(filename):
    """ โหลดแผนที่จากไฟล์ CSV """
    filepath = os.path.join(CSV_PATH, filename)
    with open(filepath, newline='') as csvfile:
        return [row for row in csv.reader(csvfile)]

def load_map_from_json(filename):
    """ โหลดแผนที่จากไฟล์ JSON """
    filepath = os.path.join(JSON_PATH, filename)
    with open(filepath, "r") as f:
        return json.load(f)["map"]

def load_map_from_image(filename):
    """ โหลดแผนที่จากไฟล์รูปภาพ """
    filepath = os.path.join(IMAGE_PATH, filename)
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    rows, cols = binary.shape
    return [['R' if binary[r, c] == 0 else 'H' for c in range(cols)] for r in range(rows)]

# ✅ FUNCTION TO LOAD ALL MAP FILES
def load_all_maps():
    """ โหลดแผนที่จากทุกไฟล์ในโฟลเดอร์ CSV, JSON, และ Images """
    all_maps = []
    
    # 🔹 โหลดจาก CSV
    for filename in os.listdir(CSV_PATH):
        if filename.endswith(".csv"):
            all_maps.append(load_map_from_csv(filename))
    
    # 🔹 โหลดจาก JSON
    for filename in os.listdir(JSON_PATH):
        if filename.endswith(".json"):
            all_maps.append(load_map_from_json(filename))
    
    # 🔹 โหลดจาก Images
    for filename in os.listdir(IMAGE_PATH):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            all_maps.append(load_map_from_image(filename))
    
    return all_maps

# ✅ Validation & Calculation Functions
def calculate_land_ratios(grid):
    """ คำนวณสัดส่วนพื้นที่ """
    total = sum(len(row) for row in grid)
    counts = {t: sum(row.count(t) for row in grid) for t in ['H', 'R', 'G', 'C', 'U']}
    return {t: (counts[t] / total) * 100 for t in counts}

def is_valid(grid, r, c, char):
    """ ตรวจสอบว่าตำแหน่ง `(r, c)` สามารถวางตัวอักษร `char` ได้หรือไม่ """
    return grid[r][c] == '0'

def get_reward(grid):
    """ คำนวณคะแนนของ Layout """
    score_map = {'E': 25, 'C': 20, 'H': 10, 'G': 15, 'R': 5, 'U': 10, '0': -10}
    return sum(row.count(k) * v for k, v in score_map.items() for row in grid)

def is_road_connected(grid):
    """ ตรวจสอบว่าถนนเชื่อมถึงกันหรือไม่ """
    return True  # (อาจต้องเพิ่มโค้ดตรวจสอบ BFS/DFS)

# ✅ Reinforcement Learning (Q-Table)
def load_q_table(filepath=Q_TABLE_FILE):
    """ โหลด Q-Table จากไฟล์ JSON อย่างปลอดภัย """
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

def save_q_table(q_table, filepath=Q_TABLE_FILE):
    """ บันทึก Q-Table ลงไฟล์ JSON """
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(q_table, f, indent=4)
    except Exception as e:
        print(f"⚠️ Error while saving Q-Table: {e}")

def update_q_table(q_table, state, action, reward, next_state):
    """ อัปเดตค่า Q-Table """
    q_table[str(state)] = q_table.get(str(state), {})
    q_table[str(state)][str(action)] = reward
    return q_table

def choose_next_move(grid):
    """ ให้ AI เลือกตำแหน่ง `(r, c)` และตัวอักษร `char` ที่ดีที่สุด """
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == '0':
                return r, c, random.choice(['H', 'R', 'G', 'C', 'U'])
    return None

# ✅ Database of Best Layouts
def load_best_layout(filepath="best_layouts.json"):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None, float('-inf')
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {filepath}: {e}")
        return None, float('-inf')

def save_best_layout(layout, score, filepath="best_layouts.json"):
    with open(filepath, "w") as f:
        json.dump({"layout": layout, "score": score}, f, indent=4)

# ✅ AI TRAINING & GENERATION
def train_ai():
    """ เทรน AI โดยใช้ Q-Table และข้อมูลจากทุกแผนที่ """
    q_table = load_q_table()
    all_maps = load_all_maps()
    
    for episode in range(EPISODES):
        for grid in all_maps:  # 🔹 ใช้ทุกแผนที่ในการฝึก AI
            for step in range(GRID_ROWS * GRID_COLS):
                move = choose_next_move(grid)
                if move is None:
                    break
                r, c, char = move
                prev_state = tuple(map(tuple, grid))
                grid[r][c] = char
                reward = get_reward(grid)
                q_table = update_q_table(q_table, prev_state, (r, c, char), reward, grid)
    save_q_table(q_table)

def generate_best_layout(rows=GRID_ROWS, cols=GRID_COLS, runs=RUNS, map_file=None):
    best_layout, best_score = load_best_layout()

    # Ensure best_score is a numeric value
    if isinstance(best_score, str):
        try:
            best_score = float(best_score)
        except ValueError:
            best_score = float('-inf')

    for _ in range(runs):
        grid = [['0' for _ in range(cols)] for _ in range(rows)]
        for _ in range(rows * cols):
            move = choose_next_move(grid)
            if move is None:
                break
            r, c, char = move
            grid[r][c] = char
        score = get_reward(grid)
        if score > best_score:
            best_score = score
            best_layout = [row[:] for row in grid]
            save_best_layout(best_layout, best_score)
    return best_layout, best_score

# ✅ MAIN EXECUTION
if __name__ == "__main__":
    train_ai()
    best_layout, best_score = generate_best_layout()
    print(f"\n🏆 Layout ที่ดีที่สุด ได้คะแนน: {best_score}")
    for row in best_layout:
        print(" ".join(row))
