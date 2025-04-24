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
RUNS = 1000  # 🟢 เปลี่ยนจำนวนรอบของการสร้าง Layout ที่ดีที่สุด

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
    
    # ❌ ห้ามวางทับตำแหน่งที่มีอยู่แล้ว
    if grid[r][c] != '0':
        return False

    rows, cols = len(grid), len(grid[0])
    
    if char == 'E':
        if r != 0 and r != rows - 1 and c != 0 and c != cols - 1:
            return False  # ❌ E ต้องติดขอบแผนที่เท่านั้น
        if sum(grid[nr][nc] == 'E' for nr, nc in neighbors(r, c, rows, cols)) == 0:
            return False  # ❌ ต้องมี E ติดกัน
        return True

    if char == 'R':
        if not is_road_connected(grid, r, c):
            return False  # ❌ ถนนต้องเชื่อมกันเสมอ
        return True

    # ✅ เงื่อนไขสำหรับ G (Green Space)
    if char == 'G':
        adjacent_G = sum(grid[nr][nc] == 'G' for nr, nc in neighbors(r, c, rows, cols))
        if adjacent_G == 0:
            return False  # ❌ G ต้องมี G ติดกันเสมอ
        return True

    if char == 'H':
        adjacent_R = sum(grid[nr][nc] == 'R' for nr, nc in neighbors(r, c, rows, cols))
        adjacent_G = sum(grid[nr][nc] == 'G' for nr, nc in neighbors(r, c, rows, cols))
        if adjacent_R == 0:
            return False  # ❌ บ้านต้องติดถนนเสมอ
        if adjacent_G == 0:
            return False  # ❌ บ้านต้องติดสวนด้วย
        return True

    # ✅ เงื่อนไขสำหรับ C (Community Area)
    if char == 'C':
        if sum(grid[nr][nc] == 'R' for nr, nc in neighbors(r, c, rows, cols)) == 0:
            return False  # ❌ ต้องติดถนน
        if sum(grid[nr][nc] == 'C' for nr, nc in neighbors(r, c, rows, cols)) == 0:
            return False  # ❌ ต้องมี C ติดกัน
        return True

    # ✅ เงื่อนไขสำหรับ P (Parking)
    if char == 'P':
        if sum(grid[nr][nc] == 'C' for nr, nc in neighbors(r, c, rows, cols)) == 0:
            return False  # ❌ ต้องติดพื้นที่ส่วนกลาง
        if sum(grid[nr][nc] == 'R' for nr, nc in neighbors(r, c, rows, cols)) == 0:
            return False  # ❌ ต้องติดถนน
        return True

    # ✅ เงื่อนไขสำหรับ W (Water)
    if char == 'W':
        if sum(grid[nr][nc] == 'G' for nr, nc in neighbors(r, c, rows, cols)) == 0:
            return False  # ❌ ต้องติดสวน
        if any(grid[nr][nc] in ['H', 'R'] for nr, nc in neighbors(r, c, rows, cols)):
            return False  # ❌ ห้ามติดบ้านหรือถนน
        return True

    return True  # ✅ อื่น ๆ วางได้

def get_reward(grid):
    """ คำนวณคะแนนของ Layout """
    score_map = {
        'E': 5, 'R': -5, 'G': 15, 'H': 20, 'C': 10, 'U': 10, 'W': 5, 'P': 10, 'D': 10
    }
    score = 0
    first_char = None  # ใช้เช็คว่าตัวแรกที่ลงคืออะไร

    total_cells = len(grid) * len(grid[0])  # จำนวนช่องทั้งหมด
    count_E = sum(row.count('E') for row in grid)  # นับจำนวน E

    # 🟢 ค้นหากลุ่ม E ที่แยกกัน
    visited = set()
    e_clusters = 0

    def dfs(r, c):
        """ ค้นหากลุ่มที่เชื่อมกันของ 'E' """
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if (cr, cc) in visited:
                continue
            visited.add((cr, cc))
            for nr, nc in neighbors(cr, cc, len(grid), len(grid[0])):
                if grid[nr][nc] == 'E' and (nr, nc) not in visited:
                    stack.append((nr, nc))

    # นับจำนวนกลุ่มของ E
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 'E' and (r, c) not in visited:
                e_clusters += 1
                dfs(r, c)

    for r in range(len(grid)):
        for c in range(len(grid[0])):
            cell = grid[r][c]

            if cell != '0' and first_char is None:
                first_char = cell  # เก็บตัวแรกที่ลง
                if first_char == 'E':
                    score += 50  # ✅ ถ้า E ลงเป็นตัวแรก → ให้คะแนนเยอะ
                else:
                    score -= 30  # ❌ ถ้าตัวอื่นลงก่อน → หักคะแนน

            score += score_map.get(cell, 0)  # ✅ คำนวณคะแนนปกติของตัวอักษร

            # ✅ กฎเพิ่มเติม
            if cell == 'E':
                if r == 0 or r == len(grid) - 1 or c == 0 or c == len(grid[0]) - 1:
                    score += 10  # ✅ E อยู่ขอบ → เพิ่มคะแนน
                else:
                    score -= 20  # ❌ E ไม่อยู่ขอบ → หักคะแนน

            if cell == 'R':
                if not is_road_connected(grid):
                    score -= 20  # ❌ ถนนไม่เชื่อมกัน → หักคะแนน

            if cell == 'G':
                neighbors_G = sum(grid[nr][nc] == 'G' for nr, nc in neighbors(r, c, len(grid), len(grid[0])))
                if neighbors_G > 0:
                    score += 10  # ✅ G อยู่ติดกันเป็นกลุ่ม → เพิ่มคะแนน
                else:
                    score -= 10  # ❌ G อยู่เดี่ยว → หักคะแนน

    # ✅ **ปรับคะแนนตามจำนวน `E`**
    E_ratio = count_E / total_cells  # คำนวณสัดส่วนของ E ใน Grid
    if E_ratio < 0.075:  # ✅ **เปลี่ยนจาก 10% → 7.5%**
        score += 30  # ✅ ถ้า E น้อยกว่า 7.5% → **ให้คะแนนเพิ่ม**
    else:
        score -= 20  # ❌ ถ้า E มากกว่า 7.5% → **หักคะแนน**

    # ✅ **ถ้า `E` แยกเป็นหลายกลุ่ม → หักคะแนน**
    if e_clusters > 1:
        score -= 60 * (e_clusters - 1)  # ❌ หัก 50 คะแนนต่อกลุ่มที่แยกกัน

    return score


def is_road_connected(grid, r=None, c=None):
    """
    ตรวจสอบว่าถนนทั้งหมดเชื่อมต่อกันหรือไม่
    ถ้าระบุ (r, c) จะตรวจเฉพาะจุดนั้น
    """
    rows, cols = len(grid), len(grid[0])
    visited = set()
    
    # หา R จุดเริ่มต้น
    road_start = None
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 'R':
                road_start = (i, j)
                break
        if road_start:
            break
    
    if not road_start:
        return False  # ❌ ไม่มีถนนเลย

    # BFS ตรวจสอบว่าถนนเชื่อมกันไหม
    queue = [road_start]
    while queue:
        x, y = queue.pop(0)
        if (x, y) in visited:
            continue
        visited.add((x, y))
        for nx, ny in neighbors(x, y, rows, cols):
            if grid[nx][ny] == 'R' and (nx, ny) not in visited:
                queue.append((nx, ny))

    # ✅ ตรวจสอบว่าถนนทั้งหมดถูกเยี่ยมครบไหม
    all_roads = {(i, j) for i in range(rows) for j in range(cols) if grid[i][j] == 'R'}
    if visited != all_roads:
        return False  # ❌ มีถนนที่เชื่อมไม่ถึงกัน
    
    # ✅ ถ้ามี (r, c) ให้เช็คแค่จุดนั้นว่าติดถนนที่เชื่อมไหม
    if r is not None and c is not None:
        return (r, c) in visited

    return True  # ✅ ถนนเชื่อมกันทั้งหมด

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
    """ เลือกตำแหน่งถัดไปที่ AI จะวางอาคาร """
    empty_cells = [(r, c) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] == '0']
    
    if not empty_cells:
        return None  # ❌ ไม่มีตำแหน่งว่างแล้ว
    
    r, c = random.choice(empty_cells)  # ✅ เลือกตำแหน่งที่ว่างเสมอ
    char = random.choice(['H', 'R', 'G', 'C', 'U'])  # ✅ เลือกตัวอักษรแบบสุ่ม (สามารถปรับได้ตามกฎ)
    
    return r, c, char  # ✅ คืนค่าตำแหน่งที่ถูกเลือก

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
        print(f"🧠 Training AI: Episode {episode+1}/{EPISODES}")  # ✅ แจ้ง Progress ระหว่าง Train AI
        
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
    
    # ✅ สร้าง grid เปล่า
    grid = [['0' for _ in range(cols)] for _ in range(rows)]  

    # ✅ วาง E ที่ขอบแผนที่ก่อนเริ่มสร้าง Layout
    edge_positions = [(0, c) for c in range(cols)] + [(r, 0) for r in range(rows)] + \
                     [(rows - 1, c) for c in range(cols)] + [(r, cols - 1) for r in range(rows)]
    random.shuffle(edge_positions)

    e_placed = 0
    for r, c in edge_positions:
        if e_placed < 3 and grid[r][c] == '0':  # ✅ ตรวจสอบว่าที่วางว่างก่อน
            grid[r][c] = 'E'
            e_placed += 1
        if e_placed >= 3:
            break

    # ✅ ตรวจสอบว่า E วางถูกต้องหรือไม่
    print("🔍 ตรวจสอบการวาง E:")
    for row in grid:
        print(" ".join(row))

    # ✅ แปลง `best_score` เป็นตัวเลขถ้าเป็น string
    if isinstance(best_score, str):
        try:
            best_score = float(best_score)
        except ValueError:
            best_score = float('-inf')

    # ✅ สร้าง Layout โดยใช้ AI
    for _ in range(runs):
        temp_grid = [row[:] for row in grid]  # ✅ ใช้สำเนาของ grid เพื่อไม่ให้ทับกัน
        for _ in range(rows * cols):
            move = choose_next_move(temp_grid)  # ✅ ให้ AI เลือกจุดวางตัวอักษร
            if move is None:
                break  # ❌ หยุดถ้าไม่มีที่วางแล้ว
            r, c, char = move
            if temp_grid[r][c] == '0':  # ✅ ตรวจสอบว่าที่วางว่างก่อนวาง
                temp_grid[r][c] = char

        score = get_reward(temp_grid)  # ✅ คำนวณคะแนน Layout
        if score > best_score:  # ✅ อัปเดต Layout ที่ดีที่สุด
            best_score = score
            best_layout = [row[:] for row in temp_grid]
            save_best_layout(best_layout, best_score)

    return best_layout, best_score

# ✅ MAIN EXECUTION
if __name__ == "__main__":
    train_ai()
    best_layout, best_score = generate_best_layout()
    print(f"\n🏆 Layout ที่ดีที่สุด ได้คะแนน: {best_score}")
    for row in best_layout:
        print(" ".join(row))
