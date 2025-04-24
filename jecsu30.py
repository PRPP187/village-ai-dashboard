# =========================================================
# 📌 1️⃣ Import Library และกำหนดค่าพื้นฐาน
# =========================================================
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
import copy
import hashlib
import heapq
import psutil
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count
import multiprocessing
import sqlite3
from tabulate import tabulate
from flask import Flask, request, jsonify  # ✅ Import Flask components
from collections import defaultdict
from scipy.ndimage import label

# =========================================================
# 📌 2️⃣ กำหนดค่าพื้นฐานสำหรับการทำงานของ AI
# =========================================================
# ✅ ตั้งค่าพื้นฐาน
grid_sizes = [(3, 3), (4, 3), (3, 4), (4, 4)]

EPISODES = 10

ALPHA_START = 0.1
ALPHA_END = 0.01
ALPHA_DECAY_RATE = 0.001
GAMMA = 0.9
EPSILON_START = 0.7  # ✅ ลดลงจาก 1.0 เป็น 0.5
EPSILON_END = 0.01    # ค่าต่ำสุดของ epsilon
EPSILON_DECAY = 0.95 # ✅ ลดช้าลง

SCORES = {'E': 50, 'G': 15, 'H': 20, 'R': 10, '0': -50}

Q_TABLE_FILE = "q_table.json"
conn = sqlite3.connect("q_table.db", check_same_thread=False)
cursor = conn.cursor()

q_table_lock = threading.Lock()  # ✅ ใช้ Lock ป้องกันปัญหาการอัปเดตพร้อมกัน
LOCK_FILE = "q_table.lock"  # ✅ ใช้ Lock ป้องกันการเขียนไฟล์ซ้อนกัน
# 🔹 ใช้ Lock ป้องกันไฟล์เสียหายจากการเขียนพร้อมกัน

last_load_time = time.time()
last_update_time = time.time() 
last_q_table_hash = ""
# 🔹 เก็บเวลาอัปเดตล่าสุด

q_table = {}
# 🔹 สร้างตัวแปร Q-Table (ต้องมาก่อน load_q_table)

conn = sqlite3.connect("q_table.db", check_same_thread=False)
cursor = conn.cursor()

# ✅ วนลูป Q-Table เพื่อดึงค่า state, action, q_value
for state, actions in q_table.items():
    for action, q_value in actions.items():
        cursor.execute("INSERT OR REPLACE INTO q_table (state_key, action_key, q_value) VALUES (?, ?, ?)",
                       (str(state), str(action), q_value))

conn.commit()
conn.close()

app = Flask(__name__)  # ✅ สร้าง Flask Application
@app.route('/update_q_table', methods=['POST'])

# =========================================================
# 📌 3️⃣ ฟังก์ชันเกี่ยวกับ Grid (การสร้าง, โหลด, และให้คะแนน)
# =========================================================
def load_or_create_grid(rows, cols, e_position=(1, 1), csv_folder="data/maps/CSV/goodcsv"):
    """ ✅ โหลด Grid จากไฟล์ CSV หรือสร้างใหม่ถ้าไม่มี """
    print(f"\n🔹 [STEP] กำลังโหลด/สร้าง Grid ขนาด {rows}x{cols}, วาง `E` ที่ {e_position}")
    
    csv_files = glob.glob(f"{csv_folder}/**/*.csv", recursive=True)
    best_grid = None
    best_score = float('-inf')

    # ✅ ตรวจสอบไฟล์ CSV ทั้งหมด
    for file in csv_files:
        df = pd.read_csv(file, header=None)
        if df.shape == (rows, cols) and 'E' in df.values:
            grid_from_csv = df.astype(str).values.tolist()
            score = calculate_reward_verbose(grid_from_csv)  # ✅ ใช้ฟังก์ชันให้คะแนน

            # ✅ ใช้ Grid ที่ให้คะแนนสูงสุด
            if score > best_score:
                best_score = score
                best_grid = grid_from_csv

    if best_grid is not None:
        print(f"✅ โหลด Grid ที่ดีที่สุดจาก CSV (คะแนน: {best_score})")
        print_grid(best_grid)  # ✅ แสดง Grid ที่โหลดมา
        return best_grid

    # ✅ ถ้าไม่มี Grid ที่เหมาะสม สร้างใหม่
    print(f"⚠️ ไม่มี Grid ที่เหมาะสม สร้างใหม่ {rows}x{cols}")
    grid = [['0' for _ in range(cols)] for _ in range(rows)]
    r, c = min(e_position[0], rows-1), min(e_position[1], cols-1)
    grid[r][c] = 'E'
    
    print(f"✅ Grid ถูกสร้างขึ้นใหม่ พร้อมวาง `E` ที่ {e_position}")
    print_grid(grid)  # ✅ แสดง Grid ที่สร้างใหม่
    return grid

def get_edge_positions(rows, cols):
    """ ✅ คืนค่าตำแหน่งขอบของ Grid โดยป้องกันค่าเกินขอบเขต """
    edge_positions = []
    for r in range(rows):
        for c in range(cols):
            if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                if 0 <= r < rows and 0 <= c < cols:  # ✅ เช็คก่อนบันทึก
                    edge_positions.append((r, c))
    return edge_positions

def calculate_reward_verbose(grid):
    """ ✅ คำนวณคะแนนของแผนที่ โดยพิจารณาจากบ้าน ถนน พื้นที่สีเขียว และพื้นที่เชิงพาณิชย์ """
    
    # ✅ กรณีที่ Grid เป็น None หรือว่าง
    if grid is None or len(grid) == 0 or len(grid[0]) == 0:
        print(f"⚠️ Error: grid เป็น None หรือว่าง! กำลังใช้ Grid เปล่าแทน...")
        grid_size = 5
        grid = np.full((grid_size, grid_size), '0')

    grid = np.array(grid)
    rows, cols = grid.shape

    # ✅ คำนวณคะแนนพื้นฐาน (Base Score)
    SCORES = {'H': 50, 'R': 20, 'E': 100, 'G': 10}
    base_score = sum(SCORES.get(cell, 0) for row in grid for cell in row)

    # ✅ โบนัส (Bonus)
    bonus = 0
    bonus_details = {}

    # ✅ โบนัส +50 ถ้าบ้าน (H) อยู่ที่ขอบ
    h_positions = np.argwhere(grid == 'H')
    edge_houses = np.sum((h_positions[:, 0] == 0) | (h_positions[:, 0] == rows - 1) |
                          (h_positions[:, 1] == 0) | (h_positions[:, 1] == cols - 1))
    bonus_details["H ติดขอบ"] = edge_houses * 50
    bonus += edge_houses * 50

    # ✅ โบนัสจาก Pattern
    bonus_details["HHH"] = np.sum((grid[:, :-2] == 'H') & (grid[:, 1:-1] == 'H') & (grid[:, 2:] == 'H')) * 100
    bonus_details["RRR"] = np.sum((grid[:, :-2] == 'R') & (grid[:, 1:-1] == 'R') & (grid[:, 2:] == 'R')) * 100
    bonus_details["H-R-H"] = np.sum((grid[:-2, :] == 'H') & (grid[1:-1, :] == 'R') & (grid[2:, :] == 'H')) * 100

    # ✅ เพิ่มโบนัสใหม่
    bonus_details["RR-HH"] = np.sum(
        (grid[:-1, :] == 'R') & (grid[1:, :] == 'R') & 
        (grid[:-1, :] == 'H') & (grid[1:, :] == 'H')
    ) * 100

    bonus_details["HR-HR"] = np.sum(
        (grid[:, :-1] == 'H') & (grid[:, 1:] == 'H') & 
        (grid[:, :-1] == 'R') & (grid[:, 1:] == 'R')
    ) * 100

    for k, v in bonus_details.items():
        bonus += v  

    # ✅ ค่าปรับ (Penalty)
    penalty = 0
    penalty_details = {}

    # ✅ ตรวจสอบว่าบ้าน (`H`) เชื่อมต่อกับถนน (`R`) หรือไม่
    h_neighbors_r = np.any([
        np.roll(grid == 'R', shift, axis=axis)[h_positions[:, 0], h_positions[:, 1]]
        for shift, axis in [(1, 0), (-1, 0), (0, 1), (0, -1)]
    ], axis=0)
    num_h_not_connected = np.count_nonzero(~h_neighbors_r)
    penalty_details["H ไม่ติดถนน"] = -100 * num_h_not_connected
    penalty -= 300 * num_h_not_connected

    # ✅ ตรวจสอบ `E` ที่ไม่ติดถนน `-1000`
    e_positions = np.argwhere(grid == 'E')
    e_neighbors_r = np.any([
        np.roll(grid == 'R', shift, axis=axis)[e_positions[:, 0], e_positions[:, 1]]
        for shift, axis in [(1, 0), (-1, 0), (0, 1), (0, -1)]
    ], axis=0)
    num_e_not_connected = np.count_nonzero(~e_neighbors_r)
    penalty_details["E ไม่ติดถนน"] = -1000 * num_e_not_connected
    penalty -= 1000 * num_e_not_connected

    # ✅ ตรวจสอบค่าปรับใหม่
    penalty_details["HH-RR"] = -np.sum(
        (grid[:-1, :] == 'H') & (grid[1:, :] == 'H') & 
        (grid[:-1, :] == 'R') & (grid[1:, :] == 'R')
    ) * 50

    penalty_details["RH-RH"] = -np.sum(
        (grid[:, :-1] == 'R') & (grid[:, 1:] == 'R') & 
        (grid[:, :-1] == 'H') & (grid[:, 1:] == 'H')
    ) * 50

    penalty += penalty_details["HH-RR"]
    penalty += penalty_details["RH-RH"]

    # ✅ คำนวณคะแนนรวม
    total_score = base_score + bonus + penalty
    print(f"🎯 Debug: คะแนน Grid = {total_score} (Base: {base_score}, Bonus: {bonus}, Penalty: {penalty})")

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

def train_grid_size(grid_size):
    """ ✅ ฝึก AI สำหรับขนาด Grid ที่กำหนด """
    GRID_ROWS, GRID_COLS = grid_size

    # ✅ โหลดหรือสร้าง Grid ใหม่
    grid = load_or_create_grid(GRID_ROWS, GRID_COLS)

    # ✅ ใช้ Parallel Processing ในการฝึกแต่ละ `E`
    best_results = {}
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            e_position = (r, c)
            temp_grid = copy.deepcopy(grid)
            temp_grid[r][c] = 'E'  # ✅ วาง `E` ในตำแหน่งใหม่
            best_results[e_position] = measure_execution_time(train_ai, EPISODES, temp_grid, e_position)

    return best_results

# =========================================================
# 📌 4️⃣ ฟังก์ชันเกี่ยวกับ Q-Table (โหลด, อัปเดต, และส่งไปเซิร์ฟเวอร์)
# =========================================================
def load_q_table():
    """ ✅ โหลด Q-Table จากฐานข้อมูล SQLite อย่างปลอดภัย """
    global q_table
    q_table = {}

    try:
        conn = sqlite3.connect("q_table.db", check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("SELECT state_key, action_key, q_value FROM q_table")
        rows = cursor.fetchall()
        conn.close()

        for state_key, action_key, q_value in rows:
            try:
                state_key = json.loads(state_key)  # ✅ ใช้ JSON แทน eval() เพื่อป้องกันปัญหาความปลอดภัย
                action_key = json.loads(action_key) if isinstance(action_key, str) else action_key

                if isinstance(state_key, list):  # ✅ ถ้าเป็น list ให้แปลงเป็น tuple
                    state_key = tuple(map(tuple, state_key))

                if state_key not in q_table:
                    q_table[state_key] = {}
                q_table[state_key][action_key] = q_value
            except Exception as e:
                print(f"⚠️ [AI] Error แปลง state_key: {state_key} -> {e}")

        print(f"✅ [AI] Q-Table Loaded from SQLite: {len(q_table)} states")

    except sqlite3.Error as e:
        print(f"⚠️ [AI] SQLite Error: {e}")
        q_table = {}  # ถ้าโหลดไม่ได้ให้ใช้ dictionary ว่าง

def update_q_table(state, action, reward, next_state, episode):
    """ ✅ ปรับปรุง Q-Table โดยตรวจสอบ key ก่อนอัปเดต """
    print(f"\n🔹 [STEP] อัปเดต Q-Table (Episode {episode})")

    state_key = tuple(map(tuple, state.tolist() if isinstance(state, np.ndarray) else state))
    next_state_key = tuple(map(tuple, next_state.tolist() if isinstance(next_state, np.ndarray) else next_state))
    action_key = tuple(action) if isinstance(action, (list, np.ndarray)) else action

    alpha = max(ALPHA_END, ALPHA_START / (1 + episode * ALPHA_DECAY_RATE))

    with q_table_lock:
        if state_key not in q_table:
            q_table[state_key] = {}

        if action_key not in q_table[state_key]:
            q_table[state_key][action_key] = 0  

        max_future_q = max(q_table.get(next_state_key, {}).values(), default=0)
        old_q_value = q_table[state_key][action_key]
        new_q_value = (1 - alpha) * old_q_value + alpha * (reward + GAMMA * max_future_q)
        q_table[state_key][action_key] = round(new_q_value, 4)  # ✅ ป้องกันค่าเล็กเกินไป

        print(f"📌 อัปเดต Q-Table: State = {state_key[:30]}... | Action = {action_key} | Old Q = {old_q_value:.4f} → New Q = {q_table[state_key][action_key]:.4f}")

        print(f"🔍 ก่อน Clean: Q-Table มี {len(q_table)} states")
        # ✅ Clean Q-Table ทุก 500 Episodes
        if episode % 500 == 0:
            print(f"🔍 Clean Q-Table ทุก 500 Episodes (Episode {episode})")
            clean_q_table()
        print(f"🔍 หลัง Clean: Q-Table มี {len(q_table)} states")

def send_q_table_to_server():
    """ ✅ ส่ง Q-Table ไปยังเซิร์ฟเวอร์ Flask """
    url = "http://127.0.0.1:5000/update_q_table"

    if not q_table:  
        print("⚠️ [AI] Q-Table ว่างเปล่า ไม่ส่งไปเซิร์ฟเวอร์")
        return

    print(f"📤 [AI] กำลังส่ง Q-Table ไปเซิร์ฟเวอร์ (ตอนนี้มี {len(q_table)} states)")

    with q_table_lock:
        try:
            q_table_serializable = [
                {"state_key": json.dumps(state), "action_key": json.dumps(action), "q_value": q_value}
                for state, actions in q_table.items()
                for action, q_value in actions.items()
            ]
        except Exception as e:
            print(f"⚠️ [AI] ERROR: แปลง Q-Table ไม่สำเร็จ: {e}")
            return

    try:
        response = requests.post(url, json=q_table_serializable, timeout=10)
        response.raise_for_status()
        print(f"✅ [AI] Q-Table Sent | Server Response: {response.json()}")  
    except requests.exceptions.Timeout:
        print("⚠️ [AI] ERROR: Timeout! ไม่สามารถเชื่อมต่อเซิร์ฟเวอร์ได้")
    except requests.exceptions.ConnectionError:
        print("⚠️ [AI] ERROR: ไม่สามารถเชื่อมต่อเซิร์ฟเวอร์ (Connection Error)")
    except requests.exceptions.RequestException as e:
        print(f"⚠️ [AI] ERROR: ไม่สามารถส่ง Q-Table ไปเซิร์ฟเวอร์: {e}")
    except json.JSONDecodeError:
        print("⚠️ [AI] ERROR: ตอบกลับจากเซิร์ฟเวอร์ไม่ใช่ JSON ที่ถูกต้อง")

def update_q_table_server():
    global q_table

    new_q_table = request.get_json()
    episode = request.args.get("episode", type=int)  # ✅ รับค่า episode จาก request

    if not new_q_table:
        return jsonify({"status": "error", "message": "Received empty Q-Table"}), 400

    print(f"🔍 ก่อน Clean: Q-Table มี {len(q_table)} states")
    
    # ✅ Clean Q-Table ทุก 500 Episodes
    if episode and episode % 500 == 0:
        print(f"🔍 Clean Q-Table ทุก 500 Episodes (Episode {episode})")
        clean_q_table()

    print(f"🔍 หลัง Clean: Q-Table มี {len(q_table)} states")

def batch_update_q_table(stop_event):
    global last_q_table_hash
    while not stop_event.is_set():
        time.sleep(30)  # ✅ ปรับเวลาหน่วงเป็น 30 วินาที ลดการล็อกไฟล์บ่อยเกินไป

        with q_table_lock:
            q_table_copy = dict(q_table)

        try:
            q_table_hash = hashlib.md5(json.dumps(q_table_copy, sort_keys=True).encode()).hexdigest()

            if q_table_hash == last_q_table_hash:
                print("⚠️ [Batch Update] Q-Table ไม่เปลี่ยนแปลง ข้ามการบันทึก")
                continue

            print(f"💾 [Batch Update] กำลังบันทึก Q-Table ({len(q_table_copy)} states)")

            with FileLock(LOCK_FILE, timeout=10):  # ✅ เพิ่ม timeout เป็น 10 วินาที
                with open(Q_TABLE_FILE, "w") as f:
                    json.dump(q_table_copy, f)

            last_q_table_hash = q_table_hash
            print(f"✅ [Batch Update] บันทึกเสร็จ ({len(q_table_copy)} states)")

        except TimeoutError:
            print("❌ [Batch Update] ERROR: Timeout acquiring file lock! ป้องกัน Deadlock")
        except Exception as e:
            print(f"❌ [Batch Update] เกิดข้อผิดพลาดในการบันทึก: {e}")

def check_memory_before_training():
    mem = psutil.virtual_memory()
    print(f"🛑 Memory Usage: {mem.percent}% | Available: {mem.available / 1e6} MB")
    
    if mem.available < 4 * 1024 * 1024 * 1024:  # ✅ RAM เหลือน้อยกว่า 4GB ให้หยุด
        print("❌ RAM เหลือน้อยเกินไป หยุดการฝึก AI")
        exit(1)  # หยุดโปรแกรม

def check_q_table_size():
    """ ✅ ตรวจสอบจำนวน state และขนาดของ Grid ที่ถูกเก็บใน Q-Table """
    conn, cursor = get_db_connection()

    cursor.execute("SELECT COUNT(*) FROM q_table")
    num_states = cursor.fetchone()[0]

    cursor.execute("SELECT DISTINCT state_key FROM q_table")
    state_keys = cursor.fetchall()

    grid_sizes = set()
    for key in state_keys:
        try:
            grid_size = eval(key[0])  # ✅ แปลง string กลับเป็น tuple
            grid_sizes.add(grid_size)
        except:
            pass  # ข้าม state ที่ผิดปกติ

    conn.close()  # ✅ ปิด Connection หลังจากดึงข้อมูล

    print(f"📌 Q-Table มีทั้งหมด {num_states} states")
    print(f"📌 Grid ที่บันทึกไว้มีขนาด: {sorted(grid_sizes) if grid_sizes else 'ยังไม่มีข้อมูล'}")

def check_q_table():
    conn, cursor = get_db_connection()
    cursor.execute("SELECT COUNT(*) FROM q_table")
    num_records = cursor.fetchone()[0]
    conn.close()
    print(f"📌 Q-Table มีทั้งหมด {num_records} records")

def print_grid(grid):
    """ ✅ แสดงผล Grid เป็นตัวอักษรธรรมดา โดยไม่มีการจัดระยะห่างพิเศษ """
    for row in grid:
        print(" ".join(row))  # ✅ ใช้ " " คั่นระหว่างตัวอักษร
    print()  # ✅ เพิ่มบรรทัดเว้นระหว่าง Grid แต่ละรอบ

def get_db_connection():
    """ ✅ จัดการ Database Connection ป้องกันการเปิดซ้ำ """
    conn = sqlite3.connect("q_table.db", check_same_thread=False)
    cursor = conn.cursor()
    return conn, cursor

def clean_q_table():
    """ ✅ ลบค่าที่ต่ำกว่าค่า threshold ออกจาก Q-Table โดยแยกตามขนาดของ Grid """
    global q_table  

    if not q_table:
        print("⚠️ Q-Table ว่างเปล่า! ไม่มีอะไรต้องล้าง")
        return

    # ✅ 1. แยก Q-values ตามขนาดของกริด
    grid_q_values = {}
    
    for state, actions in q_table.items():
        grid_size = (len(state), len(state[0]))  # ขนาดของ Grid
        max_q_value = max(actions.values(), default=0)
        
        if grid_size not in grid_q_values:
            grid_q_values[grid_size] = []
        grid_q_values[grid_size].append(max_q_value)

    # ✅ 2. คำนวณ threshold แยกตามขนาดของ Grid
    thresholds = {
        grid_size: np.percentile(np.array(q_values), 35) if q_values else float('-inf')
        for grid_size, q_values in grid_q_values.items()
    }

    # ✅ 3. ลบค่าที่ต่ำกว่า threshold ของขนาด Grid นั้น ๆ
    q_table_keys = list(q_table.keys())  # ✅ ป้องกันการลบ key ระหว่าง loop
    deleted_count = 0  # นับจำนวน states ที่ถูกลบ

    for key in q_table_keys:
        grid_size = (len(key), len(key[0]))  # ขนาดของ Grid
        threshold = thresholds.get(grid_size, float('-inf'))
        q_values = np.fromiter(q_table[key].values(), dtype=float)

        if np.mean(q_values) < threshold:  # ✅ ใช้ mean แทน max
            del q_table[key]
            deleted_count += 1  # ✅ เพิ่มตัวนับ states ที่ถูกลบ

    print(f"✅ Q-Table Cleaned: ลบ {deleted_count} states ที่ต่ำกว่าค่า threshold ของแต่ละขนาด, เหลือ {len(q_table)} states")

# =========================================================
# 📌 5️⃣ ฟังก์ชันเกี่ยวกับการเลือก Action
# =========================================================
def get_state(grid, r, c, size=3):
    """
    ✅ ดึงแพทเทิร์นขนาด `size x size` รอบจุดที่วาง
    ✅ ถ้าตำแหน่งอยู่นอกกระดาน ให้ใส่ 'X'
    """
    half = size // 2
    state = []
    for i in range(-half, half+1):
        row = []
        for j in range(-half, half+1):
            nr, nc = r + i, c + j
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
                row.append(grid[nr][nc])
            else:
                row.append("X")  # ✅ ใส่ 'X' แทนพื้นที่นอกกระดาน
        state.append(tuple(row))
    return tuple(state)

def choose_action(grid, e_position, epsilon=0.1):
    """ ✅ เลือกจุดวางแบบ BFS (จิ๊กซอ) และเลือกอาคารที่ให้โบนัสแพทเทิร์นสูงสุด """
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid)

    rows, cols = grid.shape
    visited = set()
    queue = deque([e_position])
    build_order = []

    while queue:
        r, c = queue.popleft()
        if (r, c) in visited:
            continue
        visited.add((r, c))

        if grid[r, c] == '0':
            build_order.append((r, c))

        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                queue.append((nr, nc))

    if not build_order:
        return None  # ถ้ากริดเต็มแล้ว ไม่ต้องเลือก action ใหม่

    # ✅ ถ้าอยู่ในช่วง Exploration (สุ่มเลือก)
    if random.random() < epsilon:
        r, c = random.choice(build_order)
        chosen = random.choices(['H', 'R', 'G'], weights=[0.5, 0.3, 0.2])[0]  # ✅ ปรับความน่าจะเป็น
        return r, c, chosen

    # ✅ ใช้คะแนนจาก Q-Table หรือ Reward System ในการเลือกอาคาร
    best_score = float('-inf')
    best_choice = None
    best_position = None

    for r, c in build_order:
        for option in ['H', 'R', 'G']:
            temp_grid = grid.copy()
            temp_grid[r, c] = option  # ✅ ทดลองวาง

            # ✅ ลดการเรียก calculate_reward_verbose() บ่อยเกินไป
            score = calculate_reward_verbose(temp_grid)  

            if score > best_score:
                best_score = score
                best_choice = option
                best_position = (r, c)

    return best_position[0], best_position[1], best_choice if best_choice else random.choice(['H', 'R', 'G'])

def validate_final_grid(grid):
    """ ✅ ตรวจสอบว่าผังหมู่บ้านถูกต้องตามเงื่อนไข """
    grid = np.array(grid)

    # ✅ ตรวจสอบว่ามีถนนอย่างน้อย 1 เส้น
    roads_exist = np.sum(grid == 'R') > 0
    if not roads_exist:
        return False

    # ✅ ตรวจสอบว่าบ้านทุกหลังต้องติดถนน หรืออยู่ห่างไม่เกิน 1 ช่อง
    houses = np.argwhere(grid == 'H')
    for r, c in houses:
        if not ((r > 0 and grid[r-1, c] == 'R') or 
                (r < grid.shape[0]-1 and grid[r+1, c] == 'R') or
                (c > 0 and grid[r, c-1] == 'R') or 
                (c < grid.shape[1]-1 and grid[r, c+1] == 'R') or
                (r > 1 and grid[r-2, c] == 'R') or
                (r < grid.shape[0]-2 and grid[r+2, c] == 'R') or
                (c > 1 and grid[r, c-2] == 'R') or 
                (c < grid.shape[1]-2 and grid[r, c+2] == 'R')):
            return False  # ถ้าพบบ้านที่ไม่ติดถนน ให้คืนค่า False

    return True  # ✅ ถ้าผ่านทุกเงื่อนไข ถือว่า Grid ใช้งานได้

def get_empty_cells(grid):
    """ ✅ คืนตำแหน่งที่ยังเป็น '0' (ช่องว่าง) ใน grid """
    return [(r, c) for r in range(len(grid)) for c in range(len(grid[0])) if grid[r][c] == '0']

# =========================================================
# 📌 6️⃣ ฟังก์ชัน Train AI
# =========================================================
def get_neighbors(grid, r, c, visited=None):
    """ ✅ ดึงตำแหน่งรอบๆ ที่สามารถใช้ได้ """
    if grid is None:
        print(f"⚠️ Error: grid เป็น None ใน get_neighbors()! กำลังใช้ Grid เปล่าแทน...")
        return []

    if visited is None:
        visited = set()

    neighbors = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and (nr, nc) not in visited:
            if grid[nr][nc] == '0':
                neighbors.append((nr, nc))

    return neighbors

def train_ai(episodes, grid, e_position):
    """ ✅ AI ฝึกการเรียนรู้ และบังคับให้วางอาคารให้เต็ม Grid อย่างมีประสิทธิภาพ """
    
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid)  # ✅ แปลงเป็น numpy array ถ้ายังไม่ใช่

    best_grid = None
    best_score = float('-inf')
    episode_scores = []  # เพิ่มตัวแปรเก็บคะแนนแต่ละ episode

    print("\n" + "="*50)
    print("🎮 เริ่มการฝึก AI...")
    print(f"📊 จำนวน Episodes: {episodes}")
    print(f"📐 ขนาด Grid: {grid.shape}")
    print(f"📍 ตำแหน่ง E: {e_position}")
    print("="*50 + "\n")

    for episode in range(episodes):
        print(f"\n🔹 Episode {episode + 1}/{episodes}")
        state = grid.copy()  # ✅ ใช้ copy() แทน deepcopy() เพื่อลดเวลา

        # ✅ ถ้า E ถูกเขียนทับ ให้แน่ใจว่าอยู่ที่ตำแหน่งเดิม
        if state[e_position] != 'E':
            state[e_position] = 'E'  

        empty_cells = np.argwhere(state == '0')  # ✅ ใช้ NumPy หา index ช่องว่าง
        np.random.shuffle(empty_cells)  # ✅ สุ่มตำแหน่งให้เร็วขึ้น

        placed_buildings = {'H': 0, 'R': 0, 'G': 0}  # นับจำนวนอาคารที่วาง

        for r, c in empty_cells:
            action = choose_action(state, e_position)  # ✅ เลือก action

            if action:
                r, c, char = action
            else:
                char = np.random.choice(['H', 'R', 'G'], p=[0.5, 0.3, 0.2])  # ✅ ปรับอัตราสุ่ม

            state[r, c] = char  # ✅ วางอาคาร
            placed_buildings[char] += 1  # เพิ่มจำนวนอาคารที่วาง
            next_state = state.copy()  # ✅ ใช้ copy() แทน deepcopy()
            reward = calculate_reward_verbose(state)
            update_q_table(state, (r, c, char), reward, next_state, episode)

        total_reward = calculate_reward_verbose(state)
        episode_scores.append(total_reward)  # เก็บคะแนนของ episode นี้

        # แสดงผลแบบสรุป
        print(f"EP {episode + 1} E{e_position} คะแนน Grid = {total_reward} (Base: {total_reward - reward}, Bonus: {reward}, Penalty: {total_reward - reward - reward})")

        # แจ้งเตือนกรณีสำคัญ
        if total_reward < -1000:
            print(f"⚠️ แจ้งเตือน: คะแนนต่ำมาก! ({total_reward})")
        elif total_reward > 0:
            print(f"✅ แจ้งเตือน: คะแนนดี! ({total_reward})")

        if placed_buildings['R'] == 0:
            print(f"⚠️ แจ้งเตือน: ไม่มีการวางถนน!")
        elif placed_buildings['H'] == 0:
            print(f"⚠️ แจ้งเตือน: ไม่มีการวางบ้าน!")
        elif placed_buildings['G'] == 0:
            print(f"⚠️ แจ้งเตือน: ไม่มีการวางพื้นที่สีเขียว!")

        if total_reward > best_score or best_grid is None:
            best_score = total_reward
            best_grid = state.copy()
            print(f"   🎯 คะแนนใหม่สูงสุด: {best_score}")

        # ✅ ส่ง Q-Table ไปเซิร์ฟเวอร์ทุก 10 EPISODES แทนที่จะส่งทุกครั้ง
        if episode % 10 == 0:
            send_q_table_to_server()

    # ✅ แสดงผลสรุปการฝึกทั้งหมด
    print("\n" + "="*50)
    print("📊 สรุปการฝึกทั้งหมด:")
    print(f"   - จำนวน Episodes: {episodes}")
    print(f"   - คะแนนเฉลี่ย: {sum(episode_scores)/len(episode_scores):.2f}")
    print(f"   - คะแนนสูงสุด: {max(episode_scores)}")
    print(f"   - คะแนนต่ำสุด: {min(episode_scores)}")
    print(f"   - คะแนนสูงสุดตลอดการฝึก: {best_score}")
    print("\n🎯 Grid ที่ดีที่สุด:")
    for row in best_grid:
        print("     " + " ".join(row))
    print("="*50 + "\n")

    return best_grid, best_score

def train_grid(grid_size, epsilon):
    """ ✅ ฝึก AI สำหรับขนาด Grid ที่กำหนด """
    rows, cols = grid_size
    e_positions = get_edge_positions(rows, cols)  # ✅ ต้องหาตำแหน่ง `E` ก่อน

    best_results = {}
    for e_position in e_positions:
        grid = load_or_create_grid(rows, cols, e_position)  # ✅ เพิ่ม `e_position` ตอนโหลด Grid
        best_results[e_position] = train_ai(EPISODES, grid, e_position)

    return best_results

def train_grid_wrapper(size, epsilon):
    return train_grid(size, epsilon)

def train_ai_parallel(grid_sizes):
    num_workers = min(4, cpu_count() // 2)  # ✅ จำกัดให้ใช้ไม่เกิน 4 Core
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.starmap(train_grid_wrapper, [(size, EPSILON_START) for size in grid_sizes])

    best_results = {}

    for size, result in zip(grid_sizes, results):
        best_results[size] = {}

        for e_position, grid_info in result.items():
            if isinstance(grid_info, tuple) and len(grid_info) == 2:
                best_grid, best_score = grid_info

                if best_grid is None or not isinstance(best_grid, (list, np.ndarray)):
                    print(f"⚠️ Warning: best_grid เป็น None, กำลังสร้างใหม่...")
                    best_grid = [["H" if (i+j) % 3 == 0 else "R" if (i+j) % 3 == 1 else "G" for j in range(size[1])] for i in range(size[0])]

                if any('0' in row for row in best_grid):
                    print(f"⚠️ Warning: best_grid มีช่องว่าง, กำลังแก้ไข...")
                    best_grid = np.where(np.array(best_grid) == '0', np.random.choice(['H', 'R', 'G']), best_grid)

                if best_grid[e_position[0]][e_position[1]] != 'E':
                    print(f"⚠️ Warning: best_grid ไม่มี `E`, กำลังแก้ไข...")
                    best_grid[e_position[0]][e_position[1]] = 'E'

                best_results[size][e_position] = (best_grid, best_score)

    return best_results

# =========================================================
# 📌 7️⃣ ฟังก์ชันวัดเวลา (สำหรับ Debugging)
# =========================================================
def measure_execution_time(function, *args, **kwargs):
    start_time = time.time()
    result = function(*args, **kwargs)  # ✅ ใช้ *args เพื่อรองรับ argument ที่แตกต่างกัน
    elapsed_time = time.time() - start_time
    print(f"\n⏳ ใช้เวลาทั้งหมด: {elapsed_time:.2f} วินาที")
    return result, elapsed_time

# =========================================================
# 📌 8️⃣ ส่วน `main()` สำหรับรันโปรแกรม
# =========================================================

if __name__ == "__main__":
    start_time = time.perf_counter()  # ⏳ เริ่มจับเวลา

    check_memory_before_training()  # ✅ ตรวจสอบหน่วยความจำก่อนฝึก AI

    # ✅ โหลด Q-Table ก่อน ถ้าไม่มีให้สร้างใหม่
    load_q_table()
    if not q_table:
        print("⚠️ Q-Table ว่างเปล่า! AI ต้องเริ่มเรียนรู้ใหม่")
        q_table = {}
    else:
        print(f"✅ Q-Table Loaded: มี {len(q_table)} states")

    # ✅ Clean Q-Table ก่อนฝึก
    print(f"🔍 ก่อน Clean: Q-Table มี {len(q_table)} states")
    clean_q_table()
    print(f"🔍 หลัง Clean: Q-Table มี {len(q_table)} states")

    # ✅ ฝึก AI และจับเวลาฝึก
    print("\n📌 Best Layouts for Each Grid Size:")
    train_start_time = time.perf_counter()
    raw_results = train_ai_parallel(grid_sizes)  # ✅ ฝึก AI ที่นี่ครั้งเดียว
    train_end_time = time.perf_counter()
    train_time = train_end_time - train_start_time  # ⏳ เวลาที่ใช้ฝึก AI

    # ✅ ตรวจสอบว่า `train_ai_parallel` คืนค่าถูกต้อง
    if not isinstance(raw_results, dict):
        print(f"⚠️ Error: train_ai_parallel() คืนค่าผิดพลาด -> {raw_results}")
        raw_results = {size: {} for size in grid_sizes}

    best_results = {}
    for grid_size, result in raw_results.items():
        if isinstance(result, dict):
            best_results[grid_size] = result
        else:
            print(f"⚠️ Warning: train_ai_parallel คืนค่าผิดปกติสำหรับ Grid {grid_size} -> {result}")
            best_results[grid_size] = {}

    print(f"🔍 Debug: best_results = {best_results}")

    # ✅ ถ้า `best_results` ว่างเปล่า ให้สร้างใหม่
    if not best_results:
        print("⚠️ Warning: best_results ว่างเปล่า! กำลังสร้างใหม่...")
        best_results = {size: {} for size in grid_sizes}

    # ✅ ตรวจสอบและแสดงผล Best Layouts
    for grid_size, best_layout_dict in best_results.items():
        if not isinstance(best_layout_dict, dict):
            print(f"⚠️ Warning: best_results[{grid_size}] มีค่าผิดปกติ -> {best_layout_dict}")
            continue

        print(f"\n🔹 Best Layouts for Grid {grid_size}:")

        for e_position, value in best_layout_dict.items():
            if not isinstance(value, tuple) or len(value) != 2:
                print(f"⚠️ Warning: best_results[{grid_size}][{e_position}] มีค่าผิดปกติ -> {value}")
                continue

            best_grid, best_score = value

            # ✅ ถ้า best_grid เป็น None หรือไม่มีข้อมูล ให้สร้างใหม่
            if best_grid is None or not isinstance(best_grid, (list, np.ndarray)) or any('0' in row for row in best_grid):
                print(f"⚠️ Warning: best_grid ไม่มีข้อมูลหรือยังมีที่ว่าง, กำลังแก้ไข...")
                best_grid = [["H" if (i+j) % 3 == 0 else "R" if (i+j) % 3 == 1 else "G" for j in range(grid_size[1])] for i in range(grid_size[0])]

            # ✅ ถ้า best_grid ไม่มี `E` ให้ใส่กลับไป
            if best_grid[e_position[0]][e_position[1]] != 'E':
                print(f"⚠️ Warning: best_grid ไม่มี `E`, กำลังแก้ไข...")
                best_grid[e_position[0]][e_position[1]] = 'E'

            e_position_1based = (e_position[0] + 1, e_position[1] + 1)  # ✅ เปลี่ยนเป็น index ที่อ่านง่ายขึ้น
            print(f"\n📌 Grid {grid_size}, E at {e_position_1based} | Score: {int(best_score)}")

            # ✅ แปลง grid เป็น string อย่างปลอดภัย
            try:
                best_grid_str = "\n".join([" ".join(map(str, row)) for row in best_grid])
            except Exception as e:
                print(f"⚠️ Error: ไม่สามารถแปลง Grid เป็น String ได้ -> {e}")
                print(f"🔧 กำลังใช้ Grid เปล่าแทน")
                best_grid = [["0" for _ in range(grid_size[1])] for _ in range(grid_size[0])]
                best_grid_str = "\n".join([" ".join(map(str, row)) for row in best_grid])

            print(best_grid_str)

    # ✅ ตรวจสอบ Q-Table หลังจากฝึก AI เสร็จ
    check_start_time = time.perf_counter()
    check_q_table()
    check_end_time = time.perf_counter()
    check_time = check_end_time - check_start_time  # ⏳ เวลาที่ใช้ในการตรวจสอบ Q-Table

    # ✅ แสดงเวลาทั้งหมด
    end_time = time.perf_counter()
    total_time = end_time - start_time  # ⏳ คำนวณเวลาทั้งหมด

    print(f"\n⏳ เวลาที่ใช้ทั้งหมด: {total_time:.2f} วินาที")
    print(f"🚀 เวลาที่ใช้ฝึก AI: {train_time:.2f} วินาที")
    print(f"📊 เวลาที่ใช้ตรวจสอบ Q-Table: {check_time:.2f} วินาที")
