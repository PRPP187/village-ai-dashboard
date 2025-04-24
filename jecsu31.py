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

# ✅ Import จากไฟล์ใหม่
from memory_utils import check_memory_usage, get_system_memory_info, clear_memory
from error_handling import handle_errors, CustomError, log_error
from backup_utils import create_backup, restore_backup, list_backups
from validation_utils import validate_grid, validate_grid_size, validate_grid_content

# =========================================================
# 📌 2️⃣ กำหนดค่าพื้นฐานสำหรับการทำงานของ AI
# =========================================================
# ✅ ตั้งค่าพื้นฐาน
from config import (
    EPISODES, ALPHA_START, ALPHA_END, ALPHA_DECAY_RATE,
    GAMMA, EPSILON_START, EPSILON_END, EPSILON_DECAY,
    SCORES, grid_sizes, Q_TABLE_FILE
)

SCORES = {'H': 20, 'R': 10, 'E': 50, 'G': 15}  # แก้ไขคะแนนพื้นฐานให้ตรงกับ T.py

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
@handle_errors
def load_or_create_grid(rows, cols, e_position=(1, 1), csv_folder="data/maps/CSV/goodcsv"):
    """ ✅ โหลด Grid จากไฟล์ CSV หรือสร้างใหม่ถ้าไม่มี """
    print(f"\n🔹 [STEP] กำลังโหลด/สร้าง Grid ขนาด {rows}x{cols}, วาง `E` ที่ {e_position}")
    
    # ✅ ตรวจสอบขนาด Grid
    if not validate_grid_size((rows, cols)):
        raise CustomError(f"ขนาด Grid {rows}x{cols} ไม่รองรับ")
    
    csv_files = glob.glob(f"{csv_folder}/**/*.csv", recursive=True)
    best_grid = None
    best_score = float('-inf')

    # ✅ ตรวจสอบไฟล์ CSV ทั้งหมด
    for file in csv_files:
        df = pd.read_csv(file, header=None)
        if df.shape == (rows, cols) and 'E' in df.values:
            grid_from_csv = df.astype(str).values.tolist()
            
            # ✅ ตรวจสอบความถูกต้องของ Grid
            if validate_grid(grid_from_csv):
                score = calculate_reward_verbose(grid_from_csv)
                if score > best_score:
                    best_score = score
                    best_grid = grid_from_csv

    if best_grid is not None:
        print(f"✅ โหลด Grid ที่ดีที่สุดจาก CSV (คะแนน: {best_score})")
        print_grid(best_grid)
        return best_grid

    # ✅ ถ้าไม่มี Grid ที่เหมาะสม สร้างใหม่
    print(f"⚠️ ไม่มี Grid ที่เหมาะสม สร้างใหม่ {rows}x{cols}")
    grid = [['0' for _ in range(cols)] for _ in range(rows)]
    r, c = min(e_position[0], rows-1), min(e_position[1], cols-1)
    grid[r][c] = 'E'
    
    # ✅ ตรวจสอบ Grid ที่สร้างใหม่
    if not validate_grid(grid):
        raise CustomError("ไม่สามารถสร้าง Grid ที่ถูกต้องได้")
    
    print(f"✅ Grid ถูกสร้างขึ้นใหม่ พร้อมวาง `E` ที่ {e_position}")
    print_grid(grid)
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
    SCORES = {'H': 50, 'R': 20, 'E': 100, 'G': 10}  # แก้ไขคะแนนพื้นฐาน
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
    h_positions = np.argwhere(grid == 'H')
    num_h_not_connected = 0
    
    for r, c in h_positions:
        has_road_neighbor = False
        # ตรวจสอบ 4 ทิศทาง
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr,nc] == 'R':
                has_road_neighbor = True
                break
        if not has_road_neighbor:
            num_h_not_connected += 1
    
    penalty_details["H ไม่ติดถนน"] = -300 * num_h_not_connected
    penalty -= 300 * num_h_not_connected

    # ✅ ตรวจสอบ `E` ที่ไม่ติดถนน
    e_positions = np.argwhere(grid == 'E')
    num_e_not_connected = 0
    
    for r, c in e_positions:
        has_road_neighbor = False
        # ตรวจสอบ 4 ทิศทาง
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr,nc] == 'R':
                has_road_neighbor = True
                break
        if not has_road_neighbor:
            num_e_not_connected += 1
    
    penalty_details["E ไม่ติดถนน"] = -1000 * num_e_not_connected
    penalty -= 1000 * num_e_not_connected

    # ✅ ตรวจสอบถนนที่ติดกัน
    r_positions = np.argwhere(grid == 'R')
    num_r_clusters = count_r_clusters(grid)  # นับจำนวนกลุ่มของ R ที่แยกกัน
    
    if num_r_clusters > 1:  # ถ้ามี R มากกว่า 1 กลุ่ม
        penalty_details["R ไม่เชื่อมต่อกัน"] = -500 * num_r_clusters  # เพิ่มค่าปรับตามจำนวนกลุ่ม
        penalty -= 500 * num_r_clusters
    elif num_r_clusters == 0:  # ถ้าไม่มี R เลย
        penalty_details["ไม่มีถนน"] = -1000
        penalty -= 1000

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

    # ✅ ตรวจสอบสัดส่วนพื้นที่สีเขียว (G)
    total_cells = rows * cols
    num_green = np.sum(grid == 'G')
    green_ratio = num_green / total_cells

    if green_ratio < 0.05:
        penalty_details["พื้นที่สีเขียวน้อยเกินไป"] = -500  # ✅ เพิ่มรายละเอียดค่าปรับ
        penalty -= 500  # ✅ ปรับลดคะแนนหากพื้นที่สีเขียวน้อยเกินไป
    if green_ratio > 0.20:
        penalty_details["พื้นที่สีเขียวมากเกินไป"] = -500  # ✅ เพิ่มรายละเอียดค่าปรับ
        penalty -= 500  # ✅ ปรับลดคะแนนหากพื้นที่สีเขียวมากเกินไป

    # ✅ คำนวณคะแนนรวม
    total_score = base_score + bonus + penalty
    print(f"🎯 Debug: คะแนน Grid = {total_score} (Base: {base_score}, Bonus: {bonus}, Penalty: {penalty})")
    print(f"📊 รายละเอียดค่าปรับ: {penalty_details}")  # ✅ แสดงรายละเอียดค่าปรับ

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

def convert_to_hashable(obj):
    """ ✅ แปลง object เป็น hashable type """
    if isinstance(obj, (list, np.ndarray)):
        return tuple(map(tuple, obj.tolist() if isinstance(obj, np.ndarray) else obj))
    return obj

def get_grid_value(grid, r, c):
    """ ✅ ดึงค่าจาก grid โดยรองรับทั้ง numpy array และ list """
    if isinstance(grid, np.ndarray):
        return grid[r, c]
    return grid[r][c]

def update_q_table(state, action, reward, next_state, episode):
    """ ✅ ปรับปรุง Q-Table โดยตรวจสอบ key ก่อนอัปเดต """
    print(f"\n🔹 [STEP] อัปเดต Q-Table (Episode {episode})")

    state_key = convert_to_hashable(state)
    next_state_key = convert_to_hashable(next_state)
    action_key = convert_to_hashable(action)

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
    """ ✅ ตรวจสอบหน่วยความจำก่อนเริ่มการฝึก AI """
    if not check_memory_usage():
        raise CustomError("หน่วยความจำไม่เพียงพอสำหรับการฝึก AI")
    
    memory_info = get_system_memory_info()
    if memory_info:
        print(f"🛑 ข้อมูลหน่วยความจำระบบ:")
        print(f"   - หน่วยความจำทั้งหมด: {memory_info['total']:.2f} GB")
        print(f"   - หน่วยความจำที่ใช้: {memory_info['used']:.2f} GB")
        print(f"   - หน่วยความจำว่าง: {memory_info['free']:.2f} GB")
        print(f"   - ใช้หน่วยความจำ: {memory_info['percent']}%")

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

        if get_grid_value(grid, r, c) == '0':
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
        grid = np.array(grid)

    best_grid = None
    best_score = float('-inf')
    episode_scores = []
    start_time = time.time()

    print("\n" + "="*50)
    print("🎮 เริ่มการฝึก AI...")
    print(f"📊 จำนวน Episodes: {episodes}")
    print(f"📐 ขนาด Grid: {grid.shape}")
    print(f"📍 ตำแหน่ง E: {e_position}")
    print("="*50 + "\n")

    for episode in range(episodes):
        episode_start = time.time()
        print(f"\n🔹 Episode {episode + 1}/{episodes}")
        print(f"⏰ เวลาที่ใช้ไปแล้ว: {time.time() - start_time:.2f} วินาที")
        
        state = grid.copy()
        if state[e_position] != 'E':
            state[e_position] = 'E'

        print("\nสถานะ Grid ปัจจุบัน:")
        print_grid(state)
        time.sleep(0.5)  # หน่วงเวลาให้เห็นการเปลี่ยนแปลง

        empty_cells = np.argwhere(state == '0')
        np.random.shuffle(empty_cells)
        placed_buildings = {'H': 0, 'R': 0, 'G': 0}

        for r, c in empty_cells:
            action = choose_action(state, e_position)
            if action:
                r, c, char = action
            else:
                char = np.random.choice(['H', 'R', 'G'], p=[0.5, 0.3, 0.2])

            state[r, c] = char
            placed_buildings[char] += 1
            
            print(f"\nวางอาคาร {char} ที่ตำแหน่ง ({r+1}, {c+1})")
            print(f"จำนวนอาคารที่วางแล้ว: H={placed_buildings['H']}, R={placed_buildings['R']}, G={placed_buildings['G']}")
            print_grid(state)
            time.sleep(0.1)  # หน่วงเวลาระหว่างการวางแต่ละอาคาร

            next_state = state.copy()
            reward = calculate_reward_verbose(state)
            update_q_table(state, (r, c, char), reward, next_state, episode)

        total_reward = calculate_reward_verbose(state)
        episode_scores.append(total_reward)

        episode_time = time.time() - episode_start
        print(f"\n📊 สรุป Episode {episode + 1}:")
        print(f"⏱️ เวลาที่ใช้: {episode_time:.2f} วินาที")
        print(f"🎯 คะแนน: {total_reward}")
        print(f"🏗️ อาคารที่วาง: H={placed_buildings['H']}, R={placed_buildings['R']}, G={placed_buildings['G']}")

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

        if episode % 10 == 0:
            send_q_table_to_server()
            
        print(f"\nรอ 1 วินาทีก่อนเริ่ม Episode ถัดไป...")
        time.sleep(1)

    total_time = time.time() - start_time
    print("\n" + "="*50)
    print("📊 สรุปการฝึกทั้งหมด:")
    print(f"   - จำนวน Episodes ที่ฝึกจริง: {episodes}")
    print(f"   - เวลาที่ใช้ทั้งหมด: {total_time:.2f} วินาที")
    print(f"   - เวลาเฉลี่ยต่อ Episode: {total_time/episodes:.2f} วินาที")
    print(f"   - คะแนนเฉลี่ย: {sum(episode_scores)/len(episode_scores):.2f}")
    print(f"   - คะแนนสูงสุด: {max(episode_scores)}")
    print(f"   - คะแนนต่ำสุด: {min(episode_scores)}")
    print(f"   - คะแนนสูงสุดตลอดการฝึก: {best_score}")
    print("\n🎯 Grid ที่ดีที่สุด:")
    print_grid(best_grid)
    print("="*50 + "\n")

    return best_grid, best_score

def train_grid(grid_size, epsilon):
    """ ✅ ฝึก AI สำหรับขนาด Grid ที่กำหนด """
    rows, cols = grid_size
    e_positions = get_edge_positions(rows, cols)  # ✅ หาตำแหน่งขอบทั้งหมด

    print(f"\n🔹 กำลังสร้าง Grid ขนาด {rows}x{cols}")
    # แปลงทุกตำแหน่งเป็น 1-based index ก่อนแสดงผล
    e_positions_1based = [(r+1, c+1) for r, c in e_positions]
    print(f"📌 ตำแหน่ง E ที่จะทดสอบ: {e_positions_1based}")

    best_results = {}
    for e_position in e_positions:
        e_position_1based = (e_position[0] + 1, e_position[1] + 1)  # แปลงเป็น 1-base
        print(f"\n📌 ทดสอบตำแหน่ง E ที่ {e_position_1based}")
        
        # สร้างกริดเปล่า
        grid = [['0' for _ in range(cols)] for _ in range(rows)]
        grid[e_position[0]][e_position[1]] = 'E'
        
        # เรียกใช้ train_ai โดยตรง
        best_results[e_position] = train_ai(EPISODES, grid, e_position)

    return best_results

def train_grid_wrapper(size, epsilon):
    return train_grid(size, epsilon)

def train_ai_parallel(grid_sizes):
    """ ✅ ฝึก AI แบบขนาน แต่จำกัดการใช้ CPU """
    num_workers = 1  # ใช้แค่ 1 core เพื่อให้เห็นการทำงานชัดเจน
    print(f"\n🔄 เริ่มการฝึก AI แบบขนาน (ใช้ {num_workers} CPU core)")
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = []
        for size in grid_sizes:
            print(f"\n📐 กำลังฝึก Grid ขนาด {size}...")
            result = pool.apply_async(train_grid_wrapper, (size, EPSILON_START))
            results.append((size, result))
            time.sleep(1)  # รอ 1 วินาทีระหว่างแต่ละ grid size
        
        # รอผลลัพธ์และแสดงความคืบหน้า
        best_results = {}
        for size, result in results:
            print(f"\n⏳ รอผลลัพธ์สำหรับ Grid ขนาด {size}...")
            grid_result = result.get()
            best_results[size] = grid_result
            print(f"✅ เสร็จสิ้นการฝึก Grid ขนาด {size}")
            time.sleep(0.5)  # รอ 0.5 วินาทีก่อนแสดงผลถัดไป

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
    train_time = 0  # กำหนดค่าเริ่มต้น

    try:
        print("\n🚀 เริ่มการทำงานของโปรแกรม...")
        
        # ✅ สร้างสำรองข้อมูลก่อนเริ่มการทำงาน
        create_backup()
        
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
        print("\n📌 เริ่มการฝึก AI สำหรับทุกขนาดกริด...")
        try:
            train_start_time = time.perf_counter()
            raw_results = train_ai_parallel(grid_sizes)  # ✅ ฝึก AI ที่นี่ครั้งเดียว
            train_end_time = time.perf_counter()
            train_time = train_end_time - train_start_time  # ⏳ เวลาที่ใช้ฝึก AI
        except Exception as e:
            log_error(e, "เกิดข้อผิดพลาดในการฝึก AI")

        # ✅ สร้างสำรองข้อมูลหลังจากฝึกเสร็จ
        create_backup()
        
        # ✅ แสดงรายการไฟล์สำรองข้อมูล
        print("\n📁 รายการไฟล์สำรองข้อมูล:")
        backups = list_backups()
        for backup in backups:
            print(f"   - {backup}")

    except Exception as e:
        log_error(e, "เกิดข้อผิดพลาดในส่วน main")
        # ✅ พยายามกู้คืนข้อมูลจากสำรองล่าสุด
        try:
            backups = list_backups()
            if backups:
                latest_backup = backups[0]['filename'].split('_')[1].split('.')[0]
                restore_backup(latest_backup)
                print("✅ กู้คืนข้อมูลจากสำรองล่าสุดสำเร็จ")
            else:
                print("⚠️ ไม่มีไฟล์สำรองข้อมูล")
        except Exception as restore_error:
            log_error(restore_error, "ไม่สามารถกู้คืนข้อมูลได้")

    finally:
        # ✅ แสดงเวลาทั้งหมด
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # ✅ ตรวจสอบ Q-Table และจับเวลา
        check_start_time = time.perf_counter()
        check_q_table()
        check_end_time = time.perf_counter()
        check_time = check_end_time - check_start_time
        
        print(f"\n⏳ เวลาที่ใช้ทั้งหมด: {total_time:.2f} วินาที")
        print(f"🚀 เวลาที่ใช้ฝึก AI: {train_time:.2f} วินาที")
        print(f"📊 เวลาที่ใช้ตรวจสอบ Q-Table: {check_time:.2f} วินาที")
        print("\n✨ จบการทำงานของโปรแกรม")
