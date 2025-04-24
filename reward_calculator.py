import numpy as np
from collections import deque

# ✅ ตั้งค่าการให้คะแนน (แหล่งข้อมูลหลัก)
SCORES_CONFIG = {
    'E': 50,  # พื้นที่เชิงพาณิชย์
    'G': 15,  # พื้นที่สีเขียว
    'H': 20,  # บ้าน
    'R': 10,  # ถนน
    '0': -50  # พื้นที่ว่าง
}

def get_scores_config():
    """ ✅ คืนค่าตัวแปร SCORES_CONFIG ไปใช้ในไฟล์อื่น """
    return SCORES_CONFIG

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
    SCORES = get_scores_config()
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

    # ✅ โบนัสจาก Pattern
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

    return total_score 