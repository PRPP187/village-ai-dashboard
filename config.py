"""
ไฟล์ตั้งค่าการทำงานของโปรแกรม
เป็นแหล่งข้อมูลหลักสำหรับการตั้งค่าทั้งหมด
"""

import numpy as np
from reward_calculator import get_scores_config

# ตั้งค่าการทำงานของ AI
AI_CONFIG = {
    'EPISODES': 100,  # ลดจำนวน episodes ลงเพื่อให้เห็นการทำงานชัดเจน
    'ALPHA_START': 0.1,
    'ALPHA_END': 0.01,
    'ALPHA_DECAY_RATE': 0.001,
    'GAMMA': 0.9,
    'EPSILON_START': 0.7,
    'EPSILON_END': 0.01,
    'EPSILON_DECAY': 0.95
}

# ตั้งค่าขนาด Grid ที่รองรับ
GRID_SIZES = [(3, 3)]

# ตั้งค่าระบบ
SYSTEM_CONFIG = {
    'MAX_MEMORY_USAGE': 1e9,  # จำกัดการใช้ RAM (1GB)
    'BACKUP_INTERVAL': 3600,  # ตั้งเวลาสำรองข้อมูล (1 ชั่วโมง)
    'LOG_LEVEL': 'INFO',      # ระดับการบันทึก
    'Q_TABLE_FILE': "q_table.json",  # ไฟล์เก็บ Q-Table
    'DB_FILE': "q_table.db"   # ไฟล์ฐานข้อมูล
}

# ตั้งค่าเส้นทางไฟล์
PATHS = {
    'MAPS': "data/maps/CSV/goodcsv",
    'BACKUPS': "data/backups",
    'LOGS': "logs"
}

# สร้างตัวแปรสำหรับใช้ใน jecsu.py
EPISODES = AI_CONFIG['EPISODES']
ALPHA_START = AI_CONFIG['ALPHA_START']
ALPHA_END = AI_CONFIG['ALPHA_END']
ALPHA_DECAY_RATE = AI_CONFIG['ALPHA_DECAY_RATE']
GAMMA = AI_CONFIG['GAMMA']
EPSILON_START = AI_CONFIG['EPSILON_START']
EPSILON_END = AI_CONFIG['EPSILON_END']
EPSILON_DECAY = AI_CONFIG['EPSILON_DECAY']

SCORES = get_scores_config()
grid_sizes = GRID_SIZES
Q_TABLE_FILE = SYSTEM_CONFIG['Q_TABLE_FILE']

def convert_to_hashable(obj):
    """ ✅ แปลง object เป็น hashable type """
    try:
        if isinstance(obj, np.ndarray):
            obj = obj.tolist()
            
        if isinstance(obj, (list, tuple)):
            # ถ้าเป็น nested structure (เช่น tuple ของ tuple)
            if obj and isinstance(obj[0], (list, tuple, np.ndarray)):
                return tuple(tuple(str(x) for x in row) for row in obj)
            # ถ้าเป็น flat structure
            return tuple(str(x) for x in obj)
            
        return str(obj)
        
    except Exception as e:
        print(f"⚠️ [AI] Error ใน convert_to_hashable: {e}")
        print(f"⚠️ [AI] Input type: {type(obj)}")
        if isinstance(obj, (list, tuple, np.ndarray)):
            print(f"⚠️ [AI] Input length: {len(obj)}")
            if len(obj) > 0:
                print(f"⚠️ [AI] First element type: {type(obj[0])}")
        return str(obj)

def update_q_table(state, next_state, action, reward, q_table):
    """ ✅ อัปเดต Q-Table """
    try:
        state_key = convert_to_hashable(state)
        next_state_key = convert_to_hashable(next_state)
        action_key = convert_to_hashable(action)

        # คำนวณค่า Q(s, a) เดิม
        current_q_value = q_table.get((state_key, action_key), 0)

        # คำนวณค่า Q(s', a') สูงสุด
        max_future_q_value = max(q_table.get((next_state_key, a), 0) for a in range(4))

        # คำนวณค่า Q(s, a) ใหม่
        new_q_value = current_q_value + ALPHA_START * (reward + GAMMA * max_future_q_value - current_q_value)

        # อัปเดต Q-Table
        if state_key not in q_table:
            q_table[state_key] = {}
        q_table[state_key][action_key] = round(new_q_value, 4)
    except Exception as e:
        print(f"⚠️ [AI] Error อัปเดต Q-Table: {e}")
        return

def load_or_create_grid(rows, cols):
    """ ✅ สร้าง Grid เปล่า """
    from validation_utils import validate_grid_size
    from error_handling import CustomError
    
    if not validate_grid_size((rows, cols)):
        raise CustomError(f"ขนาด Grid {rows}x{cols} ไม่รองรับ")

    # สร้าง Grid เป็น list ของ string
    grid = [['0' for _ in range(cols)] for _ in range(rows)]
    return grid 