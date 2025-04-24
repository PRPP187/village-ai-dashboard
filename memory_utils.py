"""
ระบบตรวจสอบและจัดการหน่วยความจำ
"""

import psutil
import logging
from config import SYSTEM_CONFIG

def check_memory_usage():
    """
    ตรวจสอบการใช้หน่วยความจำ
    """
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        # แปลงเป็น MB
        memory_used = memory_info.rss / (1024 * 1024)
        memory_limit = SYSTEM_CONFIG['MAX_MEMORY_USAGE'] / (1024 * 1024)
        
        logging.info(f"การใช้หน่วยความจำ: {memory_used:.2f}MB / {memory_limit:.2f}MB ({memory_percent:.1f}%)")
        
        # ถ้าใช้หน่วยความจำเกินกำหนด
        if memory_info.rss > SYSTEM_CONFIG['MAX_MEMORY_USAGE']:
            logging.warning("การใช้หน่วยความจำสูงเกินไป!")
            return False
            
        return True
        
    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดในการตรวจสอบหน่วยความจำ: {str(e)}")
        return False

def get_system_memory_info():
    """
    ดึงข้อมูลหน่วยความจำของระบบ
    """
    try:
        memory = psutil.virtual_memory()
        return {
            'total': memory.total / (1024 * 1024 * 1024),  # GB
            'available': memory.available / (1024 * 1024 * 1024),  # GB
            'percent': memory.percent,
            'used': memory.used / (1024 * 1024 * 1024),  # GB
            'free': memory.free / (1024 * 1024 * 1024)  # GB
        }
    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดในการดึงข้อมูลหน่วยความจำระบบ: {str(e)}")
        return None

def clear_memory():
    """
    ลดการใช้หน่วยความจำ
    """
    try:
        import gc
        gc.collect()  # เรียก garbage collector
        logging.info("ทำความสะอาดหน่วยความจำเรียบร้อย")
        return True
    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดในการทำความสะอาดหน่วยความจำ: {str(e)}")
        return False 