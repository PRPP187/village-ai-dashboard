"""
ระบบสำรองข้อมูลอัตโนมัติ
"""

import os
import shutil
import datetime
import logging
from config import PATHS, SYSTEM_CONFIG

def create_backup():
    """
    สร้างสำรองข้อมูล Q-Table และฐานข้อมูล
    """
    try:
        # สร้างชื่อไฟล์สำรองพร้อมเวลา
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # สร้างโฟลเดอร์สำรองถ้ายังไม่มี
        if not os.path.exists(PATHS['BACKUPS']):
            os.makedirs(PATHS['BACKUPS'])
        
        # สำรองไฟล์ Q-Table
        q_table_backup = os.path.join(PATHS['BACKUPS'], f"q_table_{timestamp}.json")
        if os.path.exists(SYSTEM_CONFIG['Q_TABLE_FILE']):
            shutil.copy2(SYSTEM_CONFIG['Q_TABLE_FILE'], q_table_backup)
            logging.info(f"สำรอง Q-Table เรียบร้อย: {q_table_backup}")
        
        # สำรองฐานข้อมูล
        db_backup = os.path.join(PATHS['BACKUPS'], f"q_table_{timestamp}.db")
        if os.path.exists(SYSTEM_CONFIG['DB_FILE']):
            shutil.copy2(SYSTEM_CONFIG['DB_FILE'], db_backup)
            logging.info(f"สำรองฐานข้อมูลเรียบร้อย: {db_backup}")
        
        # ลบไฟล์สำรองเก่า (เก็บไว้ 7 วัน)
        cleanup_old_backups()
        
        return True
        
    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดในการสำรองข้อมูล: {str(e)}")
        return False

def cleanup_old_backups():
    """
    ลบไฟล์สำรองที่เก่ากว่า 7 วัน
    """
    try:
        current_time = datetime.datetime.now()
        for filename in os.listdir(PATHS['BACKUPS']):
            filepath = os.path.join(PATHS['BACKUPS'], filename)
            file_time = datetime.datetime.fromtimestamp(os.path.getctime(filepath))
            
            # ถ้าไฟล์เก่ากว่า 7 วัน
            if (current_time - file_time).days > 7:
                os.remove(filepath)
                logging.info(f"ลบไฟล์สำรองเก่า: {filename}")
                
    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดในการลบไฟล์สำรองเก่า: {str(e)}")

def restore_backup(timestamp):
    """
    กู้คืนข้อมูลจากไฟล์สำรอง
    """
    try:
        # กู้คืน Q-Table
        q_table_backup = os.path.join(PATHS['BACKUPS'], f"q_table_{timestamp}.json")
        if os.path.exists(q_table_backup):
            shutil.copy2(q_table_backup, SYSTEM_CONFIG['Q_TABLE_FILE'])
            logging.info(f"กู้คืน Q-Table เรียบร้อย: {q_table_backup}")
        
        # กู้คืนฐานข้อมูล
        db_backup = os.path.join(PATHS['BACKUPS'], f"q_table_{timestamp}.db")
        if os.path.exists(db_backup):
            shutil.copy2(db_backup, SYSTEM_CONFIG['DB_FILE'])
            logging.info(f"กู้คืนฐานข้อมูลเรียบร้อย: {db_backup}")
        
        return True
        
    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดในการกู้คืนข้อมูล: {str(e)}")
        return False

def list_backups():
    """
    แสดงรายการไฟล์สำรองที่มี
    """
    try:
        backups = []
        for filename in os.listdir(PATHS['BACKUPS']):
            filepath = os.path.join(PATHS['BACKUPS'], filename)
            file_time = datetime.datetime.fromtimestamp(os.path.getctime(filepath))
            backups.append({
                'filename': filename,
                'timestamp': file_time.strftime("%Y-%m-%d %H:%M:%S"),
                'size': os.path.getsize(filepath)
            })
        return backups
    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดในการดึงรายการสำรอง: {str(e)}")
        return [] 