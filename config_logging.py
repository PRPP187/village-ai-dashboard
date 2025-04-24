import logging
import os

def setup_logging():
    """
    ตั้งค่าระบบ logging สำหรับโปรแกรม
    - สร้างไฟล์ log ในโฟลเดอร์ logs
    - ตั้งระดับการบันทึกเป็น INFO
    - กำหนดรูปแบบการบันทึก
    """
    # สร้างโฟลเดอร์ logs ถ้ายังไม่มี
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # ตั้งค่า logging
    logging.basicConfig(
        filename='logs/ai_village.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # เพิ่มการแสดงผลในคอนโซลด้วย
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger('').addHandler(console_handler)
    
    logging.info("ระบบ Logging เริ่มทำงาน") 