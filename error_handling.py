"""
ระบบจัดการข้อผิดพลาด
"""

import logging
import traceback
from functools import wraps

def handle_errors(func):
    """
    Decorator สำหรับจัดการข้อผิดพลาดในฟังก์ชัน
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"เกิดข้อผิดพลาดในฟังก์ชัน {func.__name__}: {str(e)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            return None
    return wrapper

class CustomError(Exception):
    """
    คลาสสำหรับข้อผิดพลาดที่กำหนดเอง
    """
    def __init__(self, message, error_code=None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

def log_error(error, context=None):
    """
    บันทึกข้อผิดพลาดพร้อมบริบท
    """
    error_msg = f"ข้อผิดพลาด: {str(error)}"
    if context:
        error_msg += f"\nบริบท: {context}"
    logging.error(error_msg)
    logging.error(traceback.format_exc())

def validate_input(func):
    """
    Decorator สำหรับตรวจสอบข้อมูลนำเข้า
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # ตรวจสอบ args
            for arg in args:
                if arg is None:
                    raise CustomError("ข้อมูลนำเข้าไม่สามารถเป็น None ได้")
            
            # ตรวจสอบ kwargs
            for key, value in kwargs.items():
                if value is None:
                    raise CustomError(f"พารามิเตอร์ {key} ไม่สามารถเป็น None ได้")
            
            return func(*args, **kwargs)
        except CustomError as e:
            log_error(e, f"ฟังก์ชัน: {func.__name__}")
            return None
        except Exception as e:
            log_error(e, f"ฟังก์ชัน: {func.__name__}")
            return None
    return wrapper 