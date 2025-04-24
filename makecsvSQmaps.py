import os
import csv
import numpy as np
import time

# 1. กำหนดค่าพื้นฐาน
base_folder = r"C:\Users\USER\Desktop\AI_Housing_Project\data\maps\CSV\goodcsv"

# 2. รับค่าจากผู้ใช้ (ให้ป้อนหลายบรรทัด กด Enter เว้นบรรทัดเพื่อจบ)
print(f"📌 กรุณาป้อน Grid (พิมพ์ทีละบรรทัด กด Enter เว้นบรรทัดเพื่อจบ):")
print(f"🔹 ตัวอย่าง:\n  0 | H | 2 | 3\n  4 | 5 | 6 | 7\n  (Enter เว้นบรรทัดเมื่อจบ)")

grid = []
while True:
    row_input = input().strip()
    if not row_input:  # ถ้ากด Enter เว้นบรรทัดให้หยุด
        break
    row_data = [cell.strip() for cell in row_input.replace("|", " ").split()]
    grid.append(row_data)

# 3. ตรวจสอบขนาดของ Grid
rows = len(grid)
cols = len(grid[0]) if rows > 0 else 0

# ตรวจสอบว่าทุกแถวมีจำนวนคอลัมน์เท่ากัน
if not all(len(row) == cols for row in grid):
    print("❌ Error: จำนวนคอลัมน์ในแต่ละแถวไม่เท่ากัน! กรุณาลองใหม่")
    exit()

# 4. แสดง Grid ที่ได้รับ
print(f"\n✅ Grid ที่คุณป้อน ({rows}x{cols}):")
def display_grid(g):
    print("\n".join([" | ".join(row) for row in g]))
    print("\n")

display_grid(grid)

# 5. สร้างโฟลเดอร์ใหม่ตามขนาด Grid และเวลาที่รัน
run_id = time.strftime("%Y%m%d_%H%M%S")
folder_path = os.path.join(base_folder, f"run_{rows}x{cols}_{run_id}")
file_prefix = "grid_"
os.makedirs(folder_path, exist_ok=True)

# 6. ฟังก์ชันหมุนและกลับด้าน Grid
def rotate_grid(g, angle):
    """ หมุน grid ตามมุมที่กำหนด (0°, 90°, 180°, 270°) """
    array = np.array(g, dtype=object)  # ใช้ object รองรับตัวเลขและตัวอักษร
    if angle == 90:
        return np.rot90(array, k=-1).tolist()
    elif angle == 180:
        return np.rot90(array, k=2).tolist()
    elif angle == 270:
        return np.rot90(array, k=1).tolist()
    return g  # หมุน 0° คือเหมือนเดิม

def flip_horizontal(g):
    """ กลับด้านแนวนอน """
    return [row[::-1] for row in g]

def flip_vertical(g):
    """ กลับด้านแนวตั้ง """
    return g[::-1]

# 7. หมุน Grid และเก็บเป็นลิสต์
grids = {
    "0deg": grid,
    "90deg": rotate_grid(grid, 90),
    "180deg": rotate_grid(grid, 180),
    "270deg": rotate_grid(grid, 270),
}

# 8. สร้าง Grid ที่กลับด้าน
flipped_horizontal = {key + "_H": flip_horizontal(g) for key, g in grids.items()}
flipped_vertical = {key + "_V": flip_vertical(g) for key, g in grids.items()}

# รวม Grid ทั้งหมด
all_grids = {**grids, **flipped_horizontal, **flipped_vertical}

# 9. ตรวจสอบและบันทึกเฉพาะ Grid ที่ไม่ซ้ำ
saved_grids = set()
saved_count = 0

for name, g in all_grids.items():
    grid_tuple = tuple(map(tuple, g))  # แปลงเป็น Tuple เพื่อตรวจสอบซ้ำ

    if grid_tuple not in saved_grids:
        saved_grids.add(grid_tuple)
        file_path = os.path.join(folder_path, f"{file_prefix}{name}.csv")
        with open(file_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(g)
        print(f"✅ Saved: {file_path}")
        saved_count += 1
    else:
        print(f"⚠️ Skipped duplicate: {name}")

print(f"\n🎉 บันทึกทั้งหมด {saved_count} ไฟล์ ในโฟลเดอร์ใหม่: {folder_path}")
