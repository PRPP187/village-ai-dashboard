import streamlit as st
import numpy as np
import pandas as pd
import random
import io
import sys
from jecsun import initialize_grid, load_or_initialize_grid, train_ai, apply_house_types, analyze_profit, GRID_ROWS, GRID_COLS, E_START_POSITION, EPISODES, csv_folder, measure_execution_time

st.set_page_config(page_title="AI ผังหมู่บ้าน", layout="wide")
st.title("\U0001F3D8️ AI วางผังหมู่บ้านอัตโนมัติด้วย Q-Learning")

# --- Sidebar ---
st.sidebar.header("\U0001F527 ตั้งค่าก่อนเริ่ม")
rows = st.sidebar.slider("จำนวนแถว (rows)", 3, 10, GRID_ROWS)
cols = st.sidebar.slider("จำนวนคอลัมน์ (cols)", 3, 10, GRID_COLS)
e_row = st.sidebar.number_input("ตำแหน่งแถวของ E (1-based)", 1, rows, E_START_POSITION[0])
e_col = st.sidebar.number_input("ตำแหน่งคอลัมน์ของ E (1-based)", 1, cols, E_START_POSITION[1])
e_position = (e_row, e_col)
green_ratio_min = st.sidebar.slider("สัดส่วนพื้นที่สีเขียวขั้นต่ำ (%)", 0, 50, 10)

# --- ฟังก์ชันช่วยแสดง Grid ---
def render_colored_grid(grid, title):
    st.subheader(title)
    color_map = {
        'E': '#FFD700',  # เหลืองทอง
        'R': '#A9A9A9',  # เทา
        'G': '#98FB98',  # เขียวอ่อน
        'H': '#FFB6C1',  # ชมพู
        'H1': '#FFA07A',
        'H2': '#F08080',
        'H3': '#FA8072',
        'H4': '#E9967A',
        '0': '#F0F0F0',
    }
    html = "<table style='border-collapse: collapse;'>"
    for row in grid:
        html += "<tr>"
        for cell in row:
            color = color_map.get(cell, '#FFFFFF')
            html += f"<td style='border: 1px solid black; background-color: {color}; width: 40px; height: 40px; text-align: center;'>{cell}</td>"
        html += "</tr>"
    html += "</table>"
    st.markdown(html, unsafe_allow_html=True)

# --- ปุ่มเริ่มฝึก AI ---
if st.sidebar.button("\U0001F680 เริ่มฝึก AI"):
    with st.spinner("กำลังโหลดหรือสร้าง Grid..."):
        grid, new_e = initialize_grid(rows, cols, e_position)
        grid, _ = load_or_initialize_grid(csv_folder, rows, cols, new_e)
    
    st.success(f"โหลด Grid ขนาด {rows}x{cols} เรียบร้อยแล้ว | E ที่ {new_e}")
    render_colored_grid(grid, "\U0001F4CC แผนผังเริ่มต้น (ก่อนฝึก AI)")

    with st.spinner("⏳ กำลังฝึก AI..."):
        best_grid, best_score, exec_time = measure_execution_time(train_ai, EPISODES, grid)

    final_grid = apply_house_types([row[:] for row in best_grid])

    render_colored_grid(best_grid, "\U0001F3C6 ผังที่ดีที่สุดที่ AI หาได้")
    st.success(f"คะแนนสูงสุด: {best_score}")
    st.info(f"⏱️ ใช้เวลาฝึกทั้งหมด: {exec_time:.2f} วินาที")

    render_colored_grid(final_grid, "\U0001F4CC ผังหลังวาง H1–H4")

    st.subheader("\U0001F4CA วิเคราะห์ผลกำไร")
    buffer = io.StringIO()
    sys.stdout = buffer
    analyze_profit(final_grid)
    sys.stdout = sys.__stdout__
    st.text(buffer.getvalue())

    # --- กราฟคะแนนต่อรอบ (mock data) ---
    st.subheader("\U0001F4C8 กราฟคะแนนต่อรอบ")
    reward_data = pd.DataFrame({
        'Episode': list(range(1, 51)),
        'Score': [random.randint(200, best_score) for _ in range(50)]
    })
    st.line_chart(reward_data.set_index('Episode'))

    st.balloons()
else:
    st.info("👈 กรุณากำหนดค่าและกด 'เริ่มฝึก AI'")
