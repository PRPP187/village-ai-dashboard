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
        best_grid = None
        best_score = float('-inf')
        top3 = []
        reward_per_episode = []

        for episode in range(EPISODES):
            result, score = train_ai(1, grid)
            reward_per_episode.append(score)

            if score > best_score:
                best_score = score
                best_grid = result
                top3.append((score, [row[:] for row in result]))
                top3 = sorted(top3, key=lambda x: -x[0])[:3]

        exec_time = len(reward_per_episode)

    st.success(f"คะแนนสูงสุด: {best_score}")
    st.info(f"เทรนทั้งหมด {exec_time} รอบ")

    options = [f"อันดับ {i+1} - คะแนน {s}" for i, (s, _) in enumerate(top3)]
    selected = st.selectbox("เลือกผังที่ต้องการแสดง", options)
    selected_grid = top3[options.index(selected)][1]

    render_colored_grid(selected_grid, "\U0001F3C6 ผังที่เลือก")

    final_grid = apply_house_types([row[:] for row in selected_grid])
    render_colored_grid(final_grid, "\U0001F4CC ผังหลังวาง H1–H4")

    st.subheader("\U0001F4CA วิเคราะห์ผลกำไร")
    buffer = io.StringIO()
    sys.stdout = buffer
    analyze_profit(final_grid)
    sys.stdout = sys.__stdout__
    st.text(buffer.getvalue())

    # --- กราฟคะแนนต่อรอบจริง ---
    st.subheader("\U0001F4C8 กราฟคะแนนต่อรอบ")
    reward_data = pd.DataFrame({
        'Episode': list(range(1, exec_time + 1)),
        'Score': reward_per_episode
    })
    st.line_chart(reward_data.set_index('Episode'))

    st.balloons()
else:
    st.info("👈 กรุณากำหนดค่าและกด 'เริ่มฝึก AI'")
