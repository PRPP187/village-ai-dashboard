import streamlit as st
import numpy as np
import pandas as pd
import random
from jecsuN import initialize_grid, load_or_initialize_grid, train_ai, apply_house_types, analyze_profit, GRID_ROWS, GRID_COLS, E_START_POSITION, EPISODES, csv_folder

st.set_page_config(page_title="AI ผังหมู่บ้าน", layout="wide")
st.title("🏘️ AI วางผังหมู่บ้านอัตโนมัติด้วย Q-Learning")

# --- Sidebar ---
st.sidebar.header("🔧 ตั้งค่าก่อนเริ่ม")
rows = st.sidebar.slider("จำนวนแถว (rows)", 3, 10, GRID_ROWS)
cols = st.sidebar.slider("จำนวนคอลัมน์ (cols)", 3, 10, GRID_COLS)
e_row = st.sidebar.number_input("ตำแหน่งแถวของ E (1-based)", 1, rows, E_START_POSITION[0])
e_col = st.sidebar.number_input("ตำแหน่งคอลัมน์ของ E (1-based)", 1, cols, E_START_POSITION[1])
e_position = (e_row, e_col)

if st.sidebar.button("🚀 เริ่มฝึก AI"):
    with st.spinner("กำลังโหลดหรือสร้าง Grid..."):
        grid, new_e = initialize_grid(rows, cols, e_position)
        grid, _ = load_or_initialize_grid(csv_folder, rows, cols, new_e)

    st.success(f"โหลด Grid ขนาด {rows}x{cols} เรียบร้อยแล้ว | E ที่ {new_e}")

    st.subheader("📌 แผนผังเริ่มต้น (ก่อนฝึก AI)")
    st.dataframe(pd.DataFrame(grid))

    with st.spinner("⏳ กำลังฝึก AI..."):
        best_grid, best_score = train_ai(EPISODES, grid)

    final_grid = apply_house_types([row[:] for row in best_grid])

    st.subheader("🏆 ผังที่ดีที่สุดที่ AI หาได้")
    st.dataframe(pd.DataFrame(best_grid))
    st.success(f"คะแนนสูงสุด: {best_score}")

    st.subheader("📌 ผังหลังวาง H1–H4")
    st.dataframe(pd.DataFrame(final_grid))

    st.subheader("📊 วิเคราะห์ผลกำไร")
    with st.redirect_stdout():
        analyze_profit(final_grid)

    st.balloons()
else:
    st.info("👈 กรุณากำหนดค่าและกด 'เริ่มฝึก AI'")
