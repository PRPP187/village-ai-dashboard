import streamlit as st
import numpy as np
import pandas as pd
import random
import io
import sys
from jecsun import initialize_grid, load_or_initialize_grid, train_ai, apply_house_types, analyze_profit, GRID_ROWS, GRID_COLS, E_START_POSITION, EPISODES, csv_folder

# --- Page Config ---
st.set_page_config(page_title="AI Village Planner", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.image("Jecsu logo.png", width=150)  # <<-- เพิ่มโลโก้ตรงนี้
    st.header("🔧 Configuration Settings")
    rows = st.slider("Number of Rows", 3, 7, GRID_ROWS)
    cols = st.slider("Number of Columns", 3, 7, GRID_COLS)
    e_row = st.number_input("E Position (Row, 1-based)", 1, rows, E_START_POSITION[0])
    e_col = st.number_input("E Position (Column, 1-based)", 1, cols, E_START_POSITION[1])
    e_position = (e_row, e_col)
    train_ai_clicked = st.button("🚀 Train AI")

# --- Main Header ---
st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h1 style='color: #FFFFFF;'>🏘️ AI Village Layout Optimization with Q-Learning</h1>
        <p style='color: #DDDDDD; font-size:18px;'>
            Optimize village layouts intelligently using Q-Learning.<br>
            Design smarter. Build better. Profit more.
        </p>
    </div>
""", unsafe_allow_html=True)

# --- Info Note Box ---
st.markdown("""
    <div style='background-color: #FFF3CD; color: #856404; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
        ⚡ <strong>Note:</strong><br>
        This is a basic demonstration version of the AI. It does <strong>not</strong> save newly generated layouts for future learning.<br><br>
        For the best experience, we recommend using a <strong>3×3 grid size</strong>.<br>
        Using larger grids (4×4 or 5×5) may result in:<br>
        - <strong>Longer training times</strong><br>
        - <strong>Occasional layout inaccuracies</strong><br>
        because the AI retrains from scratch every time, and the current number of episodes may not be sufficient to consistently find optimal layouts.<br><br>
        📐 Additionally, <strong>horizontal layouts</strong> (e.g., 3×4, 3×5) tend to perform better than vertical layouts (e.g., 4×3, 5×3).<br>
        This is because the AI has been trained under the assumption that <strong>houses should face north and south</strong>,<br>
        where the top side of the grid always represents <strong>north</strong>.
    </div>
""", unsafe_allow_html=True)

# --- Grid Rendering ---
def render_colored_grid(grid, title):
    st.subheader(title)
    color_map = {
        'E': '#f8a11d',  # Gold
        'R': '#A9A9A9',  # Gray
        'G': '#217424',  # Green
        'H': '#dd563f',  # Light Pink
        'H1': '#c21c00',
        'H2': '#9d1801',
        'H3': '#801300',
        'H4': '#640f00',
        '0': '#c4c4c4',
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

# --- Main Logic ---
if train_ai_clicked:
    with st.spinner("Loading or creating grid..."):
        grid, new_e = initialize_grid(rows, cols, e_position)
        grid, _ = load_or_initialize_grid(csv_folder, rows, cols, new_e)

    st.success(f"✅ Grid {rows}x{cols} loaded successfully | E Position: {new_e}")
    render_colored_grid(grid, "📌 Initial Layout (Before AI Training)")

    with st.spinner("⏳ Training AI... Please wait..."):
        best_grid, best_score = train_ai(EPISODES, grid)

    final_grid = apply_house_types([row[:] for row in best_grid])

    render_colored_grid(best_grid, "🏆 Best Layout Found by AI")
    st.success(f"🏆 Best Score Achieved: {best_score}")

    render_colored_grid(final_grid, "📌 Final Layout with House Types (H1–H4)")

    st.subheader("📊 Profitability Analysis")
    buffer = io.StringIO()
    sys.stdout = buffer
    analyze_profit(final_grid)
    sys.stdout = sys.__stdout__
    st.text(buffer.getvalue())

    st.snow()

else:
    st.info("👈 Please configure settings and click 'Train AI' to start.")
