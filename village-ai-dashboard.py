import streamlit as st
import numpy as np
import pandas as pd
import random
import io
import sys
from jecsun import initialize_grid, load_or_initialize_grid, train_ai, apply_house_types, analyze_profits, GRID_ROWS, GRID_COLS, E_START_POSITION, EPISODES, csv_folder

# --- Page Config ---
st.set_page_config(page_title="Jecsu AI Village Planner", layout="wide")

# --- Sidebar ---
with st.sidebar:
    # ✅ ใช้โลโก้ล่าสุดที่คุณอัปโหลด
    st.image("Jecsu logo.png", width=200)
    st.title("Configuration")

    rows = st.slider("Number of Rows", 3, 5, GRID_ROWS)
    cols = st.slider("Number of Columns", 3, 5, GRID_COLS)
    e_row = st.number_input("E Position (Row, 1-based)", 1, rows, E_START_POSITION[0])
    e_col = st.number_input("E Position (Col, 1-based)", 1, cols, E_START_POSITION[1])
    e_position = (e_row, e_col)
    train_ai_clicked = st.button("🚀 Train AI")

# --- Title ---
st.title("Jecsu AI Village Planner")

# --- Grid Rendering ---
def render_colored_grid(grid, title):
    st.subheader(title)
    color_map = {
        'E': '#FFD700', 'R': '#A9A9A9', 'G': '#98FB98', 'H': '#FFB6C1',
        'H1': '#FFA07A', 'H2': '#F08080', 'H3': '#FA8072', 'H4': '#E9967A',
        '0': '#F0F0F0'
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
    with st.spinner("Initializing Grid..."):
        grid, new_e = initialize_grid(rows, cols, e_position)
        grid, _ = load_or_initialize_grid(csv_folder, rows, cols, new_e)

    st.success(f"✅ Grid Loaded: {rows}x{cols} | Start Position: {new_e}")
    render_colored_grid(grid, "📌 Initial Layout (Before AI)")

    with st.spinner("🧠 Training AI... Please wait..."):
        top_layouts, action_log = train_ai(EPISODES, grid)
        best_score, best_grid = top_layouts[0]


    render_colored_grid(best_grid, "🏆 Best Layout Found by AI")
    st.success(f"Best Score Achieved: {best_score}")

    final_grid = apply_house_types([row[:] for row in best_grid])
    render_colored_grid(final_grid, "📌 Final Layout with House Types (H1–H4)")

    st.subheader("📊 Profitability Analysis")
    buffer = io.StringIO()
    sys.stdout = buffer
    analyze_profit(final_grid)
    sys.stdout = sys.__stdout__
    st.text(buffer.getvalue())

    st.success("✅ AI Training Complete. Scroll down to see the optimized layout and analysis.")
else:
    st.info("👈 Please configure settings and press 'Train AI' to start.")
