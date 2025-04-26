import streamlit as st
import numpy as np
import pandas as pd
import random
import io
import sys
from jecsun import initialize_grid, load_or_initialize_grid, train_ai, apply_house_types, analyze_profit, GRID_ROWS, GRID_COLS, E_START_POSITION, EPISODES, csv_folder

st.set_page_config(page_title="🏘️ AI Village Planner", layout="wide")

# --- Header Section ---
st.markdown("""
    <h1 style='text-align: center; color: #6C63FF;'>🏘️ Smart Village Layout AI</h1>
    <p style='text-align: center; font-size: 18px; color: #999;'>Powered by Q-Learning Algorithm • Real Estate Optimization</p>
    <hr>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.header("🔧 Configuration")
rows = st.sidebar.slider("Number of Rows", 3, 6, GRID_ROWS)
cols = st.sidebar.slider("Number of Columns", 3, 6, GRID_COLS)
e_row = st.sidebar.number_input("E Position (Row, 1-based)", 1, rows, E_START_POSITION[0])
e_col = st.sidebar.number_input("E Position (Col, 1-based)", 1, cols, E_START_POSITION[1])
e_position = (e_row, e_col)

def render_colored_grid(grid, title):
    st.markdown(f"<h3 style='color:#6C63FF;'>{title}</h3>", unsafe_allow_html=True)
    color_map = {
        'E': '#FFD700',   # Entry (Gold)
        'R': '#A9A9A9',   # Road (Grey)
        'G': '#98FB98',   # Green (Light green)
        'H': '#FFB6C1',   # House base (Pink)
        'H1': '#FFA07A',
        'H2': '#F08080',
        'H3': '#FA8072',
        'H4': '#E9967A',
        '0': '#F0F0F0',   # Empty
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

# --- Main AI Training Logic ---
if st.sidebar.button("🚀 Train AI"):
    with st.spinner("Initializing Grid..."):
        grid, new_e = initialize_grid(rows, cols, e_position)
        grid, _ = load_or_initialize_grid(csv_folder, rows, cols, new_e)

    st.success(f"Grid Loaded: {rows}x{cols} | Start Position: {new_e}")
    render_colored_grid(grid, "📌 Initial Layout (Before AI)")

    with st.spinner("🧠 Training AI... Please wait..."):
        best_grid, best_score, action_log = train_ai(EPISODES, grid)

    final_grid = apply_house_types([row[:] for row in best_grid])
    render_colored_grid(best_grid, "🏆 Best Layout Found by AI")
    st.success(f"Best Score Achieved: {best_score}")
    render_colored_grid(final_grid, "📌 Final Layout with House Types (H1–H4)")

    st.subheader("📊 Profitability Analysis")
    buffer = io.StringIO()
    sys.stdout = buffer
    analyze_profit(final_grid)
    sys.stdout = sys.__stdout__
    st.text(buffer.getvalue())

    st.balloons()
else:
    st.info("👈 Please configure settings and press 'Train AI'")
