import streamlit as st
import numpy as np
import pandas as pd
import random
import io
import sys
from jecsun import initialize_grid, load_or_initialize_grid, train_ai, apply_house_types, analyze_profit, GRID_ROWS, GRID_COLS, E_START_POSITION, EPISODES, csv_folder

# --- Basic Page Config ---
st.set_page_config(page_title="Jecsu AI Village Planner", layout="wide")

# --- Inject Custom CSS ---
st.markdown("""
    <style>
        /* Background Color */
        .stApp {
            background-color: #131314;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #7B83FF 0%, #6A5ACD 100%);
            border-top-right-radius: 20px;
            border-bottom-right-radius: 20px;
        }

        /* Sidebar Elements */
        .sidebar-title {
            font-size: 22px;
            font-weight: bold;
            color: #FFFFFF;
            text-align: center;
            margin-bottom: 20px;
        }

        .sidebar-content {
            font-size: 16px;
            color: #FFFFFF;
        }

        /* Card Style */
        .card {
            background-color: #FFFFFF;
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            margin-bottom: 30px;
        }

        /* Title Style */
        h1, h2, h3 {
            color: #6C63FF;
        }

        /* Button Custom */
        .stButton>button {
            background: linear-gradient(90deg, #7B83FF 0%, #6A5ACD 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6em 2em;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.image("ChatGPT_Image_Apr_26__2025__11_50_45_AM-removebg-preview.png", width=180)  # เปลี่ยนชื่อไฟล์ตามที่มีจริง
    st.markdown('<div class="sidebar-title">Configuration</div>', unsafe_allow_html=True)

    rows = st.slider("Number of Rows", 3, 6, GRID_ROWS)
    cols = st.slider("Number of Columns", 3, 6, GRID_COLS)
    e_row = st.number_input("E Position (Row, 1-based)", 1, rows, E_START_POSITION[0])
    e_col = st.number_input("E Position (Col, 1-based)", 1, cols, E_START_POSITION[1])
    e_position = (e_row, e_col)

    train_ai_clicked = st.button("🚀 Train AI")

# --- Main Content ---
st.markdown('<h1 style="text-align: center;">🏘️ Jecsu AI Village Planner</h1>', unsafe_allow_html=True)

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

# --- AI Process ---
if train_ai_clicked:
    with st.spinner("Initializing Grid..."):
        grid, new_e = initialize_grid(rows, cols, e_position)
        grid, _ = load_or_initialize_grid(csv_folder, rows, cols, new_e)

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.success(f"Grid Loaded: {rows}x{cols} | Start Position: {new_e}")
        render_colored_grid(grid, "📌 Initial Layout (Before AI)")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("🧠 Training AI... Please wait..."):
        best_grid, best_score, action_log = train_ai(EPISODES, grid)

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        render_colored_grid(best_grid, "🏆 Best Layout Found by AI")
        st.success(f"Best Score Achieved: {best_score}")
        st.markdown('</div>', unsafe_allow_html=True)

    final_grid = apply_house_types([row[:] for row in best_grid])

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        render_colored_grid(final_grid, "📌 Final Layout with House Types (H1–H4)")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Profitability Analysis")
        buffer = io.StringIO()
        sys.stdout = buffer
        analyze_profit(final_grid)
        sys.stdout = sys.__stdout__
        st.text(buffer.getvalue())
        st.markdown('</div>', unsafe_allow_html=True)

    st.balloons()
else:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.info("👈 Please configure settings and press 'Train AI' to start.")
        st.markdown('</div>', unsafe_allow_html=True)
