import streamlit as st
import numpy as np
import pandas as pd
import random
import io
import sys
from jecsun import initialize_grid, load_or_initialize_grid, train_ai, apply_house_types, analyze_profit, GRID_ROWS, GRID_COLS, E_START_POSITION, EPISODES, csv_folder

# --- Page Config ---
st.set_page_config(page_title="Jecsu AI Village Planner", layout="wide")

# --- Custom Theme CSS ---
st.markdown("""
    <style>
        /* Global Theme */
        .stApp {
            background-color: #1e1e2f;
            color: #F5F7FA;
        }

        html, body, [class*="css"] {
            font-family: 'Segoe UI', sans-serif;
        }

        /* Sidebar Style */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #7B83FF 0%, #4B4C7C 100%);
            border-top-right-radius: 20px;
            border-bottom-right-radius: 20px;
            color: white;
        }

        .sidebar-title {
            font-size: 24px;
            font-weight: bold;
            color: #F5F7FA;
            text-align: center;
            margin-bottom: 20px;
        }

        .stSlider > div[data-testid="stTickBar"] > div,
        .stSlider label,
        .stNumberInput label,
        .stTextInput label {
            color: #FFFFFF !important;
        }

        /* Button */
        .stButton>button {
            background: linear-gradient(90deg, #7B83FF, #6C63FF);
            color: white;
            font-weight: bold;
            padding: 0.6em 2em;
            border: none;
            border-radius: 12px;
        }

        /* Card Container */
        .card {
            background-color: #2a2a3f;
            padding: 30px;
            border-radius: 20px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
            margin-bottom: 30px;
        }

        /* Headers */
        h1, h2, h3 {
            color: #B6A9F2;
        }

        /* Info bar inside card */
        .grid-label {
            background-color: #4B4C7C;
            padding: 10px;
            border-radius: 10px;
            color: white;
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("ChatGPT_Image_Apr_26__2025__11_50_45_AM-removebg-preview.png", width=180)
    st.markdown('<div class="sidebar-title">Configuration</div>', unsafe_allow_html=True)

    rows = st.slider("Number of Rows", 3, 6, GRID_ROWS)
    cols = st.slider("Number of Columns", 3, 6, GRID_COLS)
    e_row = st.number_input("E Position (Row, 1-based)", 1, rows, E_START_POSITION[0])
    e_col = st.number_input("E Position (Col, 1-based)", 1, cols, E_START_POSITION[1])
    e_position = (e_row, e_col)

    train_ai_clicked = st.button("🚀 Train AI")

# --- Title ---
st.markdown('<h1 style="text-align: center;">Jecsu AI Village Planner</h1>', unsafe_allow_html=True)

# --- Grid Rendering ---
def render_colored_grid(grid, title):
    st.markdown(f"<h3 style='color:#B6A9F2;'>{title}</h3>", unsafe_allow_html=True)
    color_map = {
        'E': '#FFD700',   # Entry
        'R': '#A9A9A9',   # Road
        'G': '#98FB98',   # Green
        'H': '#FFB6C1',   # House
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

# --- Main Logic ---
if train_ai_clicked:
    with st.spinner("Initializing Grid..."):
        grid, new_e = initialize_grid(rows, cols, e_position)
        grid, _ = load_or_initialize_grid(csv_folder, rows, cols, new_e)

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="grid-label">✅ Grid Loaded: {rows}x{cols} | Start Position: {new_e}</div>', unsafe_allow_html=True)
        render_colored_grid(grid, "📌 Initial Layout (Before AI)")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("🧠 Training AI... Please wait..."):
        best_grid, best_score, action_log = train_ai(EPISODES, grid)

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        render_colored_grid(best_grid, "🏆 Best Layout Found by AI")
        st.markdown(f'<div class="grid-label">Best Score Achieved: {best_score}</div>', unsafe_allow_html=True)
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
