# village-ai-dashboard.py

import streamlit as st
import numpy as np
import pandas as pd
import io
import sys
from jecsun import initialize_grid, load_or_initialize_grid, train_ai, apply_house_types, GRID_ROWS, GRID_COLS, E_START_POSITION, EPISODES, csv_folder, HOUSE_PRICES

# --- Page Config ---
st.set_page_config(page_title="Jecsu AI Village Planner", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.image("Jecsu logo.png", width=200)
    st.title("Configuration")

    rows = st.slider("Number of Rows", 3, 5, GRID_ROWS)
    cols = st.slider("Number of Columns", 3, 5, GRID_COLS)
    e_row = st.number_input("E Position (Row, 1-based)", 1, rows, E_START_POSITION[0])
    e_col = st.number_input("E Position (Col, 1-based)", 1, cols, E_START_POSITION[1])
    e_position = (e_row, e_col)
    train_ai_clicked = st.button("Train AI")

# --- Title Section ---
st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h1 style='color:#F8F8F8;'>🏘️ Jecsu AI Village Planner</h1>
        <p style='color:#BBBBBB; font-size:18px;'>
            Optimize village layouts intelligently using Q-Learning.<br>
            Design smarter. Build better. Profit more.
        </p>
    </div>
""", unsafe_allow_html=True)

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

# --- Profit Analysis ---
def analyze_profit(grid):
    summary = {k: 0 for k in HOUSE_PRICES}
    total_cost = total_sale = total_size = weighted_profit = 0

    for row in grid:
        for cell in row:
            if cell in HOUSE_PRICES:
                info = HOUSE_PRICES[cell]
                summary[cell] += 1
                total_cost += info['cost']
                total_sale += info['sale']
                total_size += info['size']
                weighted_profit += (info['sale'] - info['cost']) * info['weight']

    total_profit = total_sale - total_cost
    avg_profit_per_sqm = total_profit / total_size if total_size else 0

    st.subheader("📊 Profit Analysis")
    for htype, count in summary.items():
        if count:
            info = HOUSE_PRICES[htype]
            st.markdown(f"🏠 **{htype}**: {count} units | Cost/unit: {info['cost']:,} | Sale/unit: {info['sale']:,}")

    st.markdown(f"**💸 Total Construction Cost:** {total_cost:,} Baht")
    st.markdown(f"**💰 Total Revenue:** {total_sale:,} Baht")
    st.markdown(f"**📈 Total Profit:** {total_profit:,} Baht")
    st.markdown(f"**📐 Average Profit per sqm:** {avg_profit_per_sqm:,.2f} Baht/sqm")
    st.markdown(f"**🎯 Weighted Profit:** {weighted_profit:,.2f} Baht")

# --- Main Logic ---
if train_ai_clicked:
    with st.spinner("🧱 Initializing Grid..."):
        grid, new_e_position = initialize_grid(rows, cols, e_position)
        grid, _ = load_or_initialize_grid(csv_folder, rows, cols, new_e_position)

    st.success(f"✅ Grid Loaded: {rows} x {cols} | Entry Position: {new_e_position}")
    render_colored_grid(grid, "📌 Initial Layout (Before Training)")

    with st.spinner("🧠 Training AI... Please wait..."):
        top_layouts, training_time_sec = train_ai(EPISODES, grid)

    best_score, best_grid = top_layouts[0]

    st.success(f"✅ Training Completed in {training_time_sec:.2f} seconds.")

    # --- Show Final Result ---
    st.header("🏡 Best Layout Found by AI")

    render_colored_grid(best_grid, "📌 Best Layout (Before House Types)")

    final_layout = apply_house_types([row[:] for row in best_grid])
    render_colored_grid(final_layout, "📌 Final Layout with House Types (H1–H4)")

    analyze_profit(final_layout)

else:
    st.info("👈 Please configure settings and click 'Train AI' to start.")
