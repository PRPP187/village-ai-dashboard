import streamlit as st
import numpy as np
import pandas as pd
import random
import io
import sys
import os
from jecsun import initialize_grid, load_or_initialize_grid, train_ai, apply_house_types, GRID_ROWS, GRID_COLS, E_START_POSITION, EPISODES, csv_folder, HOUSE_PRICES, save_q_table

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
    train_ai_clicked = st.button("\ud83d\ude80 Train AI")

# --- Intro Section ---
st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h1 style='color:#F8F8F8;'>\ud83c\udfe8\ufe0f Jecsu AI Village Planner</h1>
        <p style='color:#BBBBBB; font-size:18px;'>
            An intelligent layout optimization tool powered by Q-Learning.<br>
            Plan roads, green spaces, and house types to maximize profitability and community connectivity.
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

# --- Analyze Profit per Layout ---
def analyze_single_profit(grid):
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

    st.subheader("\ud83d\udcca Profit Analysis")
    for htype, count in summary.items():
        if count:
            info = HOUSE_PRICES[htype]
            st.markdown(f"\ud83c\udfe0 **{htype}**: {count} units | Cost/unit: {info['cost']:,} | Sale/unit: {info['sale']:,}")

    st.markdown(f"**\ud83d\udcb8 Total Construction Cost:** {total_cost:,} Baht")
    st.markdown(f"**\ud83d\udcb0 Total Revenue:** {total_sale:,} Baht")
    st.markdown(f"**\ud83d\udcc8 Total Profit:** {total_profit:,} Baht")
    st.markdown(f"**\ud83d\udcc0 Average Profit per sqm:** {avg_profit_per_sqm:,.2f} Baht/sqm")
    st.markdown(f"**\ud83c\udf1f Weighted Profit:** {weighted_profit:,.2f} Baht")

# --- Main Logic ---
if train_ai_clicked:
    with st.spinner("Initializing Grid..."):
        grid, new_e = initialize_grid(rows, cols, e_position)
        grid, _ = load_or_initialize_grid(csv_folder, rows, cols, new_e)

    st.success(f"\u2705 Grid Loaded: {rows}x{cols} | Start Position: {new_e}")
    render_colored_grid(grid, "\ud83d\udccc Initial Layout (Before AI)")

    with st.spinner("\ud83e\uddec Training AI... Please wait..."):
        top_layouts, action_log = train_ai(EPISODES, grid)

    st.success("\u2705 AI Training Complete.")

    save_q_table()

    # --- Show all layouts together ---
    st.header("\ud83c\udfe1 All Top 3 Layouts Found")

    for i, (score, layout) in enumerate(top_layouts, 1):
        st.subheader(f"\ud83c\udfc6 Layout #{i} \u2014 Raw Score: {score}")
        render_colored_grid(layout, f"\ud83d\udccc Layout #{i} (Before House Types)")

        final_layout = apply_house_types([row[:] for row in layout])
        render_colored_grid(final_layout, f"\ud83d\udccc Layout #{i} with H1\u2013H4")
        analyze_single_profit(final_layout)

        st.markdown("---")

else:
    st.info("\ud83d\udc48 Please configure settings and press 'Train AI' to start.")
