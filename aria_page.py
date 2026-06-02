"""
Aria Watchlist page for the Village AI Dashboard
"""

import streamlit as st
from aria_watchlist import ARIA_TICKERS, ARIA_GROUPS


def render_aria_page():
    st.markdown("""
        <div style='text-align: center; margin-bottom: 24px;'>
            <h1 style='color: #FFFFFF;'>Aria Watchlist</h1>
            <p style='color: #CCCCCC; font-size: 16px;'>
                {count} tickers across {groups} sectors
            </p>
        </div>
    """.format(count=len(ARIA_TICKERS), groups=len(ARIA_GROUPS)), unsafe_allow_html=True)

    for group_name, tickers in ARIA_GROUPS.items():
        st.markdown(f"### {group_name}")
        cols = st.columns(6)
        for i, ticker in enumerate(tickers):
            cols[i % 6].markdown(
                f"""<div style='background:#1e2130;border:1px solid #444;border-radius:8px;
                    padding:8px 4px;text-align:center;margin:4px 0;'>
                    <span style='color:#00d4ff;font-weight:bold;font-size:15px;'>{ticker}</span>
                </div>""",
                unsafe_allow_html=True,
            )
        st.markdown("---")

    with st.expander("Full ticker list (flat)"):
        st.code(", ".join(ARIA_TICKERS))
