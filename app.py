import streamlit as st

# Sidebar configuration
st.sidebar.title("Navigation")
st.sidebar.markdown("Navigate through the pages:")

# Automatically load pages from the `pages/` folder
from pages.taylor_series import run as taylor_series_run

# Call the run() function of the selected page
taylor_series_run()
