import streamlit as st

# Set page config
st.set_page_config(page_title="Optimization Playground", layout="wide")

# Sidebar - No section select box needed
st.sidebar.markdown("## Hello")
st.sidebar.markdown("Welcome to the **Optimization Playground**!")
st.sidebar.markdown("""
This app is designed to help you explore and experiment with optimization techniques and understand their foundations.

Navigate through the sections below:
""")

# Buttons in sidebar for each section
guide_button = st.sidebar.button("Guide")
taylor_series_button = st.sidebar.button("Taylor Series")
optimizer_playground_button = st.sidebar.button("Optimizer Playground")

# Sections below based on button clicks
if guide_button:
    st.markdown("## Guide")
    st.markdown("""
    ### Introduction to optimization concepts and how to use the app.
    """)

elif taylor_series_button:
    st.markdown("## Taylor Series")
    st.markdown("""
    Learn how first- and second-order Taylor expansions relate to optimizers.
    """)

elif optimizer_playground_button:
    st.markdown("## Optimizer Playground")
    st.markdown("""
    Experiment interactively with different optimizers and functions.
    """)

