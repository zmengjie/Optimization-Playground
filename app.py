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

st.sidebar.markdown("[Guide](#Guide)")
st.sidebar.markdown("[Taylor Series](#Taylor-Series)")
st.sidebar.markdown("[Optimizer Playground](#Optimizer-Playground)")

# Sections below
st.markdown("## Guide")
st.markdown("""
### Introduction to optimization concepts and how to use the app.
""")

st.markdown("## Taylor Series")
st.markdown("""
Learn how first- and second-order Taylor expansions relate to optimizers.
""")

st.markdown("## Optimizer Playground")
st.markdown("""
Experiment interactively with different optimizers and functions.
""")

