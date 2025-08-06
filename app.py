import streamlit as st

# Set page config
st.set_page_config(page_title="Optimization Playground", layout="wide")

# Sidebar - simple Hello and sections
st.sidebar.markdown("## Hello")
st.sidebar.markdown("Welcome to the **Optimization Playground**!")

# Use buttons or radio buttons instead of links for cleaner section navigation
sections = ["Guide", "Taylor Series", "Optimizer Playground"]
section = st.sidebar.radio("Navigate through the sections:", sections)

# Sections below (for content display on the main page)
if section == "Guide":
    st.markdown("## Guide")
    st.markdown("Introduction to optimization concepts and how to use the app.")

elif section == "Taylor Series":
    st.markdown("## Taylor Series")
    st.markdown("Learn how first- and second-order Taylor expansions relate to optimizers.")

elif section == "Optimizer Playground":
    st.markdown("## Optimizer Playground")
    st.markdown("Experiment interactively with different optimizers and functions.")
