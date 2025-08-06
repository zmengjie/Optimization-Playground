import streamlit as st

# Set page config
st.set_page_config(page_title="Optimization Playground", layout="wide")

# Sidebar content with "Hello" section and links to sections
with st.sidebar:
    st.header("Hello")
    st.markdown("""
    Welcome to the **Optimization Playground**!  
    This app is designed to help you explore and experiment with optimization techniques and understand their foundations.

    Navigate through the sections below:
    """)
    st.markdown("[Guide](#guide)")
    st.markdown("[Taylor Series](#taylor-series)")
    st.markdown("[Optimizer Playground](#optimizer-playground)")

# Main content of the page

# Section 1: Guide Page
st.markdown("### Guide")
st.title("📘 Optimization Playground Guide")
st.markdown("""
Welcome to the **Optimization Playground**!

This app is divided into three sections:
1. **Guide (this page)** – Introduction to optimization concepts and how to use the app.
2. **Taylor Series** – Learn how first- and second-order Taylor expansions relate to optimizers.
3. **Optimizer Playground** – Experiment interactively with different optimizers and functions.
""")

# Section 2: Taylor Series
st.markdown("### Taylor Series")
st.title("🧠 Taylor Series & Optimizer Foundations")
st.markdown("""
### 📚 How Taylor Series Explains Optimizers
Many optimization algorithms are grounded in the **Taylor series expansion**, which provides a local approximation of a function using its derivatives:
- **First-order Taylor expansion**: Forms the basis of **Gradient Descent**.
- **Second-order Taylor expansion**: Used in **Newton's Method** to accelerate convergence.
""")

# Section 3: Optimizer Playground
st.markdown("### Optimizer Playground")
st.title("🧪 Optimizer Visual Playground")
st.markdown("""
### 🤖 Choose Your Optimizer
Experiment with various optimizers and see how they perform on different functions.
""")

    # Add interactivity for optimizer selection, function selection, etc. Here you can import and use your existing optimizer playground code.



