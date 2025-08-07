import streamlit as st

st.markdown(
    """
    <style>
    /* Make the content container full-width */
    .block-container {
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }

    /* Optional: Make the page background wider as well */
    .main {
        max-width: 100vw !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.set_page_config(page_title="Optimization Playground", layout="wide")


# Title and main description
st.title("🎯 Welcome to the Optimization Playground")

st.markdown("""
This interactive app lets you explore:
- 📘 **Key optimization concepts** like gradient, Hessian, KKT conditions
- 🧪 **Playground** for experimenting with optimizers on custom functions
- 📐 **Symbolic tools** for visualizing Taylor expansions, curvature, and more
""")

st.info("👉 Use the sidebar to dive into each section. You can switch between **Taylor Series** and **Optimizer Playground** tabs.")

# Taylor Series Section
st.subheader("🧠 What is a Taylor Series?")
st.markdown("""
The **Taylor series** approximates complex functions using their derivatives at a single point.

- **1st-order** captures local slope:  
  \\( f(x + \Delta x) \\approx f(x) + f'(x) \\Delta x \\)

- **2nd-order** includes curvature:  
  \\( f(x + \Delta x) \\approx f(x) + f'(x) \\Delta x + \\frac{1}{2} f''(x) (\Delta x)^2 \\)

This is the foundation of **Gradient Descent**, **Newton’s Method**, and many optimization techniques.
""")