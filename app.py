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
- 🧮 **Taylor Series** module to understand how optimizers arise from first- and second-order approximations

""")

st.info("👉 Use the sidebar to dive into each section. You can switch between **Taylor Series** and **Optimizer Playground** tabs.")

