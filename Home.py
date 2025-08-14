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


st.set_page_config(page_title="ML Visual Explorer", page_icon="🤖" , layout="wide")


# Title and main description
st.title("🎯 Welcome to the ML Visual Explorer")

st.markdown("""
This interactive app lets you explore:

- 📘 **Key optimization concepts** like gradient, Hessian, KKT conditions  
- 🧪 **Playground** for experimenting with optimizers on custom functions  
- 📐 **Symbolic tools** for visualizing Taylor expansions, curvature, and more  
- 🧮 **Taylor Series** module to understand how optimizers arise from first- and second-order approximations
- 📊 **Supervised Learning** including regression and classification model visualizations  
- 🧵 **Unsupervised Learning** including clustering, dimensionality reduction, and anomaly detection  

""")

st.info("👉 Use the sidebar to dive into each section. You can switch between **Taylor Series**, **Optimizer Playground**, **Supervised Learning** and **Unsupervised Learning** tabs.")

st.markdown(
    "If you have any feedback regarding the application, kindly fill out this [form](https://forms.gle/tae4s9EH5dqcG6rm9).",
    unsafe_allow_html=True
)

