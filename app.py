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

st.markdown(
    "If you have any feedback regarding the application, kindly fill out this [form](https://forms.gle/tae4s9EH5dqcG6rm9).",
    unsafe_allow_html=True
)

import base64

PDF_PATH = "/Users/zhangmengjie/Documents/Capstone Project/Text book/convex optimisation.pdf"  # Replace with your file path

# Display PDF inline
def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'''
            <iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="700px" type="application/pdf"></iframe>
        '''
        st.markdown(pdf_display, unsafe_allow_html=True)

# Layout
st.subheader("📘 Document Viewer and Download")

col1, col2 = st.columns([4, 1])

with col1:
    display_pdf(PDF_PATH)

with col2:
    with open(PDF_PATH, "rb") as file:
        st.download_button(
            label="📥 Download PDF",
            data=file,
            file_name="your_file.pdf",
            mime="application/pdf"
        )
