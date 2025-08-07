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


st.title("Welcome to the Optimization Playground")
st.write("Select a page from the left to get started.")
