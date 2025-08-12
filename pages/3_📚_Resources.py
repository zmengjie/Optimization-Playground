import streamlit as st
import os
import base64

PDF_DIR = "files"

def list_pdfs():
    return [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

def render_pdf_viewer(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        pdf_display = f"""
            <iframe src="data:application/pdf;base64,{base64_pdf}"
            width="100%" height="700px" type="application/pdf"></iframe>
        """
        st.markdown(pdf_display, unsafe_allow_html=True)

def show_resources():
    st.title("📚 Educational Resources")

    pdf_files = list_pdfs()

    if not pdf_files:
        st.warning("No PDFs available in the resources folder.")
        return

    selected_pdf = st.selectbox("Select a resource:", pdf_files)
    file_path = os.path.join(PDF_DIR, selected_pdf)

    # Display inline PDF
    render_pdf_viewer(file_path)

    # Download option
    with open(file_path, "rb") as f:
        st.download_button(
            label="📥 Download this file",
            data=f,
            file_name=selected_pdf,
            mime="application/pdf"
        )

if __name__ == "__main__":
    show_resources()
