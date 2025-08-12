import streamlit as st

def show_resources():
    st.title("📚 Educational Resources")

    # You can expand this list
    external_pdfs = {
        "Convex Optimization (Boyd & Vandenberghe)": "https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf",
        "ISLR Book (Gareth James)": "https://www.statlearning.com/s/ISLRv2_website.pdf",
        "Deep Learning (Ian Goodfellow)": "https://www.deeplearningbook.org/contents/dlbook.pdf"
    }

    selected_title = st.selectbox("Select a resource:", list(external_pdfs.keys()))
    url = external_pdfs[selected_title]

    # Link to open in new tab
    st.markdown(f"[🌐 Open in new tab]({url})", unsafe_allow_html=True)

    # Optional inline iframe preview (for some browsers)
    st.markdown(f"""
    <iframe src="{url}" width="100%" height="800px" style="border:none;"></iframe>
    """, unsafe_allow_html=True)

show_resources()
