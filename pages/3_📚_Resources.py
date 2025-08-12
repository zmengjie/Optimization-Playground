import streamlit as st

def show_resources():
    st.title("📚 Educational Resources")

    # Unified resource dictionary
    resources = {
        "📘 Convex Optimization (Boyd & Vandenberghe) [PDF]":
            "https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf",

        "🎥 Convex Optimization Lecture Playlist [YouTube]":
            "https://www.youtube.com/watch?v=McLq1hEq3UY&list=PL3940DD956CDF0622"
    }

    selected_title = st.selectbox("Select a resource:", list(resources.keys()))
    url = resources[selected_title]

    # Display the link
    st.markdown(f"[🌐 Open in new tab]({url})", unsafe_allow_html=True)

    # Optionally embed YouTube if selected
    if "youtube.com" in url:
        st.video(url)

show_resources()
