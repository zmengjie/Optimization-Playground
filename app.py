import streamlit as st

# Set page config
st.set_page_config(page_title="Optimization Playground", layout="wide")

# Sidebar content with collapsible sections
with st.sidebar:
    st.header("Hello")
    
    # Create collapsible sections in the sidebar
    with st.expander("Guide", expanded=True):
        st.write("""
        Welcome to the **Optimization Playground**!

        This app is divided into three sections:
        1. **Guide (this page)** – Introduction to optimization concepts and how to use the app.
        2. **Taylor Series** – Learn how first- and second-order Taylor expansions relate to optimizers.
        3. **Optimizer Playground** – Experiment interactively with different optimizers and functions.

        Use the menu on the left to switch between sections.
        """)
    
    with st.expander("Taylor Series"):
        st.write("""
        ### 📚 How Taylor Series Explains Optimizers
        Many optimization algorithms are grounded in the **Taylor series expansion**, which provides a local approximation of a function using its derivatives:
        - **First-order Taylor expansion**: Forms the basis of **Gradient Descent**.
        - **Second-order Taylor expansion**: Used in **Newton's Method** to accelerate convergence.
        """)

    with st.expander("Optimizer Playground"):
        st.write("""
        ### 🤖 Choose Your Optimizer
        Experiment with various optimizers and see how they perform on different functions.
        """)

# Main content of the page
if 'option' not in st.session_state:
    st.session_state['option'] = 'Guide'

option = st.sidebar.radio("Select a section", ["Guide", "Taylor Series", "Optimizer Playground"])

# Section 1: Guide Page
if option == "Guide":
    st.title("📘 Optimization Playground Guide")
    st.markdown("""
    Welcome to the **Optimization Playground**!

    This app is divided into three sections:
    1. **Guide (this page)** – Introduction to optimization concepts and how to use the app.
    2. **Taylor Series** – Learn how first- and second-order Taylor expansions relate to optimizers.
    3. **Optimizer Playground** – Experiment interactively with different optimizers and functions.

    Use the menu on the left to switch between sections.
    """)

# Section 2: Taylor Series
elif option == "Taylor Series":
    st.title("🧠 Taylor Series & Optimizer Foundations")
    st.markdown("""
    ### 📚 How Taylor Series Explains Optimizers
    Many optimization algorithms are grounded in the **Taylor series expansion**, which provides a local approximation of a function using its derivatives.
    - **First-order Taylor expansion**: Forms the basis of **Gradient Descent**.
    - **Second-order Taylor expansion**: Used in **Newton's Method** to accelerate convergence.
    """)

    # You can include further detailed content and visualizations as per your original Taylor series section code
    # For example, you can call another function like show_univariate_taylor() to display Taylor series plots

# Section 3: Optimizer Playground
elif option == "Optimizer Playground":
    st.title("🧪 Optimizer Visual Playground")

    # Example content - you can replace this with the actual content and interactivity for your optimizer playground
    st.markdown("""
    ### 🤖 Choose Your Optimizer
    Experiment with various optimizers and see how they perform on different functions.
    """)

    # Add interactivity for optimizer selection, function selection, etc. Here you can import and use your existing optimizer playground code.


    



