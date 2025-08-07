import streamlit as st
from taylor_expansion import show_univariate_taylor, show_multivariable_taylor


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
# Define the run function for this page

st.title("📐 Taylor Series & Optimizer Foundations")

tab1, tab2, tab3 = st.tabs(["📘 Guide", "📈 Univariate", "🌐 Multivariable"])


with tab1:
    # Section 1: Explaining Taylor Series
    st.markdown("### 📚 How Taylor Series Explains Optimizers")
    st.markdown("""
    Many optimization algorithms are grounded in the **Taylor series expansion**, 
    which provides a local approximation of a function using its derivatives:
    """)

    # First-order Taylor expansion formula
    st.latex(r"""
    f(x + \Delta x) \approx f(x) + \nabla f(x)^T \Delta x
    """)
    st.markdown("""
    - This is the **first-order Taylor expansion**, forming the basis of **Gradient Descent**.  
    - It uses only the **gradient (slope)** to determine the update direction.
    """)

    # Second-order Taylor expansion formula
    st.latex(r"""
    f(x + \Delta x) \approx f(x) + \nabla f(x)^T \Delta x + \frac{1}{2} \Delta x^T H(x) \Delta x
    """)
    st.markdown("""
    - This is the **second-order Taylor expansion**, used in **Newton's Method**.  
    - It incorporates the **Hessian matrix (curvature)** to adjust both direction and step size, often accelerating convergence.
    """)

    # Section 2: Visual Summary
    st.markdown("### ✍️ Summary")
    st.markdown("""
    Think of it visually:

    - ✅ **First-order**: Approximates the function using a **tangent line** (just the slope).
    - ✅ **Second-order**: Approximates using a **parabola** (slope + curvature).

    These expansions reveal the underlying logic of optimizers:

    - **Gradient Descent** → uses **first-order** info (gradient only).
    - **Newton’s Method** → uses **second-order** info (gradient + curvature).

    Understanding Taylor series helps you develop a deeper intuition about how optimizers explore the **loss landscape**.
    """)

st.markdown("---")


    # st.markdown("### 📈 Univariate Taylor Expansion Visualizer")
with tab2:
    # Sidebar config
    with st.sidebar:
        st.markdown("### 📈 Univariate Settings")
        # Only place settings here
        f_expr = st.text_input("Enter function f(x)", value="sin(x)")
        xmin, xmax = st.slider("Domain range", -10.0, 10.0, (-5.0, 5.0))
        a = st.slider("Expansion point a", -10.0, 10.0, 0.0)
        show_2nd = st.checkbox("Show 2nd-order", value=True)
        animate = st.checkbox("🎬 Animate 1st & 2nd-order Approximation")

    # Main visualization
    show_univariate_taylor(f_expr, xmin, xmax, a, show_2nd, animate)

# Section 3: Interactive Visualization
# Tab 3: Multivariable Visualizer
with tab3:
    # st.markdown("### 🌐 Multivariable Taylor Expansion Visualizer")
    show_multivariable_taylor()

