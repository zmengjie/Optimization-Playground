import streamlit as st
from taylor_expansion import show_univariate_taylor, show_multivariable_taylor
import sympy as sp

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

        func_choice = st.selectbox("Choose a function:", ["cos(x)", "exp(x)", "ln(1+x)", "tanh(x)", "Custom"])
        show_3rd_4th = st.checkbox("➕ Show 3rd & 4th-order", value=False)
        show_parabola = st.checkbox("Show 2nd-order (Parabola)", value=True)
        show_linear = st.checkbox("Show 1st-order (Linear)", value=True)
        animate = st.checkbox("🎬 Animate Taylor Approximation")  # ✅ Only once

        x_sym = sp.Symbol('x')

        def get_function(choice):
            if choice == "cos(x)":
                return sp.cos(x_sym), (-3, 3)
            if choice == "exp(x)":
                return sp.exp(x_sym), (-3, 3)
            if choice == "ln(1+x)":
                return sp.ln(1 + x_sym), (-0.9, 3)
            if choice == "tanh(x)":
                return sp.tanh(x_sym), (-3, 3)
            if choice == "Custom":
                user_input = st.text_input("Enter function f(x):", "x**2 * sin(x)")
                try:
                    return sp.sympify(user_input), (-3, 3)
                except Exception as e:
                    st.error(f"Invalid input: {e}")
                    st.stop()

        f_expr, (xmin, xmax) = get_function(func_choice)
        a = st.slider("Expansion point a", xmin + 0.1, xmax - 0.1, 0.0)

        if animate:
            animate_orders = ["1st", "2nd"]  # Always animate both
        else:
            animate_orders = []


    # Main content: plot + animation
    show_univariate_taylor(
        f_expr=f_expr,
        xmin=xmin,
        xmax=xmax,
        a=a,
        show_linear=show_linear,
        show_2nd=show_parabola,
        show_3rd_4th=show_3rd_4th,
        animate=animate,
        order_to_animate=animate_orders  # Pass list instead of string
    )





# Section 3: Interactive Visualization
# Tab 3: Multivariable Visualizer
with tab3:
    # st.markdown("### 🌐 Multivariable Taylor Expansion Visualizer")
    show_multivariable_taylor()

