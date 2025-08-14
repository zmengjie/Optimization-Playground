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

mode = st.sidebar.radio("Select Mode", ["📘 Guide", "📈 Univariate", "🌐 Multivariable"])


# tab1, tab2, tab3 = st.tabs(["📘 Guide", "📈 Univariate", "🌐 Multivariable"])

if mode == "📘 Guide":
# with tab1:
    st.header("🧭 How to Use the Taylor Visualization Tool")

    st.subheader("📈 Univariate Mode")
    st.markdown("""
    - Choose from **predefined functions** (e.g., `cos(x)`) or select **custom** to enter your own.
    - Use checkboxes to toggle:
    - **1st-order** (Linear)
    - **2nd-order** (Parabola)
    - **3rd & 4th-order** terms
    - Adjust the **expansion point `a`** using the slider to see how Taylor approximation changes.
    - Enable **animation** to dynamically view the approximation update.
    """)

    st.subheader("🌐 Multivariable Mode")
    st.markdown("""
    - Select from predefined 2D functions or enter a **custom** bivariate function.
    - Adjust **center a (x)** and **b (y)** using sliders.
    - Toggle animation to move:
    - Only `a`, only `b`, or both together.
    - Visual comparison between the **true function** and its **2nd-order approximation** is shown in 3D.
    """)

    st.info("ℹ️ You can switch between Univariate and Multivariable using the sidebar selector. ")
    st.info(
        "📘 For math and theory, visit the **Resources** page. "
        "This will help you understand how Taylor series underpins optimization methods."
    )



elif mode == "📈 Univariate":
    # -------------------- LEFT: controls --------------------
    with st.sidebar:
        st.markdown("### 📈 Univariate Settings")
        func_choice = st.selectbox("Choose a function:", ["cos(x)", "exp(x)", "ln(1+x)", "tanh(x)", "Custom"])
        show_3rd_4th = st.checkbox("➕ Show 3rd & 4th-order", value=False)
        show_parabola = st.checkbox("Show 2nd-order (Parabola)", value=True)
        show_linear = st.checkbox("Show 1st-order (Linear)", value=True)
        animate = st.checkbox("🎬 Animate Taylor Approximation", value=False)

        x_sym = sp.Symbol("x")

        def get_function(choice):
            if choice == "cos(x)":   return sp.cos(x_sym), (-3, 3)
            if choice == "exp(x)":   return sp.exp(x_sym), (-3, 3)
            if choice == "ln(1+x)":  return sp.log(1 + x_sym), (-0.9, 3)   # log == natural log
            if choice == "tanh(x)":  return sp.tanh(x_sym), (-3, 3)
            if choice == "Custom":
                user = st.text_input("Enter f(x):", "x**2 * sin(x)")
                try:
                    return sp.sympify(user), (-3, 3)
                except Exception as e:
                    st.error(f"Invalid input: {e}")
                    st.stop()

        f_expr, (xmin, xmax) = get_function(func_choice)
        a = st.slider("Expansion point a", xmin + 0.1, xmax - 0.1, 0.0)
        animate_orders = ["1st", "2nd"] if animate else []


    # -------------------- RIGHT: formulas + plot --------------------

    st.markdown(r"### ✏️ Taylor Expansion at $x = a$")

    try:
        # 1st- and 2nd-order series at x=a
        t1 = sp.series(f_expr, x_sym, a, 2).removeO()  # up to (x-a)^1
        t2 = sp.series(f_expr, x_sym, a, 3).removeO()  # up to (x-a)^2
        st.latex(rf"f(x) \approx {sp.latex(t1)}")
        st.latex(rf"f(x) \approx {sp.latex(t2)}")

        # Optional 3rd & 4th
        if show_3rd_4th:
            t3 = sp.series(f_expr, x_sym, a, 4).removeO()
            t4 = sp.series(f_expr, x_sym, a, 5).removeO()
            st.latex(rf"f(x) \approx {sp.latex(t3)}")
            st.latex(rf"f(x) \approx {sp.latex(t4)}")

    except Exception as e:
        st.info(
            "Could not compute the Taylor series at this a (e.g., singularity for log near -1). "
            "Try another expansion point."
        )

    # Plot / animation
    show_univariate_taylor(
        f_expr=f_expr, xmin=xmin, xmax=xmax, a=a,
        show_linear=show_linear, show_2nd=show_parabola,
        show_3rd_4th=show_3rd_4th,
        animate=True,
        order_to_animate=animate_orders, # plot only
    )


# Section 3: Interactive Visualization
# Tab 3: Multivariable Visualizer
elif mode == "🌐 Multivariable":
    # st.markdown("### 🌐 Multivariable Taylor Expansion Visualizer")
    show_multivariable_taylor()

