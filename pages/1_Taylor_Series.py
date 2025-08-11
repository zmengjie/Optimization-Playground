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
# with tab2:
#     # Sidebar config
#     with st.sidebar:
#         st.markdown("### 📈 Univariate Settings")

#         func_choice = st.selectbox("Choose a function:", ["cos(x)", "exp(x)", "ln(1+x)", "tanh(x)", "Custom"])
#         show_3rd_4th = st.checkbox("➕ Show 3rd & 4th-order", value=False)
#         show_parabola = st.checkbox("Show 2nd-order (Parabola)", value=True)
#         show_linear = st.checkbox("Show 1st-order (Linear)", value=True)
#         animate = st.checkbox("🎬 Animate Taylor Approximation")  # ✅ Only once

#         x_sym = sp.Symbol('x')

#         def get_function(choice):
#             if choice == "cos(x)":
#                 return sp.cos(x_sym), (-3, 3)
#             if choice == "exp(x)":
#                 return sp.exp(x_sym), (-3, 3)
#             if choice == "ln(1+x)":
#                 return sp.ln(1 + x_sym), (-0.9, 3)
#             if choice == "tanh(x)":
#                 return sp.tanh(x_sym), (-3, 3)
#             if choice == "Custom":
#                 user_input = st.text_input("Enter function f(x):", "x**2 * sin(x)")
#                 try:
#                     return sp.sympify(user_input), (-3, 3)
#                 except Exception as e:
#                     st.error(f"Invalid input: {e}")
#                     st.stop()

#         f_expr, (xmin, xmax) = get_function(func_choice)
#         a = st.slider("Expansion point a", xmin + 0.1, xmax - 0.1, 0.0)

#         if animate:
#             animate_orders = ["1st", "2nd"]  # Always animate both
#         else:
#             animate_orders = []


#     # Main content: plot + animation
#     show_univariate_taylor(
#         f_expr=f_expr,
#         xmin=xmin,
#         xmax=xmax,
#         a=a,
#         show_linear=show_linear,
#         show_2nd=show_parabola,
#         show_3rd_4th=show_3rd_4th,
#         animate=animate,
#         order_to_animate=animate_orders  # Pass list instead of string
#     )


with tab2:
    left, right = st.columns([1, 3])

    # -------------------- LEFT: controls --------------------
    with left:
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
    with right:
        st.markdown(r"### ✏️ Taylor Expansion at $x = a$")

        show_steps = st.checkbox("Show derivation steps", value=True)
        max_order = 4 if show_3rd_4th else 2  # 2 → up to quadratic; 4 → up to 4th

        def render_steps(f, x, a_val, order):
            # General Taylor form
            st.latex(rf"f(x) \approx \sum_{{k=0}}^{order} \frac{{f^{({{k}})}(a)}}{{k!}}(x-a)^k")

            # Build derivatives f^(k)
            derivs = [f]
            for _ in range(1, order + 1):
                derivs.append(sp.diff(derivs[-1], x))

            # Values at a
            vals = [sp.simplify(d.subs(x, a_val)) for d in derivs]

            # List f(a), f'(a), f''(a), ...
            for k, v in enumerate(vals):
                label = r"f(a)" if k == 0 else rf"f^{{({k})}}(a)"
                st.latex(rf"{label} = {sp.latex(v)}")

            # Partial sums term-by-term
            st.markdown("**Partial sums:**")
            partial = 0
            for k, v in enumerate(vals):
                term = sp.simplify(v * (x - a_val)**k / sp.factorial(k))
                partial = sp.simplify(partial + term)
                st.latex(rf"\text{{Up to }}k={k}:\quad f(x) \approx {sp.latex(partial)}")

        try:
            if show_steps:
                render_steps(f_expr, x_sym, a, max_order)
            else:
                # Final compact formulas (like your current version)
                t2 = sp.series(f_expr, x_sym, a, 3).removeO()  # up to (x-a)^2
                st.latex(rf"f(x) \approx {sp.latex(t2)}")
                if show_3rd_4th:
                    t4 = sp.series(f_expr, x_sym, a, 5).removeO()
                    st.latex(rf"f(x) \approx {sp.latex(t4)}")
        except Exception:
            st.info("Could not compute the Taylor series at this a (e.g., singularity). Try another expansion point.")

        # ---- plot / animation ----
        st.markdown("### 📉 Taylor Approximation")
        show_univariate_taylor(
            f_expr=f_expr, xmin=xmin, xmax=xmax, a=a,
            show_linear=show_linear, show_2nd=show_parabola,
            show_3rd_4th=show_3rd_4th, animate=animate,
            order_to_animate=animate_orders
        )



# Section 3: Interactive Visualization
# Tab 3: Multivariable Visualizer
with tab3:
    # st.markdown("### 🌐 Multivariable Taylor Expansion Visualizer")
    show_multivariable_taylor()

