# taylor_expansion.py
import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from io import BytesIO
import tempfile
import base64
import streamlit.components.v1 as components
from PIL import Image
import plotly.graph_objects as go


# def show_univariate_taylor():
#     st.markdown("### 🔍 Univariate Taylor Expansion (1D Preview)")

#     try:
#         func_choice = st.selectbox("Choose a function:", ["cos(x)", "exp(x)", "ln(1+x)", "tanh(x)", "Custom"])
#         show_3rd_4th = st.checkbox("➕ Show 3rd & 4th-order", value=False)
#         show_linear = st.checkbox("Show 1st-order (Linear)", value=True)
#         show_parabola = st.checkbox("Show 2nd-order (Parabola)", value=True)

#         x_sym = sp.symbols('x')

#         def get_function(choice):
#             if choice == "cos(x)": return sp.cos(x_sym), (-3, 3)
#             if choice == "exp(x)": return sp.exp(x_sym), (-3, 3)
#             if choice == "ln(1+x)": return sp.ln(1 + x_sym), (-0.9, 3)
#             if choice == "tanh(x)": return sp.tanh(x_sym), (-3, 3)
#             if choice == "Custom":
#                 user_input = st.text_input("Enter function f(x):", "x**2 * sin(x)")
#                 try:
#                     return sp.sympify(user_input), (-3, 3)
#                 except Exception as e:
#                     st.error(f"Invalid input: {e}")
#                     st.stop()

#         f_sym, (xmin, xmax) = get_function(func_choice)
#         x_sym, a_sym = sp.symbols('x a')
#         h = x_sym - a_sym

#         # Derivatives and Taylor terms
#         f1, f2, f3, f4 = [sp.diff(f_sym, x_sym, i) for i in range(1, 5)]
#         T1 = f_sym.subs(x_sym, a_sym) + f1.subs(x_sym, a_sym) * h
#         T2 = T1 + (1/2) * f2.subs(x_sym, a_sym) * h**2
#         T4 = T2 + (1/6) * f3.subs(x_sym, a_sym) * h**3 + (1/24) * f4.subs(x_sym, a_sym) * h**4

#         st.markdown("### ✏️ Taylor Expansion at $x = a$")
#         st.latex(f"f(x) \\approx {sp.latex(T1)}")
#         st.latex(f"f(x) \\approx {sp.latex(T2)}")
#         if show_3rd_4th:
#             st.latex(f"f(x) \\approx {sp.latex(T4)}")

#         # Numeric plotting
#         f_np = sp.lambdify(x_sym, f_sym, "numpy")
#         derivs = [sp.lambdify(x_sym, d, "numpy") for d in [f1, f2, f3, f4]]
#         a = st.slider("Expansion point a:", float(xmin), float(xmax), 0.0, 0.1)
#         x = np.linspace(xmin, xmax, 500)

#         f_vals = [d(a) for d in derivs]
#         t1 = f_np(a) + f_vals[0] * (x - a)
#         t2 = t1 + 0.5 * f_vals[1] * (x - a)**2
#         t4 = t2 + (1/6) * f_vals[2] * (x - a)**3 + (1/24) * f_vals[3] * (x - a)**4

#         fig, ax = plt.subplots(figsize=(8, 4))
#         ax.plot(x, f_np(x), label=f"f(x) = {func_choice}", color='blue')
#         if show_linear: ax.plot(x, t1, '--', label='1st-order', color='red')
#         if show_parabola: ax.plot(x, t2, '--', label='2nd-order', color='orange')
#         if show_3rd_4th: ax.plot(x, t4, '--', label='3rd/4th-order', color='green')
#         ax.axvline(a, color='gray', linestyle=':')
#         ax.axhline(0, color='black', linewidth=0.8)
#         ax.scatter(a, f_np(a), color='black')
#         ax.set_title(f"Taylor Approximations at x = {a}")
#         ax.legend(); ax.grid(True)
#         st.pyplot(fig)


#         # === Optional Animation ===
#         if st.checkbox("🎬 Animate 1st & 2nd-order Approximation"):
#             st.markdown("### 🎬 Animation: 1st & 2nd-Order Taylor Approximation")
#             fig_anim, ax_anim = plt.subplots(figsize=(10,6))
        

#             line_true, = ax_anim.plot(x, f_np(x), label="f(x)", color='blue')
#             line_taylor1, = ax_anim.plot([], [], '--', label="1st-order", color='red')
#             line_taylor2, = ax_anim.plot([], [], '--', label="2nd-order", color='orange')
#             point, = ax_anim.plot([], [], 'ko')

#             ax_anim.set_xlim(xmin, xmax)
#             y_vals = f_np(x)
#             buffer = 0.4 * (np.max(y_vals) - np.min(y_vals))
#             ax_anim.set_ylim(np.min(y_vals) - buffer, np.max(y_vals) + buffer)
#             ax_anim.axhline(0, color='gray', lw=0.5)
#             ax_anim.grid(True)
#             ax_anim.legend()

#             a_vals = np.linspace(xmin + 0.1, xmax - 0.1, 60)

#             def update(frame):
#                 a_val = a_vals[frame]
#                 f_a = f_np(a_val)
#                 f1_a = derivs[0](a_val)
#                 f2_a = derivs[1](a_val)

#                 t1_anim = f_a + f1_a * (x - a_val)
#                 t2_anim = t1_anim + 0.5 * f2_a * (x - a_val)**2

#                 line_taylor1.set_data(x, t1_anim)
#                 line_taylor2.set_data(x, t2_anim)
#                 point.set_data([a_val], [f_a])
#                 ax_anim.set_title(f"Taylor Approx at a = {a_val:.2f}")
#                 return line_taylor1, line_taylor2, point

#             ani = FuncAnimation(fig_anim, update, frames=len(a_vals), interval=100, blit=True)

#             buf = BytesIO()
#             writer = PillowWriter(fps=20)
#             with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmpfile:
#                 ani.save(tmpfile.name, writer=writer)
#                 tmpfile.seek(0)
#                 gif_base64 = base64.b64encode(tmpfile.read()).decode("utf-8")

#             components.html(f'<img src="data:image/gif;base64,{gif_base64}" width="100%">', height=350)

#     except Exception as e:
#         st.error(f"Rendering error: {e}")


# === UNIVARIATE TAYLOR FUNCTION ===
def show_univariate_taylor():
    try:
        st.markdown("### 📈 Univariate Taylor Expansion")
        func_str = st.sidebar.text_input("Enter function f(x)", "sin(x)")
        xmin, xmax = st.sidebar.slider("Domain range", -10.0, 10.0, (-5.0, 5.0))
        a_val = st.sidebar.slider("Expansion point a", xmin + 0.1, xmax - 0.1, 0.0)
        show_2nd = st.sidebar.checkbox("Show 2nd-order", True)
        animate = st.sidebar.checkbox("🎬 Animate 1st & 2nd-order Approximation")

        x = np.linspace(xmin, xmax, 400)
        x_sym = sp.Symbol("x")
        f_sym = sp.sympify(func_str)
        f_np = sp.lambdify(x_sym, f_sym, modules=["numpy"])
        f1 = sp.diff(f_sym, x_sym)
        f2 = sp.diff(f1, x_sym)
        derivs = [sp.lambdify(x_sym, f1, modules=["numpy"]),
                  sp.lambdify(x_sym, f2, modules=["numpy"])]

        f_a = f_np(a_val)
        f1_a = derivs[0](a_val)
        f2_a = derivs[1](a_val)
        t1 = f_a + f1_a * (x - a_val)
        t2 = t1 + 0.5 * f2_a * (x - a_val)**2

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, f_np(x), label="f(x)", color='blue')
        ax.plot(x, t1, '--', label="1st-order", color='red')
        if show_2nd:
            ax.plot(x, t2, '--', label="2nd-order", color='orange')
        ax.axvline(a_val, color='gray', linestyle=':')
        ax.axhline(0, color='gray', lw=0.5)
        ax.set_title("Taylor Approximation at x = {:.2f}".format(a_val))
        ax.grid(True)
        ax.legend()
        st.pyplot(fig, use_container_width=True)

        # === Optional Animation ===
        if animate:
            st.markdown("### 🎬 Animation: 1st & 2nd-Order Taylor Approximation")
            fig_anim, ax_anim = plt.subplots(figsize=(10, 6))

            line_true, = ax_anim.plot(x, f_np(x), label="f(x)", color='blue')
            line_taylor1, = ax_anim.plot([], [], '--', label="1st-order", color='red')
            line_taylor2, = ax_anim.plot([], [], '--', label="2nd-order", color='orange')
            point, = ax_anim.plot([], [], 'ko')

            ax_anim.set_xlim(xmin, xmax)
            y_vals = f_np(x)
            buffer = 0.4 * (np.max(y_vals) - np.min(y_vals))
            ax_anim.set_ylim(np.min(y_vals) - buffer, np.max(y_vals) + buffer)
            ax_anim.axhline(0, color='gray', lw=0.5)
            ax_anim.grid(True)
            ax_anim.legend()

            a_vals = np.linspace(xmin + 0.1, xmax - 0.1, 60)

            def update(frame):
                a_val = a_vals[frame]
                f_a = f_np(a_val)
                f1_a = derivs[0](a_val)
                f2_a = derivs[1](a_val)

                t1_anim = f_a + f1_a * (x - a_val)
                t2_anim = t1_anim + 0.5 * f2_a * (x - a_val)**2

                line_taylor1.set_data(x, t1_anim)
                line_taylor2.set_data(x, t2_anim)
                point.set_data([a_val], [f_a])
                ax_anim.set_title(f"Taylor Approx at a = {a_val:.2f}")
                return line_taylor1, line_taylor2, point

            ani = FuncAnimation(fig_anim, update, frames=len(a_vals), interval=100, blit=True)

            buf = BytesIO()
            writer = PillowWriter(fps=20)
            with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmpfile:
                ani.save(tmpfile.name, writer=writer)
                tmpfile.seek(0)
                gif_base64 = base64.b64encode(tmpfile.read()).decode("utf-8")

            components.html(f'<img src="data:image/gif;base64,{gif_base64}" width="100%">', height=350)

    except Exception as e:
        st.error(f"Rendering error: {e}")


# --- SECTION: Multivariable Taylor Expansion (2D Preview) ---
def show_multivariable_taylor():
    st.markdown("### 🌐 Multivariable Taylor Expansion (2D Preview)")

    multi_func = st.selectbox("Choose function:", ["Quadratic Bowl", "Rosenbrock", "sin(x)cos(y)", "exp(-x² - y²)"])

    x, y = sp.symbols('x y')
    a, b = sp.symbols('a b')

    if multi_func == "Quadratic Bowl":
        fxy = x**2 + y**2
    elif multi_func == "Rosenbrock":
        fxy = (1 - x)**2 + 100 * (y - x**2)**2
    elif multi_func == "sin(x)cos(y)":
        fxy = sp.sin(x) * sp.cos(y)
    elif multi_func == "exp(-x² - y²)":
        fxy = sp.exp(-(x**2 + y**2))

    # UI-controlled values
    a_input = st.slider("Center a (x)", -5.0, 5.0, 0.0)
    b_input = st.slider("Center b (y)", -5.0, 5.0, 0.0)

    zoom_in = st.checkbox("🔍 Zoom into local neighborhood", value=False)
    xlim = (a_input - 1, a_input + 1) if zoom_in else (-5, 5)
    ylim = (b_input - 1, b_input + 1) if zoom_in else (-5, 5)

    # Derivatives
    grad = [sp.diff(fxy, v) for v in (x, y)]
    hess = [[sp.diff(g, v) for v in (x, y)] for g in grad]

    # --- Symbolic Taylor Expansion (parameterized by a, b)
    T2_symbolic = (
        fxy.subs({x: a, y: b})
        + grad[0].subs({x: a, y: b}) * (x - a)
        + grad[1].subs({x: a, y: b}) * (y - b)
        + 0.5 * hess[0][0].subs({x: a, y: b}) * (x - a)**2
        + hess[0][1].subs({x: a, y: b}) * (x - a)*(y - b)
        + 0.5 * hess[1][1].subs({x: a, y: b}) * (y - b)**2
    )

    T2_func = sp.lambdify((x, y, a, b), T2_symbolic, "numpy")
    def T2_np(X, Y, a_val, b_val):
        return T2_func(X, Y, a_val, b_val)

    # Evaluate derivatives at (a, b)
    f_a = float(fxy.subs({x: a_input, y: b_input}))
    grad_val = [float(g.subs({x: a_input, y: b_input})) for g in grad]
    hess_val = [[float(h.subs({x: a_input, y: b_input})) for h in row] for row in hess]

    # Raw expressions (numerical)
    T1_expr = f_a + grad_val[0]*(x - a_input) + grad_val[1]*(y - b_input)
    T2_expr = T1_expr + 0.5 * (
        hess_val[0][0]*(x - a_input)**2 +
        2*hess_val[0][1]*(x - a_input)*(y - b_input) +
        hess_val[1][1]*(y - b_input)**2
    )

    T1_raw_latex = sp.latex(sp.simplify(T1_expr))
    T2_raw_expr = T2_symbolic.subs({a: a_input, b: b_input})
    T2_raw_latex = sp.latex(sp.simplify(T2_raw_expr))

    # --- Display Side-by-Side Summary ---
    st.markdown(
        fr"### 📐 Taylor Expansion Summary: Expansion at \( (x, y) = ({a_input:.2f}, {b_input:.2f}) \)"
    )

    # st.markdown(fr"""### ✏️ Expansion at \((x, y) = ({a_input:.2f}, {b_input:.2f})\)""")x
    st.markdown("#### Original Function")
    st.latex(fr"\small f(x, y) = {sp.latex(sp.simplify(fxy))}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 1st-Order Expansion")
        st.markdown("**Symbolic Template**")
        st.latex(r"\small f(x, y) \approx f(a, b) + f_x(a, b)(x - a) + f_y(a, b)(y - b)")
        st.markdown("**Evaluated at (a, b)**")
        st.latex(fr"\small f(x, y) \approx {T1_raw_latex}")

    with col2:
        st.markdown("#### 2nd-Order Expansion")
        st.markdown("**Symbolic Template**")
        st.latex(r"""
        \small
        \begin{aligned}
        f(x, y) \approx\ & f(a, b) + f_x(a, b)(x - a) + f_y(a, b)(y - b) \\
        & + \frac{1}{2}f_{xx}(a, b)(x - a)^2 + f_{xy}(a, b)(x - a)(y - b) + \frac{1}{2}f_{yy}(a, b)(y - b)^2
        \end{aligned}
        """)
        st.markdown("**Evaluated at (a, b)**")
        st.latex(fr"\small f(x, y) \approx {T2_raw_latex}")



    # Evaluate
    f_np = sp.lambdify((x, y), fxy, "numpy")
    X, Y = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
    Z_true = f_np(X, Y)
    Z_taylor = T2_np(X, Y, a_input, b_input)

    # Plot
    fig_true = go.Figure(data=[go.Surface(z=Z_true, x=X, y=Y, colorscale='Viridis')])
    fig_true.update_layout(title="True Function", scene=dict(
        xaxis_title='x', yaxis_title='y', zaxis_title='f(x,y)'
    ), margin=dict(l=0, r=0, b=0, t=40))

    fig_taylor = go.Figure(data=[go.Surface(z=Z_taylor, x=X, y=Y, colorscale='RdBu')])
    fig_taylor.update_layout(title="2nd-Order Taylor Approx", scene=dict(
        xaxis_title='x', yaxis_title='y', zaxis_title='Approx'
    ), margin=dict(l=0, r=0, b=0, t=40))

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_true, use_container_width=True)
    with col2:
        st.plotly_chart(fig_taylor, use_container_width=True)


    st.markdown("---")  # separator

    with st.expander("ℹ️ Note on 2nd-order Taylor Expansion", expanded=True):
        st.write("**Note:** The 2nd-order Taylor expansion approximates the function locally around:")
        st.latex(r"(a, b)")
    
        st.write("It uses gradient and Hessian values at that point.")
    
        st.write("For smooth functions like:")
        st.latex(r"\sin(x)\cos(y)")
        st.write("the approximation is accurate near (a, b), but may diverge further away.")
    
        st.write("For quadratic functions like:")
        st.latex(r"x^2 + y^2")
        st.write("the 2nd-order Taylor expansion exactly matches the function, "
                 "because the function itself is already a polynomial of degree 2.")
    


    # --- Animation ---
    st.markdown("---")
    st.markdown("### 🎬 Animate Taylor Approximation Surface")

    animate_mode = st.radio("Animate path:", ["a only", "b only", "both a & b"], index=0)
    param_vals = np.linspace(-1.0, 1.0, 30)
    frames = []

    for val in param_vals:
        if animate_mode == "a only":
            Z_frame = T2_np(X, Y, val, b_input)
            label = f"a = {val:.2f}"
        elif animate_mode == "b only":
            Z_frame = T2_np(X, Y, a_input, val)
            label = f"b = {val:.2f}"
        else:
            Z_frame = T2_np(X, Y, val, val)
            label = f"(a, b) = ({val:.2f}, {val:.2f})"

        frames.append(go.Frame(data=[
            go.Surface(z=Z_frame, x=X, y=Y, colorscale='RdBu')
        ], name=label))

    # Initial frame
    if animate_mode == "a only":
        Z0 = T2_np(X, Y, param_vals[0], b_input)
    elif animate_mode == "b only":
        Z0 = T2_np(X, Y, a_input, param_vals[0])
    else:
        Z0 = T2_np(X, Y, param_vals[0], param_vals[0])

    fig_anim = go.Figure(
        data=[go.Surface(z=Z0, x=X, y=Y, colorscale='RdBu')],
        layout=go.Layout(
            title="Animated 2nd-Order Taylor Approximation",
            scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='Approx'),
            updatemenus=[dict(
                type="buttons",
                showactive=True,
                buttons=[dict(label="▶ Play", method="animate", args=[None])]
            )],
            sliders=[{
                "steps": [{"args": [[f.name]], "label": f.name, "method": "animate"} for f in frames],
                "currentvalue": {"prefix": "Center: "}
            }]
        ),
        frames=frames
    )

    st.plotly_chart(fig_anim, use_container_width=True)
