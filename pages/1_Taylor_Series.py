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


# --- Add near the top of your Taylor page (after imports) ---

import streamlit as st

def guide_modal(key="taylor"):
    # one-time CSS
    st.markdown("""
    <style>
      /* overlay */
      #guide-overlay { position: fixed; inset: 0; background: rgba(0,0,0,.55); z-index: 9998; }
      /* modal panel */
      .guide-modal {
        position: fixed; z-index: 9999;
        top: 8vh; left: 50%; transform: translateX(-50%);
        width: min(880px, 92vw);
        background: #fff; border-radius: 14px;
        padding: 22px 26px; box-shadow: 0 25px 80px rgba(0,0,0,.35);
      }
      .guide-modal h2 { margin: 0 0 .6rem 0; }
      .guide-modal h4 { margin: 1.1rem 0 .4rem 0; }
      .guide-modal ul { margin: .25rem 0 .6rem 1.2rem; }
      /* make Streamlit button compact and not full-width */
      .guide-actions .stButton > button {
        width: auto !important; padding: .45rem .9rem;
        border-radius: 10px;
      }
      /* right-align the Close button row */
      .guide-actions { display: flex; justify-content: flex-end; margin-top: .5rem; }
    </style>
    """, unsafe_allow_html=True)

    # state key names
    open_key  = f"{key}_guide_open"
    show_key  = f"{key}_show_guide"

    if show_key not in st.session_state:
        st.session_state[show_key] = False

    # small trigger button (put wherever you like)
    if st.button("❓ Guide", key=open_key):
        st.session_state[show_key] = True

    # render modal
    if st.session_state[show_key]:
        # overlay + panel
        st.markdown('<div id="guide-overlay"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="guide-modal">
          <h2>🧭 How to use</h2>

          <h4>Univariate Mode</h4>
          <ul>
            <li>Choose a predefined function or select <b>Custom</b> and enter your own.</li>
            <li>Toggle:
              <ul>
                <li><b>1st-order</b> (Linear)</li>
                <li><b>2nd-order</b> (Parabola)</li>
                <li><b>3rd &amp; 4th-order</b> terms</li>
              </ul>
            </li>
            <li>Drag the <b>expansion point a</b> to see how the approximation changes.</li>
            <li>Turn on <b>Animate</b> to watch the approximation update dynamically.</li>
          </ul>

          <h4>Multivariable Mode</h4>
          <ul>
            <li>Select a predefined 2D function or enter a <b>custom</b> bivariate function.</li>
            <li>Adjust the centers <b>a (x)</b> and <b>b (y)</b> with sliders.</li>
            <li>Animate the path for <b>a</b>, <b>b</b>, or <b>both</b>, then press <b>Play</b>.</li>
            <li>Compare the <b>true surface</b> with its <b>2nd-order Taylor approximation</b> in 3D.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

        # right-aligned Close button (no full-width bar)
        actions = st.container()
        with actions:
            st.markdown('<div class="guide-actions">', unsafe_allow_html=True)
            if st.button("Close", key=f"{key}_close"):
                st.session_state[show_key] = False
            st.markdown('</div>', unsafe_allow_html=True)

# --- Call this once on the page (e.g., right under your title) ---


# Define the run function for this page

st.title("📐 Taylor Series & Optimizer Foundations")

guide_modal("taylor")

mode = st.sidebar.radio("Select Mode", ["📘 Guide", "📈 Univariate", "🌐 Multivariable"])


# tab1, tab2, tab3 = st.tabs(["📘 Guide", "📈 Univariate", "🌐 Multivariable"])

if mode == "📘 Guide":
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

    st.info("ℹ️ You can switch between Univariate and Multivariable using the sidebar selector.")
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

