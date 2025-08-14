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


# Add CSS for overlay and modal with close button

def render_taylor_guide_modal():
    """
    Renders a floating guide modal for the Taylor page.
    - Shows a small '❓ Guide' button
    - On click, opens a centered modal with a top-right X close button
    - Pure Streamlit + CSS (no extra JS/components)
    """
    # --- CSS ---
    st.markdown("""
    <style>
    /* Overlay */
    #guide-overlay {
        position: fixed;
        inset: 0;
        background: rgba(0,0,0,0.55);
        z-index: 1000;
    }
    /* Modal card */
    .guide-modal {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 1001;
        background: #fff;
        width: min(900px, 92vw);
        max-height: 86vh;
        overflow: auto;
        border-radius: 18px;
        padding: 28px 30px 22px 30px;
        box-shadow: 0 20px 60px rgba(0,0,0,.35);
    }

    /* --- CLOSE BUTTON: WRAPPER + BUTTON --- */
    /* 1) Kill the big white bar (wrapper) and position it */
    .guide-modal .stButton {
        position: absolute !important;
        top: 10px !important;
        right: 12px !important;
        width: auto !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    /* 2) Style the actual button */
    .guide-modal .stButton > button {
        font-size: 22px;
        line-height: 1;
        padding: 2px 10px;
        border-radius: 10px;
        background: #f4f4f4;
        border: 1px solid #e6e6e6;
        color: #555;
        cursor: pointer;
    }
    .guide-modal .stButton > button:hover { background:#ececec; color:#111; }

    /* Optional: headings tweak */
    .guide-modal h1, .guide-modal h2, .guide-modal h3, .guide-modal h4 { margin-top: 0; }
    .guide-title { display: flex; gap: 12px; align-items: center; margin-bottom: 8px; }
    .guide-title .emoji { font-size: 28px; }
    .dim-note { color: #6b7280; font-size: 0.95rem; }
    </style>
    """, unsafe_allow_html=True)


    # Show the small launcher button
    if "show_taylor_guide" not in st.session_state:
        st.session_state.show_taylor_guide = False

    cols = st.columns([1, 9])
    with cols[0]:
        if st.button("❓ Guide"):
            st.session_state.show_taylor_guide = True
            st.rerun()

    # Render modal if needed
    if not st.session_state.show_taylor_guide:
        return

    # Dark overlay
    st.markdown('<div id="guide-overlay"></div>', unsafe_allow_html=True)

    # Modal body (use a form so we can catch the close click without JS)
    with st.container():
        st.markdown('<div class="guide-modal">', unsafe_allow_html=True)
        with st.form("taylor_guide_close"):
            # This button is styled by CSS to sit at top-right
            close = st.form_submit_button("×")

            st.markdown("""
            <div class="guide-title">
              <span class="emoji">🧭</span>
              <h2 style="margin:0;">How to use</h2>
            </div>
            """, unsafe_allow_html=True)

            # --- Your guide content (edit freely) ---
            st.markdown("### Univariate Mode")
            st.markdown("""
            - Choose a **predefined function** or select **Custom** to enter your own.
            - Toggle:
              - **1st-order** (Linear)  
              - **2nd-order** (Parabola)  
              - **3rd & 4th-order** terms
            - Drag the **expansion point `a`** slider to see how the approximation changes.
            - Turn on **Animate** to watch the approximation update dynamically.
            """)

            st.markdown("### Multivariable Mode")
            st.markdown("""
            - Select a predefined 2D function or enter a **custom** bivariate function.
            - Adjust the centers **a (x)** and **b (y)** with sliders.
            - **Animate path** for **a**, **b**, or **both**, then press **Play**.
            - Compare the **true surface** with its **2nd-order Taylor approximation** in 3D.
            """)

            st.caption("Tip: Switch between Univariate and Multivariable from the sidebar. Theory lives in the Resources page.")

            if close:
                st.session_state.show_taylor_guide = False
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)


# Define the run function for this page

st.title("📐 Taylor Series & Optimizer Foundations")


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
    render_taylor_guide_modal()
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
    render_taylor_guide_modal()
    # st.markdown("### 🌐 Multivariable Taylor Expansion Visualizer")
    show_multivariable_taylor()

