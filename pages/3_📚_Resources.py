import streamlit as st

def show_resources():
    # st.title("📚 Educational Resources")

    section = st.sidebar.selectbox("Select a topic:", [
        "🔎 Taylor Series",
        "🛠️ Optimization",
        "🌐 External References"
    ])

    if section == "🔎 Taylor Series":
        st.header("📚 How Taylor Series Explains Optimizers")
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



    elif section == "🛠️ Optimization":
        # 1. Optimization Overview
        st.header("1. 🛠️ Optimization Methods")
        st.markdown("""
        Optimization methods aim to **find the minimum (or maximum)** of a function.
        
        There are two major types:
        - **Unconstrained**: No restrictions on variable values.
        - **Constrained**: Variables must satisfy conditions (e.g., \( g(x, y) = 0 \)).

        Common methods include:
        - **Gradient Descent**  
        - **Newton's Method**  
        - **Quasi-Newton (e.g., BFGS)**
        """)
        st.markdown("---")

        # 2. Taylor Series
        st.header("2. 🔎 Taylor Series in Optimization")
        st.latex(r"f(x + \Delta x) \approx f(x) + \nabla f(x)^T \Delta x + \frac{1}{2} \Delta x^T \nabla^2 f(x) \Delta x")
        st.markdown("""
        - **1st-order**: Linear approximation  
        - **2nd-order**: Adds curvature (Newton's Method)
        """)
        st.markdown("---")

        # 3. KKT Conditions
        st.header("3. 🚦 KKT Conditions & Derivatives")

        # 3.1 Objective & Lagrangian
        st.markdown("### 🎯 Objective & Lagrangian")
        st.write("The **objective function** defines what we aim to minimize or maximize:")
        st.latex(r"f(x, y)")
        st.write("The **Lagrangian** incorporates the objective and any constraints using Lagrange multipliers:")
        st.latex(r"\mathcal{L}(x, y, \lambda) = f(x, y) + \lambda \cdot g(x, y)")
        st.markdown("---")

        # 3.2 KKT Conditions
        st.markdown("###  ✅ KKT Conditions")
        st.write("The **Karush–Kuhn–Tucker (KKT) conditions** are necessary for optimality in constrained optimization problems.")
        st.write("They state that the gradient of the Lagrangian must vanish at the optimal point (stationarity condition):")
        st.latex(r"\nabla_{x,y} \mathcal{L}(x, y, \lambda) = 0")
        st.write("For unconstrained problems, this reduces to the gradient of the objective function being zero:")
        st.latex(r"\nabla f(x, y) = 0")
        st.markdown("---")

        # 3.3 Gradient & Hessian
        st.markdown("### 🧮 Gradient & Hessian")
        st.write("The **gradient** vector points in the direction of steepest ascent or descent. At optimality, it becomes zero:")
        st.latex(r"\nabla f(x, y) = \begin{bmatrix} \frac{\partial f}{\partial x} \\\\ \frac{\partial f}{\partial y} \end{bmatrix}")
        st.write("The **Hessian matrix** captures second-order curvature (concavity/convexity) of the function:")
        st.latex(r"""
        \nabla^2 f(x, y) = \begin{bmatrix} 
        \frac{\partial^2 f}{\partial x^2} & \frac{\partial^2 f}{\partial x \partial y} \\\\
        \frac{\partial^2 f}{\partial y \partial x} & \frac{\partial^2 f}{\partial y^2} 
        \end{bmatrix}
        """)
        st.markdown("**Interpretation of the Hessian:**")
        st.markdown("""
        - **Positive definite** → Local **minimum**  
        - **Negative definite** → Local **maximum**  
        - **Indefinite** → **Saddle point**
        """)
        st.markdown("---")

        # 4. Newton Variants
        st.header("4. 🔁 Newton Variants & Quasi-Newton")

        st.markdown("### 📘 Classic Newton vs. Numerical vs. Quasi-Newton")
        st.markdown("Newton's Method is a powerful optimization technique that uses **second-order derivatives** or their approximations to accelerate convergence.")
        
        st.markdown("#### 🧮 Classic Newton (Symbolic)")
        st.markdown("- Uses the **symbolic Hessian matrix** from calculus:")
        st.latex(r"\nabla^2 f(x, y)")
        st.markdown("- ✅ Very efficient and accurate for simple analytic functions (e.g., quadratic, convex).")
        st.markdown("- ⚠️ Can fail or be unstable if the Hessian is singular or badly conditioned.")
        st.markdown("---")
        
        st.markdown("#### 🔢 Numerical Newton")
        st.markdown("- Uses **finite differences** to approximate the Hessian.")
        st.markdown("- No need for symbolic derivatives.")
        st.markdown("- ✅ More robust for complex or unknown functions.")
        st.markdown("- 🐢 Slightly slower due to extra evaluations.")
        st.markdown("---")
        
        st.markdown("#### 🔁 BFGS / L-BFGS (Quasi-Newton)")
        st.markdown("- ✅ Avoids computing the full Hessian.")
        st.markdown("- Builds curvature estimate using gradients:")
        st.latex(r"""
        H_{k+1} = H_k + \frac{y_k y_k^T}{y_k^T s_k} - \frac{H_k s_k s_k^T H_k}{s_k^T H_k s_k}
        """)
        st.markdown("Where:")
        st.latex(r"s_k = x_{k+1} - x_k")
        st.latex(r"y_k = \nabla f(x_{k+1}) - \nabla f(x_k)")
        st.markdown("- 🧠 **BFGS**: High accuracy, stores full matrix.")
        st.markdown("- 🪶 **L-BFGS**: Stores only a few recent updates — ideal for high-dimensional problems.")
        st.markdown("💡 Quasi-Newton methods **approximate** curvature and still converge fast — especially useful for functions like Rosenbrock!")
        st.markdown("---")

        # 5. Learning Rate Insight
        st.header("5. ✏️ Why No Learning Rate in Newton's Method?")
        st.markdown("Newton’s Method computes:")
        st.latex(r"x_{t+1} = x_t - H^{-1} \nabla f(x_t)")
        st.markdown("So it **naturally determines the best step direction and size** — no need for manual tuning like in gradient descent.")
        st.markdown("---")







    elif section == "🌐 External References":
        st.header("🌐 External References")
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
