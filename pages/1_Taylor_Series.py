import streamlit as st
from taylor_expansion import show_univariate_taylor

# st.set_page_config(page_title="Taylor Series", layout="wide")
def run():
  st.title("📐 Taylor Series & Optimizer Foundations")
  
  st.markdown("### 📚 How Taylor Series Explains Optimizers")
  st.markdown("""
  Many optimization algorithms are grounded in the **Taylor series expansion**, 
  which provides a local approximation of a function using its derivatives:
  """)
  
  st.latex(r"""
  f(x + \Delta x) \approx f(x) + \nabla f(x)^T \Delta x
  """)
  st.markdown("""
  - This is the **first-order Taylor expansion**, forming the basis of **Gradient Descent**.  
  - It uses only the **gradient (slope)** to determine the update direction.
  """)
  
  st.latex(r"""
  f(x + \Delta x) \approx f(x) + \nabla f(x)^T \Delta x + \frac{1}{2} \Delta x^T H(x) \Delta x
  """)
  st.markdown("""
  - This is the **second-order Taylor expansion**, used in **Newton's Method**.  
  - It incorporates the **Hessian matrix (curvature)** to adjust both direction and step size, often accelerating convergence.
  """)
  
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
  
  # Show your existing interactive univariate Taylor visualizer
  show_univariate_taylor()

