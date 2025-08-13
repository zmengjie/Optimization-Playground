
# # === Full Modular Streamlit App ===

# import streamlit as st
# import plotly.graph_objects as go
# import streamlit.components.v1 as components
# import base64
# import tempfile
# import time
# import sympy as sp

# import pandas as pd
# import numpy as np

# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# from matplotlib.animation import FuncAnimation
# from matplotlib.animation import PillowWriter
# from matplotlib import cm

# from sympy import symbols, lambdify
# from io import BytesIO
# from PIL import Image
# from mpl_toolkits.mplot3d import Axes3D
# from collections import OrderedDict
# from visualizer import plot_3d_descent, plot_2d_contour

# st.markdown(
#     """
#     <style>
#     /* Make the content container full-width */
#     .block-container {
#         padding-left: 2rem !important;
#         padding-right: 2rem !important;
#         max-width: 100% !important;
#     }

#     /* Optional: Make the page background wider as well */
#     .main {
#         max-width: 100vw !important;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )


# # === Auto-Tuning Simulation Function ===
# def run_auto_tuning_simulation(f_func, optimizer, x0, y0):
#     lr_grid = list(np.logspace(-4, -1, 6))
#     step_grid = [20, 30, 40, 50, 60, 80]
#     best_score = float("inf")
#     best_lr, best_steps = lr_grid[0], step_grid[0]
#     logs = []

#     for lr in lr_grid:
#         for steps in step_grid:
#             x_t, y_t = x0, y0
#             m, v = np.zeros(2), np.zeros(2)
#             beta1, beta2, eps = 0.9, 0.999, 1e-8
#             for t in range(1, steps + 1):
#                 dx = (f_func(x_t + 1e-5, y_t) - f_func(x_t - 1e-5, y_t)) / 2e-5
#                 dy = (f_func(x_t, y_t + 1e-5) - f_func(x_t, y_t - 1e-5)) / 2e-5
#                 grad = np.array([dx, dy])
#                 if np.linalg.norm(grad) < 1e-3:
#                     break
#                 if optimizer == "Adam":
#                     m = beta1 * m + (1 - beta1) * grad
#                     v = beta2 * v + (1 - beta2) * (grad ** 2)
#                     m_hat = m / (1 - beta1 ** t)
#                     v_hat = v / (1 - beta2 ** t)
#                     update = lr * m_hat / (np.sqrt(v_hat) + eps)
#                 elif optimizer == "RMSProp":
#                     v = beta2 * v + (1 - beta2) * (grad ** 2)
#                     update = lr * grad / (np.sqrt(v) + eps)
#                 else:
#                     update = lr * grad
#                 x_t -= update[0]
#                 y_t -= update[1]
#             loss = f_func(x_t, y_t)
#             score = loss + 0.01 * steps
#             logs.append({"lr": lr, "steps": steps, "loss": loss, "score": score})
#             if score < best_score:
#                 best_score, best_lr, best_steps = score, lr, steps
#     st.session_state.df_log = pd.DataFrame(logs)
#     return best_lr, best_steps


# def optimize_path(x0, y0, optimizer, lr, steps, f_func, grad_f=None, hessian_f=None, options=None):
#     path = [(x0, y0)]
#     options = options or {}
#     meta = {} 

#     # --- Special Cases: Return early ---
#     if optimizer == "GradientDescent" and options.get("use_backtracking", False):
#         grad_f_expr = [sp.diff(f_expr, v) for v in (x_sym, y_sym)]
#         path, alphas = backtracking_line_search_sym(f_expr, grad_f_expr, x0, y0)
#         meta["callback_steps"] = len(path)
#         return path, alphas, meta


#     if optimizer == "Newton's Method" and options.get("newton_variant") in ["BFGS", "L-BFGS"]:
#         from scipy.optimize import minimize

#         path_coords = []
#         losses = []

#         def loss_vec(v):
#             return f_func(v[0], v[1])

#         def grad_vec(v):
#             return np.atleast_1d(grad_f(v[0], v[1]))

#         def callback(vk):
#             path_coords.append((vk[0], vk[1]))
#             losses.append(f_func(vk[0], vk[1]))

#         x0_vec = np.array([x0, y0])
#         method = "L-BFGS-B" if options["newton_variant"] == "L-BFGS" else "BFGS"

#         res = minimize(loss_vec, x0_vec, method=method, jac=grad_vec, callback=callback, options={"maxiter": steps})

#         # Fallback if callback was never triggered (e.g., early convergence)
#         if not path_coords:
#             path_coords = [tuple(res.x)]
#             losses = [f_func(*res.x)]

#         meta["res_nit"] = res.nit
#         meta["callback_steps"] = len(path_coords)
#         return path_coords, losses, meta


#     if optimizer == "Simulated Annealing":
#         T, cooling = options.get("T", 2.0), options.get("cooling", 0.95)
#         current = f_func(x0, y0)
#         for _ in range(steps):
#             xn, yn = x0 + np.random.randn(), y0 + np.random.randn()
#             candidate = f_func(xn, yn)
#             if candidate < current or np.random.rand() < np.exp(-(candidate - current)/T):
#                 x0, y0 = xn, yn
#                 current = candidate
#                 path.append((x0, y0))
#             T *= cooling
#         meta["callback_steps"] = len(path)
#         return path, None, meta
    
#     if optimizer == "Genetic Algorithm":
#         pop_size = options.get("pop_size", 20)
#         mutation_std = options.get("mutation_std", 0.3)
#         pop = [np.random.uniform(-5, 5, 2) for _ in range(pop_size)]
#         for _ in range(steps // 2):
#             pop = sorted(pop, key=lambda p: f_func(p[0], p[1]))[:pop_size // 2]
#             children = [np.mean([pop[i], pop[j]], axis=0) + np.random.normal(0, mutation_std, 2)
#                         for i in range(len(pop)) for j in range(i+1, len(pop))][:pop_size // 2]
#             pop += children
#         best = sorted(pop, key=lambda p: f_func(p[0], p[1]))[0]
#         path = [tuple(best)]
#         meta["callback_steps"] = len(path)
#         return path, None, meta

#     # --- Standard Optimizer Loop ---
#     m, v = np.zeros(2), np.zeros(2)
#     beta1, beta2, eps = 0.9, 0.999, 1e-8

#     for t in range(1, steps + 1):
#         x_t, y_t = path[-1]
#         grad = grad_f(x_t, y_t)

#         if optimizer == "Adam":
#             m = beta1 * m + (1 - beta1) * grad
#             v = beta2 * v + (1 - beta2) * (grad ** 2)
#             m_hat = m / (1 - beta1 ** t)
#             v_hat = v / (1 - beta2 ** t)
#             update = lr * m_hat / (np.sqrt(v_hat) + eps)

#         elif optimizer == "RMSProp":
#             v = beta2 * v + (1 - beta2) * (grad ** 2)
#             update = lr * grad / (np.sqrt(v) + eps)

#         elif optimizer == "Newton's Method":
#             variant = options.get("newton_variant", "Classic Newton")
#             if variant == "Classic Newton":
#                 try:
#                     H = hessian_f(x_t, y_t)
#                     H_inv = np.linalg.inv(H)
#                     update = H_inv @ grad
#                 except:
#                     update = grad
#             elif variant == "Numerical Newton":
#                 eps_num = 1e-4
#                 def fx(x_, y_): return f_func(x_, y_)
#                 def second_partial(f, x, y, i, j):
#                     h = eps_num
#                     if i == 0 and j == 0:
#                         return (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / h**2
#                     elif i == 1 and j == 1:
#                         return (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / h**2
#                     else:
#                         return (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h**2)
#                 H = np.array([
#                     [second_partial(fx, x_t, y_t, 0, 0), second_partial(fx, x_t, y_t, 0, 1)],
#                     [second_partial(fx, x_t, y_t, 1, 0), second_partial(fx, x_t, y_t, 1, 1)]
#                 ])
#                 try:
#                     update = np.linalg.inv(H) @ grad
#                 except:
#                     update = grad

#         else:  # Default: GradientDescent
#             update = lr * grad

#         update = np.asarray(update)
#         new_x, new_y = x_t - update[0], y_t - update[1]
#         path.append((new_x, new_y))

#     meta["callback_steps"] = len(path)
#     return path, None, meta

# def backtracking_line_search_sym(f_sym, grad_f_sym, x0, y0, alpha0=1.0, beta=0.5, c=1e-4, max_iters=100):
#     # Declare standard symbols for lambdify
#     x, y = sp.symbols("x y")

#     # Ensure all expressions use x, y (not x_sym, y_sym)
#     f_sym = f_sym.subs({sp.Symbol("x"): x, sp.Symbol("y"): y})
#     grad_f_sym = [g.subs({sp.Symbol("x"): x, sp.Symbol("y"): y}) for g in grad_f_sym]

#     f_lambd = sp.lambdify((x, y), f_sym, modules='numpy')
#     grad_f_lambd = [sp.lambdify((x, y), g, modules='numpy') for g in grad_f_sym]

#     xk, yk = x0, y0
#     path = [(xk, yk)]
#     alphas = []

#     for _ in range(max_iters):
#         gx, gy = grad_f_lambd[0](xk, yk), grad_f_lambd[1](xk, yk)
#         grad_norm = gx**2 + gy**2

#         if grad_norm < 1e-10:
#             break

#         alpha = alpha0
#         while True:
#             x_new = xk - alpha * gx
#             y_new = yk - alpha * gy
#             lhs = f_lambd(x_new, y_new)
#             rhs = f_lambd(xk, yk) - c * alpha * grad_norm
#             if lhs <= rhs or alpha < 1e-8:
#                 break
#             alpha *= beta

#         xk, yk = xk - alpha * gx, yk - alpha * gy
#         path.append((xk, yk))
#         alphas.append(alpha)

#     return path, alphas

#     # === Main Area: Title & Playground ===
# st.title("🚀 Optimizer Visual Playground")


# x, y, w = sp.symbols("x y w")
# x_sym, y_sym, w_sym = sp.symbols("x y w")

# predefined_funcs = {
#     "Quadratic Bowl": (x**2 + y**2, [], "Convex bowl with global minimum at origin."),
#     "Saddle": (x**2 - y**2, [], "Saddle point at origin, non-convex."),
#     "Rosenbrock": ((1 - x)**2 + 100 * (y - x**2)**2, [], "Banana-shaped curved valley; classic non-convex test function."),
#     "Constrained Circle": (x * y, [x + y - 1], "Constrained optimization with linear constraint x + y = 1."),
#     "Double Constraint": (x**2 + y**2, [x + y - 1, x**2 + y**2 - 4], "Intersection of circle and line constraints."),
#     "Multi-Objective": (w * ((x - 1)**2 + (y - 2)**2) + (1 - w) * ((x + 2)**2 + (y + 1)**2), [], "Weighted sum of two quadratic objectives."),
#     "Ackley": (-20*sp.exp(-0.2*sp.sqrt(0.5*(x**2 + y**2))) - sp.exp(0.5*(sp.cos(2*sp.pi*x) + sp.cos(2*sp.pi*y))) + sp.E + 20, [], "Highly multimodal, non-convex function."),
#     "Rastrigin": (20 + x**2 - 10*sp.cos(2*sp.pi*x) + y**2 - 10*sp.cos(2*sp.pi*y), [], "Non-convex with many local minima (periodic)."),
#     "Styblinski-Tang": (0.5*((x**4 - 16*x**2 + 5*x) + (y**4 - 16*y**2 + 5*y)), [], "Non-convex with many local minima."),
#     "Sphere": (x**2 + y**2, [], "Simple convex function; global minimum at origin."),
#     "Himmelblau": ((x**2 + y - 11)**2 + (x + y**2 - 7)**2, [], "Non-convex with four global minima."),
#     "Booth": ((x + 2*y - 7)**2 + (2*x + y - 5)**2, [], "Simple convex quadratic function."),
#     "Beale": ((1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2, [], "Non-convex with several valleys.")

# }


# tab1, tab2, tab3 = st.tabs(["📘 Guide", "🧪 Optimizer Playground", "📐 Symbolic Analysis"])

# with tab1:
#     st.title("📘 Optimization Guide")

#     # 1. Optimization Overview
#     st.header("1. 🛠️ Optimization Methods")
#     st.markdown("""
#     Optimization methods aim to **find the minimum (or maximum)** of a function.
    
#     There are two major types:
#     - **Unconstrained**: No restrictions on variable values.
#     - **Constrained**: Variables must satisfy conditions (e.g., \( g(x, y) = 0 \)).

#     Common methods include:
#     - **Gradient Descent**  
#     - **Newton's Method**  
#     - **Quasi-Newton (e.g., BFGS)**
#     """)
#     st.markdown("---")

#     # 2. Taylor Series
#     st.header("2. 🔎 Taylor Series in Optimization")
#     st.latex(r"f(x + \Delta x) \approx f(x) + \nabla f(x)^T \Delta x + \frac{1}{2} \Delta x^T \nabla^2 f(x) \Delta x")
#     st.markdown("""
#     - **1st-order**: Linear approximation  
#     - **2nd-order**: Adds curvature (Newton's Method)
#     """)
#     st.markdown("---")

#     # 3. KKT Conditions
#     st.header("3. 🚦 KKT Conditions & Derivatives")

#     # 3.1 Objective & Lagrangian
#     st.markdown("### 🎯 Objective & Lagrangian")
#     st.write("The **objective function** defines what we aim to minimize or maximize:")
#     st.latex(r"f(x, y)")
#     st.write("The **Lagrangian** incorporates the objective and any constraints using Lagrange multipliers:")
#     st.latex(r"\mathcal{L}(x, y, \lambda) = f(x, y) + \lambda \cdot g(x, y)")
#     st.markdown("---")

#     # 3.2 KKT Conditions
#     st.markdown("###  ✅ KKT Conditions")
#     st.write("The **Karush–Kuhn–Tucker (KKT) conditions** are necessary for optimality in constrained optimization problems.")
#     st.write("They state that the gradient of the Lagrangian must vanish at the optimal point (stationarity condition):")
#     st.latex(r"\nabla_{x,y} \mathcal{L}(x, y, \lambda) = 0")
#     st.write("For unconstrained problems, this reduces to the gradient of the objective function being zero:")
#     st.latex(r"\nabla f(x, y) = 0")
#     st.markdown("---")

#     # 3.3 Gradient & Hessian
#     st.markdown("### 🧮 Gradient & Hessian")
#     st.write("The **gradient** vector points in the direction of steepest ascent or descent. At optimality, it becomes zero:")
#     st.latex(r"\nabla f(x, y) = \begin{bmatrix} \frac{\partial f}{\partial x} \\\\ \frac{\partial f}{\partial y} \end{bmatrix}")
#     st.write("The **Hessian matrix** captures second-order curvature (concavity/convexity) of the function:")
#     st.latex(r"""
#     \nabla^2 f(x, y) = \begin{bmatrix} 
#     \frac{\partial^2 f}{\partial x^2} & \frac{\partial^2 f}{\partial x \partial y} \\\\
#     \frac{\partial^2 f}{\partial y \partial x} & \frac{\partial^2 f}{\partial y^2} 
#     \end{bmatrix}
#     """)
#     st.markdown("**Interpretation of the Hessian:**")
#     st.markdown("""
#     - **Positive definite** → Local **minimum**  
#     - **Negative definite** → Local **maximum**  
#     - **Indefinite** → **Saddle point**
#     """)
#     st.markdown("---")

#     # 4. Newton Variants
#     st.header("4. 🔁 Newton Variants & Quasi-Newton")

#     st.markdown("### 📘 Classic Newton vs. Numerical vs. Quasi-Newton")
#     st.markdown("Newton's Method is a powerful optimization technique that uses **second-order derivatives** or their approximations to accelerate convergence.")
    
#     st.markdown("#### 🧮 Classic Newton (Symbolic)")
#     st.markdown("- Uses the **symbolic Hessian matrix** from calculus:")
#     st.latex(r"\nabla^2 f(x, y)")
#     st.markdown("- ✅ Very efficient and accurate for simple analytic functions (e.g., quadratic, convex).")
#     st.markdown("- ⚠️ Can fail or be unstable if the Hessian is singular or badly conditioned.")
#     st.markdown("---")
    
#     st.markdown("#### 🔢 Numerical Newton")
#     st.markdown("- Uses **finite differences** to approximate the Hessian.")
#     st.markdown("- No need for symbolic derivatives.")
#     st.markdown("- ✅ More robust for complex or unknown functions.")
#     st.markdown("- 🐢 Slightly slower due to extra evaluations.")
#     st.markdown("---")
    
#     st.markdown("#### 🔁 BFGS / L-BFGS (Quasi-Newton)")
#     st.markdown("- ✅ Avoids computing the full Hessian.")
#     st.markdown("- Builds curvature estimate using gradients:")
#     st.latex(r"""
#     H_{k+1} = H_k + \frac{y_k y_k^T}{y_k^T s_k} - \frac{H_k s_k s_k^T H_k}{s_k^T H_k s_k}
#     """)
#     st.markdown("Where:")
#     st.latex(r"s_k = x_{k+1} - x_k")
#     st.latex(r"y_k = \nabla f(x_{k+1}) - \nabla f(x_k)")
#     st.markdown("- 🧠 **BFGS**: High accuracy, stores full matrix.")
#     st.markdown("- 🪶 **L-BFGS**: Stores only a few recent updates — ideal for high-dimensional problems.")
#     st.markdown("💡 Quasi-Newton methods **approximate** curvature and still converge fast — especially useful for functions like Rosenbrock!")
#     st.markdown("---")

#     # 5. Learning Rate Insight
#     st.header("5. ✏️ Why No Learning Rate in Newton's Method?")
#     st.markdown("Newton’s Method computes:")
#     st.latex(r"x_{t+1} = x_t - H^{-1} \nabla f(x_t)")
#     st.markdown("So it **naturally determines the best step direction and size** — no need for manual tuning like in gradient descent.")
#     st.markdown("---")




# with tab2:
#     st.title("🧪 Optimizer Playground")
#     col_left, col_right = st.columns([1, 1])

#     with col_left:
#         mode = st.radio("Function Source", ["Predefined", "Custom"])
#         func_name = st.selectbox("Function", list(predefined_funcs.keys())) if mode == "Predefined" else None
#         expr_str = st.text_input("Enter function f(x,y):", "x**2 + y**2") if mode == "Custom" else ""
#         w_val = st.slider("Weight w (Multi-Objective)", 0.0, 1.0, 0.5) if func_name == "Multi-Objective" else None

#         optimizers = ["GradientDescent",  "Momentum", "Adam", "RMSProp", "Newton's Method", "Simulated Annealing", "Genetic Algorithm"]
#         optimizer = st.selectbox("Optimizer", optimizers)
        
#         options = {}

#         if optimizer == "Newton's Method":
#             newton_variant = st.selectbox("Newton Variant", ["Classic Newton", "Numerical Newton", "BFGS", "L-BFGS"])
#             options["newton_variant"] = newton_variant
#         elif optimizer == "Simulated Annealing":
#             options["T"] = st.slider("Initial Temperature (T)", 0.1, 10.0, 2.0)
#             options["cooling"] = st.slider("Cooling Rate", 0.80, 0.99, 0.95)
#         elif optimizer == "Genetic Algorithm":
#             options["pop_size"] = st.slider("Population Size", 10, 100, 20)
#             options["mutation_std"] = st.slider("Mutation Std Dev", 0.1, 1.0, 0.3)

#         auto_tune = False

#         if optimizer in ["GradientDescent", "Adam", "RMSProp"]:
#             if optimizer == "GradientDescent":
#                 use_backtracking = st.checkbox("🔍 Use Backtracking Line Search", value=False)
#                 options["use_backtracking"] = use_backtracking

#                 if use_backtracking:
#                     st.checkbox("⚙️ Auto-Tune Learning Rate & Steps", value=False, disabled=True, key="auto_tune_disabled")
#                     auto_tune = False
#                     if "auto_tune_checkbox" in st.session_state:
#                         st.session_state["auto_tune_checkbox"] = False
#                     st.caption("ℹ️ Disabled because backtracking search dynamically adjusts step size.")
#                 else:
#                     auto_tune = st.checkbox("⚙️ Auto-Tune Learning Rate & Steps", value=True, key="auto_tune_checkbox")
#             else:
#                 auto_tune = st.checkbox("⚙️ Auto-Tune Learning Rate & Steps", value=True, key="auto_tune_checkbox")

#         elif optimizer == "Newton's Method":
#             st.checkbox("⚙️ Auto-Tune Learning Rate & Steps", value=False, disabled=True, key="auto_tune_disabled")
#             auto_tune = False
#             if "auto_tune_checkbox" in st.session_state:
#                 st.session_state["auto_tune_checkbox"] = False
#             st.caption("ℹ️ Auto-tune is not applicable to Newton’s Method.")
            
#         start_xy_defaults = {
#             "Quadratic Bowl": (-3.0, 3.0), "Saddle": (-2.0, 2.0), "Rosenbrock": (-1.5, 1.5),
#             "Constrained Circle": (0.5, 0.5), "Double Constraint": (-1.5, 1.5),
#             "Multi-Objective": (0.0, 0.0), "Ackley": (2.0, -2.0), "Rastrigin": (3.0, 3.0),
#             "Styblinski-Tang": (-2.5, -2.5), "Sphere": (-3.0, 3.0), "Himmelblau": (0.0, 0.0),
#             "Booth": (1.0, 1.0), "Beale": (-2.0, 2.0)
#         }
#         default_x, default_y = start_xy_defaults.get(func_name, (-3.0, 3.0))
#         default_lr = 0.005
#         default_steps = 50


#         # Make sure this is properly initialized before calling the function
#         if auto_tune:
#             # Ensure function is properly initialized
#             x_sym, y_sym, w_sym = sp.symbols("x y w")
#             symbolic_expr = predefined_funcs[func_name][0]

#             if func_name == "Multi-Objective" and w_val is not None:
#                 symbolic_expr = symbolic_expr.subs(w_sym, w_val)

#             # Lambdify the symbolic function to make it usable
#             f_lambdified = sp.lambdify((x_sym, y_sym), symbolic_expr, modules="numpy")

#             # Call the auto-tuning simulation function
#             best_lr, best_steps = run_auto_tuning_simulation(f_lambdified, optimizer, default_x, default_y)
#             # st.success(f"✅ Auto-tuned: lr={best_lr}, steps={best_steps}, start=({default_x},{default_y})")
#             default_lr, default_steps = best_lr, best_steps

#         if auto_tune:
#             # Run the auto-tuning simulation to find the best lr and steps
#             best_lr, best_steps = run_auto_tuning_simulation(f_lambdified, optimizer, default_x, default_y)

#             # Update session state with the tuned values
#             st.session_state.lr = best_lr
#             st.session_state.steps = best_steps
#             st.session_state.start_x = default_x
#             st.session_state.start_y = default_y
#             st.session_state.params_set = True  # Flag to indicate parameters are set

#             # Provide feedback to the user
#             st.success(f"✅ Auto-tuned: lr={best_lr}, steps={best_steps}, start=({default_x},{default_y})")

#         # Ensure session state values exist to avoid AttributeError
#         if "params_set" not in st.session_state or st.button("🔄 Reset to Auto-Tuned"):
#             # Reset session state to the auto-tuned values when "Reset to Auto-Tuned" is clicked
#             st.session_state.lr = default_lr
#             st.session_state.steps = default_steps
#             st.session_state.start_x = default_x
#             st.session_state.start_y = default_y
#             st.session_state.params_set = True

#         # Final user inputs for learning rate and steps
#         if not (
#             optimizer == "GradientDescent" and options.get("use_backtracking", False)
#         ) and optimizer != "Newton's Method":
#             lr = st.selectbox("Learning Rate", sorted(set([0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, default_lr])), index=0, key="lr")
#             steps = st.slider("Steps", 10, 100, value=st.session_state.get("steps", 50), key="steps")
#         elif optimizer == "Newton's Method":
#             lr = None
#             steps = None
#             st.info("📌 Newton’s Method computes its own step size using the Hessian inverse — learning rate is not needed.")
#         elif optimizer == "GradientDescent" and options.get("use_backtracking", False):
#             lr = None
#             steps = None
#             st.info("📌 Using Backtracking Line Search — no need to set learning rate or step count.")

#         # Ensure session state values exist for initial positions
#         if "start_x" not in st.session_state:
#             st.session_state["start_x"] = -3.0
#         if "start_y" not in st.session_state:
#             st.session_state["start_y"] = 3.0

#         # Sliders for the initial starting positions
#         st.slider("Initial x", -5.0, 5.0, st.session_state.start_x, key="start_x")
#         st.slider("Initial y", -5.0, 5.0, st.session_state.start_y, key="start_y")

#         # Animation toggle
#         show_animation = st.checkbox("🎮 Animate Descent Steps", key="show_animation")

#         # Display auto-tuning trial log if available
#         # if auto_tune and optimizer in ["GradientDescent", "Adam", "RMSProp"]:
#         #     with col_right:
#         #         st.markdown("### 📊 Auto-Tuning Trial Log")
#         #         if "df_log" in st.session_state:
#         #             st.dataframe(st.session_state.df_log.sort_values("score").reset_index(drop=True))
#         #             st.markdown("""
#         #             **🧠 How to Read Score:**
#         #             - `score = final_loss + penalty × steps`
#         #             - ✅ Lower score is better (fast and accurate convergence).
#         #             """)
#         #         else:
#         #             st.info("Auto-tuning not yet triggered.")


#     if mode == "Predefined":
#         f_expr, constraints, description = predefined_funcs[func_name]
#         if func_name == "Multi-Objective" and w_val is not None:
#             f_expr = f_expr.subs(w, w_val)
#     else:
#         try:
#             f_expr = sp.sympify(expr_str)
#             constraints = []
#             description = "Custom function."
#         except:
#             st.error("Invalid expression.")
#             st.stop()

#     f_func = sp.lambdify((x_sym, y_sym), f_expr, modules=["numpy"])

#     grad_f = lambda x0, y0: np.array([
#         (f_func(x0 + 1e-5, y0) - f_func(x0 - 1e-5, y0)) / 2e-5,
#         (f_func(x0, y0 + 1e-5) - f_func(x0, y0 - 1e-5)) / 2e-5
#     ])

#     def hessian_f(x0, y0):
#         hess_expr = sp.hessian(f_expr, (x, y))
#         hess_func = sp.lambdify((x, y), hess_expr, modules=["numpy"])
#         return np.array(hess_func(x0, y0))


#     # st.markdown(f"### 📘 Function Description:\n> {description}")

#     # Setup for symbolic Lagrangian and KKT (if needed)
#     L_expr = f_expr + sum(sp.Symbol(f"lambda{i+1}") * g for i, g in enumerate(constraints))
#     # grad_L = [sp.diff(L_expr, v) for v in (x, y)]
#     grad_L = [sp.diff(L_expr, v) for v in (x_sym, y_sym)]
#     kkt_conditions = grad_L + constraints


#     def simulate_optimizer(opt_name, f_expr, lr=0.01, steps=50):
#         f_func = sp.lambdify((x_sym, y_sym), f_expr, modules=["numpy"])
#         # f_func = sp.lambdify((x, y), f_expr, modules="numpy")
#         x0, y0 = -3, 3
#         path = [(x0, y0)]
#         m, v = 0, 0
#         for t in range(1, steps + 1):
#             x_t, y_t = path[-1]
#             dx = (f_func(x_t + 1e-5, y_t) - f_func(x_t - 1e-5, y_t)) / 2e-5
#             dy = (f_func(x_t, y_t + 1e-5) - f_func(x_t, y_t - 1e-5)) / 2e-5
#             g = np.array([dx, dy])
#             if opt_name == "Adam":
#                 m = 0.9 * m + 0.1 * g
#                 v = 0.999 * v + 0.001 * (g ** 2)
#                 m_hat = m / (1 - 0.9 ** t)
#                 v_hat = v / (1 - 0.999 ** t)
#                 update = lr * m_hat / (np.sqrt(v_hat) + 1e-8)
#             elif opt_name == "RMSProp":
#                 v = 0.999 * v + 0.001 * (g ** 2)
#                 update = lr * g / (np.sqrt(v) + 1e-8)
#             elif opt_name == "Newton's Method":
#                 hess = sp.hessian(f_expr, (x, y))
#                 hess_func = sp.lambdify((x, y), hess, modules="numpy")
#                 try:
#                     H = np.array(hess_func(x_t, y_t))
#                     H_inv = np.linalg.inv(H)
#                     update = H_inv @ g
#                 except:
#                     update = g
#             else:
#                 update = lr * g
#             x_new, y_new = x_t - update[0], y_t - update[1]
#             path.append((x_new, y_new))
#         final_x, final_y = path[-1]
#         grad_norm = np.linalg.norm(g)
#         return {
#             "Optimizer": opt_name,
#             "Final Value": round(f_func(final_x, final_y), 4),
#             "Gradient Norm": round(grad_norm, 4),
#             "Steps": len(path) - 1
#         }
#     g_funcs = [sp.lambdify((x_sym, y_sym), g, modules=["numpy"]) for g in constraints]  
#     # g_funcs = [sp.lambdify((x, y), g, modules=["numpy"]) for g in constraints]
#     f_func = sp.lambdify((x_sym, y_sym), f_expr, modules=["numpy"])
#     # f_func = sp.lambdify((x, y), f_expr, modules=["numpy"])
#     grad_f = lambda x0, y0: np.array([
#         (f_func(x0 + 1e-5, y0) - f_func(x0 - 1e-5, y0)) / 2e-5,
#         (f_func(x0, y0 + 1e-5) - f_func(x0, y0 - 1e-5)) / 2e-5
#     ])

#     def hessian_f(x0, y0):
#         hess_expr = sp.hessian(f_expr, (x, y))
#         hess_func = sp.lambdify((x, y), hess_expr, modules=["numpy"])
#         return np.array(hess_func(x0, y0))

#     # === Pull final values from session_state
#     start_x = st.session_state.get("start_x", -3.0)
#     start_y = st.session_state.get("start_y", 3.0)
#     lr = st.session_state.get("lr", 0.01)
#     steps = st.session_state.get("steps", 50)


#     path, alpha_log, meta = optimize_path(
#         start_x, start_y,
#         optimizer=optimizer,
#         lr=lr,
#         steps=steps,
#         f_func=f_func,
#         grad_f=grad_f,
#         hessian_f=hessian_f,
#         options=options
#     )

#     xs, ys = zip(*path)
#     Z_path = [f_func(xp, yp) for xp, yp in path]

#     x_vals = np.linspace(-5, 5, 200)
#     y_vals = np.linspace(-5, 5, 200)
#     X, Y = np.meshgrid(x_vals, y_vals)
#     Z = f_func(X, Y)

#     # --- Taylor Expansion Toggle ---

#     show_taylor = st.checkbox("📐 Show Taylor Approximation at (a, b)", value=False)

#     show_2nd = False
#     Z_t1 = None
#     Z_t2 = None

#     a_val, b_val = None, None  # set early to avoid NameError
#     expansion_point = None

#     # expansion_point = (a_val, b_val) if show_taylor else None

#     if show_taylor:
#         st.markdown("**Taylor Expansion Center (a, b)**")

#         # --- Initialize session state on first run ---
#         if "a_val" not in st.session_state or "b_val" not in st.session_state:
#             if float(start_x) == 0.0 and float(start_y) == 0.0 and func_name != "Quadratic Bowl":
#                 st.session_state.a_val = 0.1
#                 st.session_state.b_val = 0.1
#                 st.info("🔁 Auto-shifted expansion point from (0,0) → (0.1, 0.1)")
#             else:
#                 st.session_state.a_val = float(start_x)
#                 st.session_state.b_val = float(start_y)

#         # --- Sliders using session state ---
#         a_val = st.slider("a (expansion x)", -5.0, 5.0, value=st.session_state.a_val, step=0.1, key="a_val")
#         b_val = st.slider("b (expansion y)", -5.0, 5.0, value=st.session_state.b_val, step=0.1, key="b_val")

#         # --- Prevent user from going back to (0,0) if it's not allowed ---
#         if a_val == 0.0 and b_val == 0.0 and func_name != "Quadratic Bowl":
#             st.warning("🚫 (0,0) is a singular point. Auto-shifting again to (0.1, 0.1)")
#             a_val = 0.1
#             b_val = 0.1
#             st.session_state.a_val = 0.1
#             st.session_state.b_val = 0.1

#         expansion_point = (a_val, b_val)
#         show_2nd = st.checkbox("Include 2nd-order terms", value=True)


#         # --- Symbolic derivatives ---
#         grad_fx = [sp.diff(f_expr, var) for var in (x_sym, y_sym)]
#         hess_fx = sp.hessian(f_expr, (x_sym, y_sym))

#         subs = {x_sym: a_val, y_sym: b_val}
#         st.text(f"📌 Taylor center used: (a={a_val:.3f}, b={b_val:.3f})")

#         f_ab = float(f_expr.subs(subs))
#         grad_vals = [float(g.subs(subs)) for g in grad_fx]
#         hess_vals = hess_fx.subs(subs)
#         Hxx = float(hess_vals[0, 0])
#         Hxy = float(hess_vals[0, 1])
#         Hyy = float(hess_vals[1, 1])

#         # dx, dy remain symbolic
#         dx, dy = x_sym - a_val, y_sym - b_val

#         # Fully numeric constants in expression
#         T1_expr = f_ab + grad_vals[0]*dx + grad_vals[1]*dy
#         T2_expr = T1_expr + 0.5 * (Hxx*dx**2 + 2*Hxy*dx*dy + Hyy*dy**2)

#         # Numerical evaluation for plotting
#         t1_np = sp.lambdify((x_sym, y_sym), T1_expr, "numpy")
#         t2_np = sp.lambdify((x_sym, y_sym), T2_expr, "numpy") if show_2nd else None
#         Z_t1 = t1_np(X, Y)


#         if show_2nd:
#             try:
#                 Z_t2 = t2_np(X, Y)

#                 # --- Safety checks ---
#                 if isinstance(Z_t2, (int, float, np.number)):
#                     st.warning(f"❌ Z_t2 is scalar: {Z_t2}. Skipping.")
#                     Z_t2 = None
#                 else:
#                     Z_t2 = np.array(Z_t2, dtype=np.float64)

#                     if Z_t2.ndim != 2:
#                         st.warning(f"❌ Z_t2 is not 2D — shape: {Z_t2.shape}")
#                         Z_t2 = None
#                     elif Z_t2.shape != (len(y_vals), len(x_vals)):
#                         if Z_t2.shape == (len(x_vals), len(y_vals)):
#                             Z_t2 = Z_t2.T
#                         else:
#                             st.warning(f"❌ Z_t2 shape mismatch: {Z_t2.shape}")
#                             Z_t2 = None
#                     elif np.isnan(Z_t2).any():
#                         st.warning("❌ Z_t2 contains NaNs.")
#                         Z_t2 = None

#             except Exception as e:
#                 st.warning(f"⚠️ Failed to evaluate 2nd-order Taylor surface: {e}")
#                 Z_t2 = None

#         # Z_t2 = t2_np(X, Y) if show_2nd else None

#         # ✅ Display symbolic formula as LaTeX
#         st.markdown("### ✏️ Taylor Approximation Formula at \\( (a, b) = ({:.1f}, {:.1f}) \\)".format(a_val, b_val))

#         fx, fy = grad_vals
#         Hxx, Hxy, Hyy = hess_vals[0, 0], hess_vals[0, 1], hess_vals[1, 1]

#         T1_latex = f"f(x, y) \\approx {f_ab:.3f} + ({fx}) (x - {a_val}) + ({fy}) (y - {b_val})"
#         T2_latex = (
#             f"{T1_latex} + \\frac{{1}}{{2}}({Hxx}) (x - {a_val})^2 + "
#             f"{Hxy} (x - {a_val})(y - {b_val}) + "
#             f"\\frac{{1}}{{2}}({Hyy}) (y - {b_val})^2"
#         )

#         st.latex(T1_latex)
#         if show_2nd:
#             st.latex(T2_latex)


#     st.markdown("### 📈 3D View")

#     if show_2nd:
#         try:
#             if Z_t2 is not None:
#                 # ❗ Reject raw scalar values
#                 if isinstance(Z_t2, (int, float)):
#                     st.warning(f"❌ Z_t2 is a scalar ({Z_t2}), not an array.")
#                     Z_t2 = None

#                 Z_t2 = np.array(Z_t2, dtype=np.float64)

#                 if Z_t2.ndim != 2:
#                     st.warning(f"❌ Z_t2 is not 2D — shape: {Z_t2.shape}")
#                     Z_t2 = None
#                 elif np.isnan(Z_t2).any():
#                     st.warning("❌ Z_t2 contains NaNs.")
#                     Z_t2 = None
#                 elif Z_t2.shape != (len(y_vals), len(x_vals)):
#                     if Z_t2.shape == (len(x_vals), len(y_vals)):
#                         Z_t2 = Z_t2.T
#                     else:
#                         st.warning(f"❌ Z_t2 shape mismatch: {Z_t2.shape} vs mesh ({len(y_vals)}, {len(x_vals)})")
#                         Z_t2 = None
#         except Exception as e:
#             st.warning(f"❌ Error processing Z_t2: {e}")
#             Z_t2 = None

            
#     plot_3d_descent(
#         x_vals=x_vals,
#         y_vals=y_vals,
#         Z=Z,
#         path=path,
#         Z_path=Z_path,
#         Z_t1=Z_t1,
#         Z_t2=Z_t2,
#         show_taylor=show_taylor,
#         show_2nd=show_2nd,
#         expansion_point=expansion_point,
#         f_func=f_func
#     )


#     st.markdown("### 🗺️ 2D View")
#     plot_2d_contour(
#         x_vals=x_vals,
#         y_vals=y_vals,
#         Z=Z,
#         path=path,
#         g_funcs=g_funcs if constraints else None,
#         X=X, Y=Y,
#         Z_t2=Z_t2,
#         show_2nd=show_2nd,
#         expansion_point=expansion_point
#     )



#     if show_taylor:
#         st.caption("🔺 Red = 1st-order Taylor, 🔷 Blue = 2nd-order Taylor, 🟢 Green = true surface")

#     if show_animation:
#         frames = []
#         fig_anim, ax_anim = plt.subplots(figsize=(5, 4))

#         for i in range(1, len(path) + 1):
#             ax_anim.clear()
#             ax_anim.contour(X, Y, Z, levels=30, cmap="viridis")
#             ax_anim.plot(*zip(*path[:i]), 'r*-')
#             ax_anim.set_xlim([-5, 5])
#             ax_anim.set_ylim([-5, 5])
#             ax_anim.set_title(f"Step {i}/{len(path)-1}")

#             buf = BytesIO()
#             fig_anim.savefig(buf, format='png', dpi=100)  # optional: set dpi
#             buf.seek(0)
#             frames.append(Image.open(buf).convert("P"))  # convert to palette for GIF efficiency
#             buf.close()

#         gif_buf = BytesIO()
#         frames[0].save(
#             gif_buf, format="GIF", save_all=True,
#             append_images=frames[1:], duration=300, loop=0
#         )
#         gif_buf.seek(0)
#         st.image(gif_buf, caption="📽️ Animated Descent Path", use_container_width=True)


# # with tab3:
# #     st.header("🧰 Optimizer Diagnostic Tools")

# #     col1, col2 = st.columns(2)

# #     with col1:
# #         st.markdown("#### 📊 Optimizer Comparison")

# #         start_x = st.session_state.get("start_x", 0.0)
# #         start_y = st.session_state.get("start_y", 0.0)

# #         selected_opts = st.multiselect(
# #             "Optimizers",
# #             ["GradientDescent", "Adam", "RMSProp", "Newton's Method"],
# #             default=["GradientDescent"],
# #             key="compare"
# #         )
# #         fig_comp, ax_comp = plt.subplots(figsize=(4, 3))

# #         results = []
# #         summary_results = []

# #         for opt in selected_opts:
# #             path_opt, losses, meta= optimize_path(  # ✅ unpack both outputs
# #                 start_x,
# #                 start_y,
# #                 optimizer=opt,
# #                 lr=lr,
# #                 steps=steps,
# #                 f_func=f_func,
# #                 grad_f=grad_f,
# #                 hessian_f=hessian_f,
# #                 options=options
# #             )

# #             zs_coords = path_opt
# #             zs_vals = losses if losses is not None and len(losses) > 0 else [f_func(xp, yp) for xp, yp in zs_coords]
# #             grad_norm = float(np.linalg.norm(grad_f(*zs_coords[-1])))

# #             results.append((opt, zs_vals))

# #             summary_results.append({
# #                 "Optimizer": opt,
# #                 "Final Value": np.round(zs_vals[-1], 4),
# #                 "Gradient Norm": np.round(grad_norm, 4),
# #                 "Steps": len(zs_vals),
# #                 "Actual Steps": meta.get("res_nit", "N/A"),   # ✅ 使用 meta.get()
# #                 "Logged Steps": meta.get("callback_steps", "N/A")
# #             })

# #         # Sort results by final loss
# #         results.sort(key=lambda x: x[1][-1])

# #         for opt, zs in results:
# #             if zs is not None and len(zs) > 0:
# #                 ax_comp.plot(zs, label=f"{opt} ({len(zs)} steps)", marker="o", markersize=2)
# #             else:
# #                 st.warning(f"⚠️ No convergence data for {opt}")

# #         ax_comp.set_title("Convergence")
# #         ax_comp.set_xlabel("Step")
# #         ax_comp.set_ylabel("f(x, y)")
# #         ax_comp.set_ylim(bottom=0)
# #         ax_comp.legend()
# #         st.pyplot(fig_comp)

# #         # Show summary table
# #         # st.markdown("#### 📋 Optimizer Summary Table")
# #         # df_summary = pd.DataFrame(summary_results)
# #         # st.dataframe(df_summary)



# #         st.markdown("#### 🔥 Gradient Norm Heatmap")
# #         # norm_grad = np.sqrt((np.gradient(Z, axis=0))**2 + (np.gradient(Z, axis=1))**2)
# #         # Temporary definition in case not computed above
# #         if 'Z_loss' not in locals():
# #             x_range = np.linspace(-5, 5, 50)
# #             y_range = np.linspace(-5, 5, 50)
# #             X_loss, Y_loss = np.meshgrid(x_range, y_range)
# #             Z_loss = (X_loss - 2)**2 + (Y_loss + 1)**2  # Default MSE-like loss

# #         norm_grad = np.sqrt((np.gradient(Z_loss, axis=0))**2 + (np.gradient(Z_loss, axis=1))**2)

# #         fig3, ax3 = plt.subplots()
# #         heat = ax3.imshow(norm_grad, extent=[-5, 5, -5, 5], origin='lower', cmap='plasma')
# #         fig3.colorbar(heat, ax=ax3, label="‖∇f‖")
# #         ax3.set_title("‖∇f(x, y)‖")
# #         st.pyplot(fig3)

# #     with col2:
# #         st.markdown("#### 🌄 Loss Surface")

# #         loss_type = st.radio("Loss Type", ["MSE", "Log Loss", "Cross Entropy", "Custom"])

# #         # Create input grid
# #         x_range = np.linspace(-5, 5, 50)  # reduced resolution for arrows
# #         y_range = np.linspace(-5, 5, 50)
# #         X_loss, Y_loss = np.meshgrid(x_range, y_range)

# #         # Compute Z surface and manually define minimum
# #         if loss_type == "MSE":
# #             Z_loss = (X_loss - 2)**2 + (Y_loss + 1)**2
# #             min_x, min_y = 2, -1
# #         elif loss_type == "Log Loss":
# #             Z_loss = np.log(1 + np.exp(-(X_loss + Y_loss)))
# #             min_x, min_y = 5, -5
# #         elif loss_type == "Cross Entropy":
# #             p = 1 / (1 + np.exp(-(X_loss + Y_loss)))
# #             Z_loss = -p * np.log(p + 1e-8) - (1 - p) * np.log(1 - p + 1e-8)
# #             min_x, min_y = 5, -5
# #         else:
# #             Z_loss = np.sin(X_loss) * np.cos(Y_loss)
# #             min_x, min_y = 0, 0

# #         # Compute gradients (numerical partial derivatives)
# #         dZ_dx, dZ_dy = np.gradient(Z_loss, x_range, y_range)

# #         # === Side-by-side plots ===
# #         col1, col2 = st.columns(2)

# #         with col1:
# #             fig3d = plt.figure(figsize=(5, 4))
# #             ax3d = fig3d.add_subplot(111, projection='3d')
# #             ax3d.plot_surface(X_loss, Y_loss, Z_loss, cmap='viridis', edgecolor='none', alpha=0.9)
# #             ax3d.scatter(min_x, min_y, np.min(Z_loss), color='red', s=50, label='Min')
# #             ax3d.set_title(f"{loss_type} Surface")
# #             ax3d.legend()
# #             st.pyplot(fig3d)

# #         with col2:
# #             fig2d, ax2d = plt.subplots(figsize=(5, 4))
# #             contour = ax2d.contourf(X_loss, Y_loss, Z_loss, levels=30, cmap='viridis')
# #             ax2d.plot(min_x, min_y, 'ro', label='Min')
# #             # Overlay gradient arrows (negative for descent)
# #             ax2d.quiver(X_loss, Y_loss, -dZ_dx, -dZ_dy, color='white', alpha=0.7, scale=50)
# #             fig2d.colorbar(contour, ax=ax2d, label="Loss")
# #             ax2d.set_title(f"{loss_type} Contour View + Gradient Field")
# #             ax2d.legend()
# #             st.pyplot(fig2d)

# #         st.markdown("#### ✅ Constraint Checker")

# #         start_x = st.session_state.get("start_x", 0.0)
# #         start_y = st.session_state.get("start_y", 0.0)


# #         # ⛳ Ensure path is updated
# #         path, losses, meta = optimize_path(
# #             start_x,
# #             start_y,
# #             optimizer=selected_opts[0],  # or st.session_state.get("optimizer", "GradientDescent")
# #             lr=lr,
# #             steps=steps,
# #             f_func=f_func,
# #             grad_f=grad_f,
# #             hessian_f=hessian_f,
# #             options=options
# #         )
# #         if constraints:
# #             fig_con, ax_con = plt.subplots(figsize=(4, 3))
# #             for i, g_func in enumerate(g_funcs):
# #                 violations = [g_func(xp, yp) for xp, yp in path]
# #                 ax_con.plot(violations, label=f"g{i+1}(x, y)")
# #             ax_con.axhline(0, color="red", linestyle="--")
# #             ax_con.set_xlabel("Step")
# #             ax_con.set_ylabel("g(x, y)")
# #             ax_con.legend()
# #             st.pyplot(fig_con)
# #         else:
# #             st.info("No constraints defined.")


# with tab3:
#     st.header("📐 Symbolic Analysis: KKT, Gradient & Hessian")


#     st.markdown("#### 🎯 Objective & Lagrangian")
#     st.latex(r"f(x, y) = " + sp.latex(f_expr))
#     st.latex(r"\mathcal{L}(x, y, \lambda) = " + sp.latex(L_expr))

#     st.markdown("#### ✅ KKT Conditions")
#     for i, cond in enumerate(kkt_conditions):
#         st.latex(fr"\text{{KKT}}_{{{i+1}}} = {sp.latex(cond)}")
#     st.markdown("#### 🧮 Gradient & Hessian")
#     grad = [sp.diff(f_expr, v) for v in (x_sym, y_sym)]
#     # grad = [sp.diff(f_expr, v) for v in (x, y)]
#     # hessian = sp.hessian(f_expr, (x, y))
#     hessian = sp.hessian(f_expr, (x_sym, y_sym))

#     st.latex("Gradient: " + sp.latex(sp.Matrix(grad)))
#     st.latex("Hessian: " + sp.latex(hessian))

#     if optimizer == "Newton's Method":
#         st.markdown("#### 🧠 Newton Method Diagnostics")        
#         hess = sp.hessian(f_expr, (x_sym, y_sym))
#         hess_func = sp.lambdify((x_sym, y_sym), hess, modules="numpy")
#         H_val = np.array(hess_func(start_x, start_y))
#         det_val = np.linalg.det(H_val)

#         st.markdown("#### 📐 Hessian Matrix")
#         st.latex(r"\text{H}(x, y) = " + sp.latex(hess))

#         st.markdown("#### 📏 Determinant")
#         st.latex(r"\det(\text{H}) = " + sp.latex(sp.det(hess)))

#         if np.isclose(det_val, 0, atol=1e-6):
#             st.error("❌ Determinant is zero — Newton's Method cannot proceed (singular Hessian).")
#         elif det_val < 0:
#             st.warning("⚠️ Negative determinant — may indicate a saddle point or non-convex region.")
#         else:
#             st.success("✅ Hessian is suitable for Newton's Method descent.")





# === Full Modular Streamlit App ===

import streamlit as st
import plotly.graph_objects as go
import streamlit.components.v1 as components
import base64
import tempfile
import time
import sympy as sp

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from matplotlib import cm

from sympy import symbols, lambdify
from io import BytesIO
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from collections import OrderedDict
from visualizer import plot_3d_descent, plot_2d_contour

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


# === Auto-Tuning Simulation Function ===
def run_auto_tuning_simulation(f_func, optimizer, x0, y0):
    lr_grid = list(np.logspace(-4, -1, 6))
    step_grid = [20, 30, 40, 50, 60, 80]
    best_score = float("inf")
    best_lr, best_steps = lr_grid[0], step_grid[0]
    logs = []

    for lr in lr_grid:
        for steps in step_grid:
            x_t, y_t = x0, y0
            m, v = np.zeros(2), np.zeros(2)
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            for t in range(1, steps + 1):
                dx = (f_func(x_t + 1e-5, y_t) - f_func(x_t - 1e-5, y_t)) / 2e-5
                dy = (f_func(x_t, y_t + 1e-5) - f_func(x_t, y_t - 1e-5)) / 2e-5
                grad = np.array([dx, dy])
                if np.linalg.norm(grad) < 1e-3:
                    break
                if optimizer == "Adam":
                    m = beta1 * m + (1 - beta1) * grad
                    v = beta2 * v + (1 - beta2) * (grad ** 2)
                    m_hat = m / (1 - beta1 ** t)
                    v_hat = v / (1 - beta2 ** t)
                    update = lr * m_hat / (np.sqrt(v_hat) + eps)
                elif optimizer == "RMSProp":
                    v = beta2 * v + (1 - beta2) * (grad ** 2)
                    update = lr * grad / (np.sqrt(v) + eps)
                else:
                    update = lr * grad
                x_t -= update[0]
                y_t -= update[1]
            loss = f_func(x_t, y_t)
            score = loss + 0.01 * steps
            logs.append({"lr": lr, "steps": steps, "loss": loss, "score": score})
            if score < best_score:
                best_score, best_lr, best_steps = score, lr, steps
    st.session_state.df_log = pd.DataFrame(logs)
    return best_lr, best_steps


def optimize_univariate(x0, optimizer, lr, steps, f_func, grad_f, options=None):
    path = [x0]
    m = 0.0
    for t in range(1, steps + 1):
        x_t = path[-1]
        grad = grad_f(x_t)
        if optimizer == "Momentum":
            momentum = options.get("momentum", 0.9)
            m = momentum * m + lr * grad
            update = m
        else:
            update = lr * grad
        x_new = x_t - update
        path.append(x_new)
    return path


def optimize_path(x0, y0, optimizer, lr, steps, f_func, grad_f=None, hessian_f=None, options=None):
    path = [(x0, y0)]
    options = options or {}
    meta = {} 

    # --- Special Cases: Return early ---
    if optimizer == "GradientDescent" and options.get("use_backtracking", False):
        grad_f_expr = [sp.diff(f_expr, v) for v in (x_sym, y_sym)]
        path, alphas = backtracking_line_search_sym(f_expr, grad_f_expr, x0, y0)
        meta["callback_steps"] = len(path)
        return path, alphas, meta


    if optimizer == "Newton's Method" and options.get("newton_variant") in ["BFGS", "L-BFGS"]:
        from scipy.optimize import minimize

        path_coords = []
        losses = []

        def loss_vec(v):
            return f_func(v[0], v[1])

        def grad_vec(v):
            return np.atleast_1d(grad_f(v[0], v[1]))

        def callback(vk):
            path_coords.append((vk[0], vk[1]))
            losses.append(f_func(vk[0], vk[1]))

        x0_vec = np.array([x0, y0])
        method = "L-BFGS-B" if options["newton_variant"] == "L-BFGS" else "BFGS"

        res = minimize(loss_vec, x0_vec, method=method, jac=grad_vec, callback=callback, options={"maxiter": steps})

        # Fallback if callback was never triggered (e.g., early convergence)
        if not path_coords:
            path_coords = [tuple(res.x)]
            losses = [f_func(*res.x)]

        meta["res_nit"] = res.nit
        meta["callback_steps"] = len(path_coords)
        return path_coords, losses, meta


    if optimizer == "Simulated Annealing":
        T, cooling = options.get("T", 2.0), options.get("cooling", 0.95)
        current = f_func(x0, y0)
        for _ in range(steps):
            xn, yn = x0 + np.random.randn(), y0 + np.random.randn()
            candidate = f_func(xn, yn)
            if candidate < current or np.random.rand() < np.exp(-(candidate - current)/T):
                x0, y0 = xn, yn
                current = candidate
                path.append((x0, y0))
            T *= cooling
        meta["callback_steps"] = len(path)
        return path, None, meta
    
    if optimizer == "Genetic Algorithm":
        pop_size = options.get("pop_size", 20)
        mutation_std = options.get("mutation_std", 0.3)
        pop = [np.random.uniform(-5, 5, 2) for _ in range(pop_size)]
        for _ in range(steps // 2):
            pop = sorted(pop, key=lambda p: f_func(p[0], p[1]))[:pop_size // 2]
            children = [np.mean([pop[i], pop[j]], axis=0) + np.random.normal(0, mutation_std, 2)
                        for i in range(len(pop)) for j in range(i+1, len(pop))][:pop_size // 2]
            pop += children
        best = sorted(pop, key=lambda p: f_func(p[0], p[1]))[0]
        path = [tuple(best)]
        meta["callback_steps"] = len(path)
        return path, None, meta

    # --- Standard Optimizer Loop ---
    m, v = np.zeros(2), np.zeros(2)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for t in range(1, steps + 1):
        x_t, y_t = path[-1]
        grad = grad_f(x_t, y_t)

        if optimizer == "Adam":
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            update = lr * m_hat / (np.sqrt(v_hat) + eps)

        elif optimizer == "RMSProp":
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            update = lr * grad / (np.sqrt(v) + eps)

        elif optimizer == "Newton's Method":
            variant = options.get("newton_variant", "Classic Newton")
            if variant == "Classic Newton":
                try:
                    H = hessian_f(x_t, y_t)
                    H_inv = np.linalg.inv(H)
                    update = H_inv @ grad
                except:
                    update = grad
            elif variant == "Numerical Newton":
                eps_num = 1e-4
                def fx(x_, y_): return f_func(x_, y_)
                def second_partial(f, x, y, i, j):
                    h = eps_num
                    if i == 0 and j == 0:
                        return (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / h**2
                    elif i == 1 and j == 1:
                        return (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / h**2
                    else:
                        return (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h**2)
                H = np.array([
                    [second_partial(fx, x_t, y_t, 0, 0), second_partial(fx, x_t, y_t, 0, 1)],
                    [second_partial(fx, x_t, y_t, 1, 0), second_partial(fx, x_t, y_t, 1, 1)]
                ])
                try:
                    update = np.linalg.inv(H) @ grad
                except:
                    update = grad

        elif optimizer == "Momentum":
            momentum = options.get("momentum", 0.9)
            m = momentum * m + lr * grad
            update = m
        else:  # Default: GradientDescent
            update = lr * grad


        update = np.asarray(update)
        new_x, new_y = x_t - update[0], y_t - update[1]
        path.append((new_x, new_y))

    meta["callback_steps"] = len(path)
    return path, None, meta

def backtracking_line_search_sym(f_sym, grad_f_sym, x0, y0, alpha0=1.0, beta=0.5, c=1e-4, max_iters=100):
    # Declare standard symbols for lambdify
    x, y = sp.symbols("x y")

    # Ensure all expressions use x, y (not x_sym, y_sym)
    f_sym = f_sym.subs({sp.Symbol("x"): x, sp.Symbol("y"): y})
    grad_f_sym = [g.subs({sp.Symbol("x"): x, sp.Symbol("y"): y}) for g in grad_f_sym]

    f_lambd = sp.lambdify((x, y), f_sym, modules='numpy')
    grad_f_lambd = [sp.lambdify((x, y), g, modules='numpy') for g in grad_f_sym]

    xk, yk = x0, y0
    path = [(xk, yk)]
    alphas = []

    for _ in range(max_iters):
        gx, gy = grad_f_lambd[0](xk, yk), grad_f_lambd[1](xk, yk)
        grad_norm = gx**2 + gy**2

        if grad_norm < 1e-10:
            break

        alpha = alpha0
        while True:
            x_new = xk - alpha * gx
            y_new = yk - alpha * gy
            lhs = f_lambd(x_new, y_new)
            rhs = f_lambd(xk, yk) - c * alpha * grad_norm
            if lhs <= rhs or alpha < 1e-8:
                break
            alpha *= beta

        xk, yk = xk - alpha * gx, yk - alpha * gy
        path.append((xk, yk))
        alphas.append(alpha)

    return path, alphas

    # === Main Area: Title & Playground ===
st.title("🚀 Optimizer Visual Playground")



tab1, tab2, tab3 = st.tabs(["📘 Guide", "🧪 Optimizer Playground", "📐 Symbolic Analysis"])

with tab1:
    st.title("📘 Optimization Guide")

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




with tab2:

    x, y, w = sp.symbols("x y w")
    x_sym, y_sym, w_sym = sp.symbols("x y w")

    predefined_funcs = {
        "Quadratic Bowl": (x**2 + y**2, [], "Convex bowl with global minimum at origin."),
        "Saddle": (x**2 - y**2, [], "Saddle point at origin, non-convex."),
        "Rosenbrock": ((1 - x)**2 + 100 * (y - x**2)**2, [], "Banana-shaped curved valley; classic non-convex test function."),
        "Constrained Circle": (x * y, [x + y - 1], "Constrained optimization with linear constraint x + y = 1."),
        "Double Constraint": (x**2 + y**2, [x + y - 1, x**2 + y**2 - 4], "Intersection of circle and line constraints."),
        "Multi-Objective": (w * ((x - 1)**2 + (y - 2)**2) + (1 - w) * ((x + 2)**2 + (y + 1)**2), [], "Weighted sum of two quadratic objectives."),
        "Ackley": (-20*sp.exp(-0.2*sp.sqrt(0.5*(x**2 + y**2))) - sp.exp(0.5*(sp.cos(2*sp.pi*x) + sp.cos(2*sp.pi*y))) + sp.E + 20, [], "Highly multimodal, non-convex function."),
        "Rastrigin": (20 + x**2 - 10*sp.cos(2*sp.pi*x) + y**2 - 10*sp.cos(2*sp.pi*y), [], "Non-convex with many local minima (periodic)."),
        "Styblinski-Tang": (0.5*((x**4 - 16*x**2 + 5*x) + (y**4 - 16*y**2 + 5*y)), [], "Non-convex with many local minima."),
        "Sphere": (x**2 + y**2, [], "Simple convex function; global minimum at origin."),
        "Himmelblau": ((x**2 + y - 11)**2 + (x + y**2 - 7)**2, [], "Non-convex with four global minima."),
        "Booth": ((x + 2*y - 7)**2 + (2*x + y - 5)**2, [], "Simple convex quadratic function."),
        "Beale": ((1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2, [], "Non-convex with several valleys.")

    }


    st.title("🧪 Optimizer Playground")
    col_left, col_right = st.columns([1, 1])

    with col_left:
        mode_dim = st.radio("Function Type", ["Bivariate (f(x,y))", "Univariate (f(x))"])
        mode = st.radio("Function Source", ["Predefined", "Custom"])
        func_name = st.selectbox("Function", list(predefined_funcs.keys())) if mode == "Predefined" else None
        expr_str = st.text_input("Enter function f(x,y):", "x**2 + y**2") if mode == "Custom" else ""
        w_val = st.slider("Weight w (Multi-Objective)", 0.0, 1.0, 0.5) if func_name == "Multi-Objective" else None

        optimizers = ["GradientDescent",  "Momentum", "Adam", "RMSProp", "Newton's Method", "Simulated Annealing", "Genetic Algorithm"]
        optimizer = st.selectbox("Optimizer", optimizers)
        
        options = {}

        if optimizer == "Newton's Method":
            newton_variant = st.selectbox("Newton Variant", ["Classic Newton", "Numerical Newton", "BFGS", "L-BFGS"])
            options["newton_variant"] = newton_variant
        elif optimizer == "Simulated Annealing":
            options["T"] = st.slider("Initial Temperature (T)", 0.1, 10.0, 2.0)
            options["cooling"] = st.slider("Cooling Rate", 0.80, 0.99, 0.95)
        elif optimizer == "Genetic Algorithm":
            options["pop_size"] = st.slider("Population Size", 10, 100, 20)
            options["mutation_std"] = st.slider("Mutation Std Dev", 0.1, 1.0, 0.3)
        elif optimizer == "Momentum":
            options["momentum"] = st.slider("Momentum Coefficient", 0.0, 0.99, 0.9)
            

        auto_tune = False

        if optimizer in ["GradientDescent", "Adam", "RMSProp"]:
            if optimizer == "GradientDescent":
                use_backtracking = st.checkbox("🔍 Use Backtracking Line Search", value=False)
                options["use_backtracking"] = use_backtracking

                if use_backtracking:
                    st.checkbox("⚙️ Auto-Tune Learning Rate & Steps", value=False, disabled=True, key="auto_tune_disabled")
                    auto_tune = False
                    if "auto_tune_checkbox" in st.session_state:
                        st.session_state["auto_tune_checkbox"] = False
                    st.caption("ℹ️ Disabled because backtracking search dynamically adjusts step size.")
                else:
                    auto_tune = st.checkbox("⚙️ Auto-Tune Learning Rate & Steps", value=True, key="auto_tune_checkbox")
            else:
                auto_tune = st.checkbox("⚙️ Auto-Tune Learning Rate & Steps", value=True, key="auto_tune_checkbox")

        elif optimizer == "Newton's Method":
            st.checkbox("⚙️ Auto-Tune Learning Rate & Steps", value=False, disabled=True, key="auto_tune_disabled")
            auto_tune = False
            if "auto_tune_checkbox" in st.session_state:
                st.session_state["auto_tune_checkbox"] = False
            st.caption("ℹ️ Auto-tune is not applicable to Newton’s Method.")
            
        start_xy_defaults = {
            "Quadratic Bowl": (-3.0, 3.0), "Saddle": (-2.0, 2.0), "Rosenbrock": (-1.5, 1.5),
            "Constrained Circle": (0.5, 0.5), "Double Constraint": (-1.5, 1.5),
            "Multi-Objective": (0.0, 0.0), "Ackley": (2.0, -2.0), "Rastrigin": (3.0, 3.0),
            "Styblinski-Tang": (-2.5, -2.5), "Sphere": (-3.0, 3.0), "Himmelblau": (0.0, 0.0),
            "Booth": (1.0, 1.0), "Beale": (-2.0, 2.0)
        }
        default_x, default_y = start_xy_defaults.get(func_name, (-3.0, 3.0))
        default_lr = 0.005
        default_steps = 50


        # Make sure this is properly initialized before calling the function
        if auto_tune:
            # Ensure function is properly initialized
            x_sym, y_sym, w_sym = sp.symbols("x y w")
            symbolic_expr = predefined_funcs[func_name][0]

            if func_name == "Multi-Objective" and w_val is not None:
                symbolic_expr = symbolic_expr.subs(w_sym, w_val)

            # Lambdify the symbolic function to make it usable
            f_lambdified = sp.lambdify((x_sym, y_sym), symbolic_expr, modules="numpy")

            # Call the auto-tuning simulation function
            best_lr, best_steps = run_auto_tuning_simulation(f_lambdified, optimizer, default_x, default_y)
            # st.success(f"✅ Auto-tuned: lr={best_lr}, steps={best_steps}, start=({default_x},{default_y})")
            default_lr, default_steps = best_lr, best_steps

        if auto_tune:
            # Run the auto-tuning simulation to find the best lr and steps
            best_lr, best_steps = run_auto_tuning_simulation(f_lambdified, optimizer, default_x, default_y)

            # Update session state with the tuned values
            st.session_state.lr = best_lr
            st.session_state.steps = best_steps
            st.session_state.start_x = default_x
            st.session_state.start_y = default_y
            st.session_state.params_set = True  # Flag to indicate parameters are set

            # Provide feedback to the user
            st.success(f"✅ Auto-tuned: lr={best_lr}, steps={best_steps}, start=({default_x},{default_y})")

        # Ensure session state values exist to avoid AttributeError
        if "params_set" not in st.session_state or st.button("🔄 Reset to Auto-Tuned"):
            # Reset session state to the auto-tuned values when "Reset to Auto-Tuned" is clicked
            st.session_state.lr = default_lr
            st.session_state.steps = default_steps
            st.session_state.start_x = default_x
            st.session_state.start_y = default_y
            st.session_state.params_set = True

        # Final user inputs for learning rate and steps
        if not (
            optimizer == "GradientDescent" and options.get("use_backtracking", False)
        ) and optimizer != "Newton's Method":
            lr = st.selectbox("Learning Rate", sorted(set([0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, default_lr])), index=0, key="lr")
            steps = st.slider("Steps", 10, 100, value=st.session_state.get("steps", 50), key="steps")
        elif optimizer == "Newton's Method":
            lr = None
            steps = None
            st.info("📌 Newton’s Method computes its own step size using the Hessian inverse — learning rate is not needed.")
        elif optimizer == "GradientDescent" and options.get("use_backtracking", False):
            lr = None
            steps = None
            st.info("📌 Using Backtracking Line Search — no need to set learning rate or step count.")

        # Ensure session state values exist for initial positions
        if "start_x" not in st.session_state:
            st.session_state["start_x"] = -3.0
        if "start_y" not in st.session_state:
            st.session_state["start_y"] = 3.0

        # Sliders for the initial starting positions
        st.slider("Initial x", -5.0, 5.0, st.session_state.start_x, key="start_x")
        st.slider("Initial y", -5.0, 5.0, st.session_state.start_y, key="start_y")

        # Animation toggle
        show_animation = st.checkbox("🎮 Animate Descent Steps", key="show_animation")

    if mode_dim == "Univariate (f(x))":
        expr_str = st.text_input("Enter univariate function f(x):", "x**2")
        try:
            f_expr = sp.sympify(expr_str)
            f_func = sp.lambdify(x_sym, f_expr, modules="numpy")
            grad_f = lambda x: (f_func(x + 1e-5) - f_func(x - 1e-5)) / 2e-5
        except:
            st.error("Invalid expression.")
            st.stop()
    else:
        if mode == "Predefined":
            f_expr, constraints, description = predefined_funcs[func_name]
            if func_name == "Multi-Objective" and w_val is not None:
                f_expr = f_expr.subs(w, w_val)
        else:
            try:
                f_expr = sp.sympify(expr_str)
                constraints = []
                description = "Custom function."
            except:
                st.error("Invalid expression.")
                st.stop()

        f_func = sp.lambdify((x_sym, y_sym), f_expr, modules=["numpy"])
        grad_f = lambda x0, y0: np.array([
            (f_func(x0 + 1e-5, y0) - f_func(x0 - 1e-5, y0)) / 2e-5,
            (f_func(x0, y0 + 1e-5) - f_func(x0, y0 - 1e-5)) / 2e-5
        ])



    def hessian_f(x0, y0):
        hess_expr = sp.hessian(f_expr, (x, y))
        hess_func = sp.lambdify((x, y), hess_expr, modules=["numpy"])
        return np.array(hess_func(x0, y0))


    # st.markdown(f"### 📘 Function Description:\n> {description}")

    # Setup for symbolic Lagrangian and KKT (only for bivariate)
    if mode_dim == "Univariate (f(x))":
        constraints = []
        L_expr = None
        grad_L = []
        kkt_conditions = []
    else:
        L_expr = f_expr + sum(sp.Symbol(f"lambda{i+1}") * g for i, g in enumerate(constraints))
        grad_L = [sp.diff(L_expr, v) for v in (x_sym, y_sym)]
        kkt_conditions = grad_L + constraints



    def simulate_optimizer(opt_name, f_expr, lr=0.01, steps=50):
        f_func = sp.lambdify((x_sym, y_sym), f_expr, modules=["numpy"])
        # f_func = sp.lambdify((x, y), f_expr, modules="numpy")
        x0, y0 = -3, 3
        path = [(x0, y0)]
        m, v = 0, 0
        for t in range(1, steps + 1):
            x_t, y_t = path[-1]
            dx = (f_func(x_t + 1e-5, y_t) - f_func(x_t - 1e-5, y_t)) / 2e-5
            dy = (f_func(x_t, y_t + 1e-5) - f_func(x_t, y_t - 1e-5)) / 2e-5
            g = np.array([dx, dy])
            if opt_name == "Adam":
                m = 0.9 * m + 0.1 * g
                v = 0.999 * v + 0.001 * (g ** 2)
                m_hat = m / (1 - 0.9 ** t)
                v_hat = v / (1 - 0.999 ** t)
                update = lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            elif opt_name == "RMSProp":
                v = 0.999 * v + 0.001 * (g ** 2)
                update = lr * g / (np.sqrt(v) + 1e-8)
            elif opt_name == "Newton's Method":
                hess = sp.hessian(f_expr, (x, y))
                hess_func = sp.lambdify((x, y), hess, modules="numpy")
                try:
                    H = np.array(hess_func(x_t, y_t))
                    H_inv = np.linalg.inv(H)
                    update = H_inv @ g
                except:
                    update = g
            else:
                update = lr * g
            x_new, y_new = x_t - update[0], y_t - update[1]
            path.append((x_new, y_new))
        final_x, final_y = path[-1]
        grad_norm = np.linalg.norm(g)
        return {
            "Optimizer": opt_name,
            "Final Value": round(f_func(final_x, final_y), 4),
            "Gradient Norm": round(grad_norm, 4),
            "Steps": len(path) - 1
        }
    g_funcs = [sp.lambdify((x_sym, y_sym), g, modules=["numpy"]) for g in constraints]  
    # g_funcs = [sp.lambdify((x, y), g, modules=["numpy"]) for g in constraints]
    f_func = sp.lambdify((x_sym, y_sym), f_expr, modules=["numpy"])
    # f_func = sp.lambdify((x, y), f_expr, modules=["numpy"])
    grad_f = lambda x0, y0: np.array([
        (f_func(x0 + 1e-5, y0) - f_func(x0 - 1e-5, y0)) / 2e-5,
        (f_func(x0, y0 + 1e-5) - f_func(x0, y0 - 1e-5)) / 2e-5
    ])

    def hessian_f(x0, y0):
        hess_expr = sp.hessian(f_expr, (x, y))
        hess_func = sp.lambdify((x, y), hess_expr, modules=["numpy"])
        return np.array(hess_func(x0, y0))

    # === Pull final values from session_state
    start_x = st.session_state.get("start_x", -3.0)
    start_y = st.session_state.get("start_y", 3.0)
    lr = st.session_state.get("lr", 0.01)
    steps = st.session_state.get("steps", 50)



    if mode_dim == "Univariate (f(x))":
        path = optimize_univariate(
            start_x,
            optimizer,
            lr,
            steps,
            f_func,
            grad_f,
            options
        )
        xs = path
        ys = [f_func(x) for x in xs]
    else:
        path, alpha_log, meta = optimize_path(
            start_x, start_y,
            optimizer=optimizer,
            lr=lr,
            steps=steps,
            f_func=f_func,
            grad_f=grad_f,
            hessian_f=hessian_f,
            options=options
        )
        xs, ys = zip(*path)
        Z_path = [f_func(xp, yp) for xp, yp in path]



    x_vals = np.linspace(-5, 5, 200)
    y_vals = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = f_func(X, Y)

    # --- Taylor Expansion Toggle ---

    show_taylor = st.checkbox("📐 Show Taylor Approximation at (a, b)", value=False)

    show_2nd = False
    Z_t1 = None
    Z_t2 = None

    a_val, b_val = None, None  # set early to avoid NameError
    expansion_point = None

    # expansion_point = (a_val, b_val) if show_taylor else None

    if show_taylor:
        st.markdown("**Taylor Expansion Center (a, b)**")

        # --- Initialize session state on first run ---
        if "a_val" not in st.session_state or "b_val" not in st.session_state:
            if float(start_x) == 0.0 and float(start_y) == 0.0 and func_name != "Quadratic Bowl":
                st.session_state.a_val = 0.1
                st.session_state.b_val = 0.1
                st.info("🔁 Auto-shifted expansion point from (0,0) → (0.1, 0.1)")
            else:
                st.session_state.a_val = float(start_x)
                st.session_state.b_val = float(start_y)

        # --- Sliders using session state ---
        a_val = st.slider("a (expansion x)", -5.0, 5.0, value=st.session_state.a_val, step=0.1, key="a_val")
        b_val = st.slider("b (expansion y)", -5.0, 5.0, value=st.session_state.b_val, step=0.1, key="b_val")

        # --- Prevent user from going back to (0,0) if it's not allowed ---
        if a_val == 0.0 and b_val == 0.0 and func_name != "Quadratic Bowl":
            st.warning("🚫 (0,0) is a singular point. Auto-shifting again to (0.1, 0.1)")
            a_val = 0.1
            b_val = 0.1
            st.session_state.a_val = 0.1
            st.session_state.b_val = 0.1

        expansion_point = (a_val, b_val)
        show_2nd = st.checkbox("Include 2nd-order terms", value=True)


        # --- Symbolic derivatives ---
        grad_fx = [sp.diff(f_expr, var) for var in (x_sym, y_sym)]
        hess_fx = sp.hessian(f_expr, (x_sym, y_sym))

        subs = {x_sym: a_val, y_sym: b_val}
        st.text(f"📌 Taylor center used: (a={a_val:.3f}, b={b_val:.3f})")

        f_ab = float(f_expr.subs(subs))
        grad_vals = [float(g.subs(subs)) for g in grad_fx]
        hess_vals = hess_fx.subs(subs)
        Hxx = float(hess_vals[0, 0])
        Hxy = float(hess_vals[0, 1])
        Hyy = float(hess_vals[1, 1])

        # dx, dy remain symbolic
        dx, dy = x_sym - a_val, y_sym - b_val

        # Fully numeric constants in expression
        T1_expr = f_ab + grad_vals[0]*dx + grad_vals[1]*dy
        T2_expr = T1_expr + 0.5 * (Hxx*dx**2 + 2*Hxy*dx*dy + Hyy*dy**2)

        # Numerical evaluation for plotting
        t1_np = sp.lambdify((x_sym, y_sym), T1_expr, "numpy")
        t2_np = sp.lambdify((x_sym, y_sym), T2_expr, "numpy") if show_2nd else None
        Z_t1 = t1_np(X, Y)


        if show_2nd:
            try:
                Z_t2 = t2_np(X, Y)

                # --- Safety checks ---
                if isinstance(Z_t2, (int, float, np.number)):
                    st.warning(f"❌ Z_t2 is scalar: {Z_t2}. Skipping.")
                    Z_t2 = None
                else:
                    Z_t2 = np.array(Z_t2, dtype=np.float64)

                    if Z_t2.ndim != 2:
                        st.warning(f"❌ Z_t2 is not 2D — shape: {Z_t2.shape}")
                        Z_t2 = None
                    elif Z_t2.shape != (len(y_vals), len(x_vals)):
                        if Z_t2.shape == (len(x_vals), len(y_vals)):
                            Z_t2 = Z_t2.T
                        else:
                            st.warning(f"❌ Z_t2 shape mismatch: {Z_t2.shape}")
                            Z_t2 = None
                    elif np.isnan(Z_t2).any():
                        st.warning("❌ Z_t2 contains NaNs.")
                        Z_t2 = None

            except Exception as e:
                st.warning(f"⚠️ Failed to evaluate 2nd-order Taylor surface: {e}")
                Z_t2 = None

        # Z_t2 = t2_np(X, Y) if show_2nd else None

        # ✅ Display symbolic formula as LaTeX
        st.markdown("### ✏️ Taylor Approximation Formula at \\( (a, b) = ({:.1f}, {:.1f}) \\)".format(a_val, b_val))

        fx, fy = grad_vals
        Hxx, Hxy, Hyy = hess_vals[0, 0], hess_vals[0, 1], hess_vals[1, 1]

        T1_latex = f"f(x, y) \\approx {f_ab:.3f} + ({fx}) (x - {a_val}) + ({fy}) (y - {b_val})"
        T2_latex = (
            f"{T1_latex} + \\frac{{1}}{{2}}({Hxx}) (x - {a_val})^2 + "
            f"{Hxy} (x - {a_val})(y - {b_val}) + "
            f"\\frac{{1}}{{2}}({Hyy}) (y - {b_val})^2"
        )

        st.latex(T1_latex)
        if show_2nd:
            st.latex(T2_latex)


    st.markdown("### 📈 3D View")

    if show_2nd:
        try:
            if Z_t2 is not None:
                # ❗ Reject raw scalar values
                if isinstance(Z_t2, (int, float)):
                    st.warning(f"❌ Z_t2 is a scalar ({Z_t2}), not an array.")
                    Z_t2 = None

                Z_t2 = np.array(Z_t2, dtype=np.float64)

                if Z_t2.ndim != 2:
                    st.warning(f"❌ Z_t2 is not 2D — shape: {Z_t2.shape}")
                    Z_t2 = None
                elif np.isnan(Z_t2).any():
                    st.warning("❌ Z_t2 contains NaNs.")
                    Z_t2 = None
                elif Z_t2.shape != (len(y_vals), len(x_vals)):
                    if Z_t2.shape == (len(x_vals), len(y_vals)):
                        Z_t2 = Z_t2.T
                    else:
                        st.warning(f"❌ Z_t2 shape mismatch: {Z_t2.shape} vs mesh ({len(y_vals)}, {len(x_vals)})")
                        Z_t2 = None
        except Exception as e:
            st.warning(f"❌ Error processing Z_t2: {e}")
            Z_t2 = None

            
    plot_3d_descent(
        x_vals=x_vals,
        y_vals=y_vals,
        Z=Z,
        path=path,
        Z_path=Z_path,
        Z_t1=Z_t1,
        Z_t2=Z_t2,
        show_taylor=show_taylor,
        show_2nd=show_2nd,
        expansion_point=expansion_point,
        f_func=f_func
    )


    st.markdown("### 🗺️ 2D View")
    plot_2d_contour(
        x_vals=x_vals,
        y_vals=y_vals,
        Z=Z,
        path=path,
        g_funcs=g_funcs if constraints else None,
        X=X, Y=Y,
        Z_t2=Z_t2,
        show_2nd=show_2nd,
        expansion_point=expansion_point
    )



    if show_taylor:
        st.caption("🔺 Red = 1st-order Taylor, 🔷 Blue = 2nd-order Taylor, 🟢 Green = true surface")

    if show_animation:
        frames = []
        fig_anim, ax_anim = plt.subplots(figsize=(5, 4))

        for i in range(1, len(path) + 1):
            ax_anim.clear()
            ax_anim.contour(X, Y, Z, levels=30, cmap="viridis")
            ax_anim.plot(*zip(*path[:i]), 'r*-')
            ax_anim.set_xlim([-5, 5])
            ax_anim.set_ylim([-5, 5])
            ax_anim.set_title(f"Step {i}/{len(path)-1}")

            buf = BytesIO()
            fig_anim.savefig(buf, format='png', dpi=100)  # optional: set dpi
            buf.seek(0)
            frames.append(Image.open(buf).convert("P"))  # convert to palette for GIF efficiency
            buf.close()

        gif_buf = BytesIO()
        frames[0].save(
            gif_buf, format="GIF", save_all=True,
            append_images=frames[1:], duration=300, loop=0
        )
        gif_buf.seek(0)
        st.image(gif_buf, caption="📽️ Animated Descent Path", use_container_width=True)



with tab3:
    st.header("📐 Symbolic Analysis: KKT, Gradient & Hessian")


    st.markdown("#### 🎯 Objective & Lagrangian")
    st.latex(r"f(x, y) = " + sp.latex(f_expr))
    st.latex(r"\mathcal{L}(x, y, \lambda) = " + sp.latex(L_expr))

    st.markdown("#### ✅ KKT Conditions")
    for i, cond in enumerate(kkt_conditions):
        st.latex(fr"\text{{KKT}}_{{{i+1}}} = {sp.latex(cond)}")
    st.markdown("#### 🧮 Gradient & Hessian")
    grad = [sp.diff(f_expr, v) for v in (x_sym, y_sym)]
    # grad = [sp.diff(f_expr, v) for v in (x, y)]
    # hessian = sp.hessian(f_expr, (x, y))
    hessian = sp.hessian(f_expr, (x_sym, y_sym))

    st.latex("Gradient: " + sp.latex(sp.Matrix(grad)))
    st.latex("Hessian: " + sp.latex(hessian))

    if optimizer == "Newton's Method":
        st.markdown("#### 🧠 Newton Method Diagnostics")        
        hess = sp.hessian(f_expr, (x_sym, y_sym))
        hess_func = sp.lambdify((x_sym, y_sym), hess, modules="numpy")
        H_val = np.array(hess_func(start_x, start_y))
        det_val = np.linalg.det(H_val)

        st.markdown("#### 📐 Hessian Matrix")
        st.latex(r"\text{H}(x, y) = " + sp.latex(hess))

        st.markdown("#### 📏 Determinant")
        st.latex(r"\det(\text{H}) = " + sp.latex(sp.det(hess)))

        if np.isclose(det_val, 0, atol=1e-6):
            st.error("❌ Determinant is zero — Newton's Method cannot proceed (singular Hessian).")
        elif det_val < 0:
            st.warning("⚠️ Negative determinant — may indicate a saddle point or non-convex region.")
        else:
            st.success("✅ Hessian is suitable for Newton's Method descent.")


