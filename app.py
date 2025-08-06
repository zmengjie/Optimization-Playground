import streamlit as st

# Set page config
st.set_page_config(page_title="Optimization Playground", layout="wide")

# Sidebar - simple Hello and sections
st.sidebar.markdown("## Hello")
st.sidebar.markdown("Welcome to the **Optimization Playground**!")

# Use buttons or radio buttons instead of links for cleaner section navigation
sections = ["Guide", "Taylor Series", "Optimizer Playground"]
section = st.sidebar.radio("Navigate through the sections:", sections)

# Sections below (for content display on the main page)
if section == "Guide":
    st.markdown("## Guide")
    st.markdown("""
    The **Optimization Playground** is a tool designed to help you understand optimization techniques and how they can be applied in real-world scenarios. In this guide, you'll learn the following:

    1. **What is Optimization?**
       - Optimization is the process of finding the best solution from a set of possible solutions. It is commonly used in machine learning, economics, and various engineering disciplines to minimize or maximize objective functions.

    2. **Types of Optimization Problems**
       - **Unconstrained Optimization**: Problems where no constraints limit the optimization process.
       - **Constrained Optimization**: Problems where certain constraints must be satisfied while optimizing the objective function.

    3. **How the App Works**
       - Navigate through different sections to learn about optimization concepts, experiment with Taylor series expansions, and apply optimizers to various functions interactively.

    Start experimenting with the sections below to dive deeper into the world of optimization!
    """)

elif section == "Taylor Series":
    st.markdown("## Taylor Series")
    st.markdown("""
    **Taylor series expansions** are a powerful mathematical tool used to approximate functions near a point. They play a key role in optimization techniques, especially in methods like Gradient Descent and Newton's Method.

    - **First-Order Taylor Expansion**:
      The first-order approximation of a function \( f(x) \) around a point \( x_0 \) is given by:
      
      \[
      f(x + \Delta x) \approx f(x) + \nabla f(x)^T \Delta x
      \]

      This approximation is used in optimization algorithms to compute step sizes in the direction of the gradient.

    - **Second-Order Taylor Expansion**:
      The second-order approximation includes the Hessian matrix (second derivatives) and provides more accurate results, especially when the function is non-linear:
      
      \[
      f(x + \Delta x) \approx f(x) + \nabla f(x)^T \Delta x + \frac{1}{2} \Delta x^T H(x) \Delta x
      \]

      This form is used in optimization methods like Newton’s Method to improve convergence speed.

    In this section, you will visualize how these expansions affect the optimization path and learn how optimizers use these approximations to find the optimal solution more efficiently.
    """)

elif section == "Optimizer Playground":
    st.markdown("## Optimizer Playground")
    st.markdown("""
    The **Optimizer Playground** allows you to interactively experiment with various optimization algorithms. Optimization is at the heart of many machine learning models, and this tool will help you understand how different algorithms work in practice.

    - **Gradient Descent**:
      The most widely used optimization technique, Gradient Descent updates the model parameters by moving in the direction opposite to the gradient of the cost function. You can adjust the learning rate and observe how it affects convergence.

    - **Newton's Method**:
      This method uses both the gradient and the Hessian (second-order derivatives) of the objective function to find the optimum. It converges faster than Gradient Descent, especially near the optimal solution.

    - **Simulated Annealing**:
      A probabilistic technique inspired by the annealing process in metallurgy. It explores the solution space by allowing occasional uphill moves to avoid getting stuck in local minima.

    - **Genetic Algorithms**:
      This optimization technique mimics natural selection by evolving a population of solutions over several generations, selecting the best solutions based on fitness criteria.

    Each optimizer has its strengths and weaknesses. Use this playground to visualize how these optimizers perform on different functions and explore their behavior. You can also experiment with custom functions to see how different parameters affect the optimization process.

    Start experimenting with your favorite optimizer below!
    """)
