import plotly.graph_objects as go
import streamlit as st
import numpy as np


def plot_3d_descent(x_vals, y_vals, Z, path, Z_path,
                    Z_t1=None, Z_t2=None,
                    show_taylor=False, show_2nd=False,
                    expansion_point=None, f_func=None):
    xs, ys = zip(*path)

    fig_3d = go.Figure()

    # Base surface
    fig_3d.add_trace(go.Surface(
        z=Z, x=x_vals, y=y_vals,
        colorscale='Greens', opacity=0.7,
        name="🟢 True Surface", showscale=False
    ))

    # Descent path
    fig_3d.add_trace(go.Scatter3d(
        x=xs, y=ys, z=Z_path,
        mode='lines+markers',
        line=dict(color='red', width=4),
        marker=dict(size=4),
        name='Descent Path',
        hoverinfo='x+y+z+text',
        text=['Step {}'.format(i) for i in range(len(path))]
    ))

    # Taylor surfaces with fading
    if show_taylor and Z_t1 is not None:
        fig_3d.add_trace(go.Surface(
            z=Z_t1, x=x_vals, y=y_vals,
            colorscale='Reds', opacity=0.5,
            # line=dict(width=2, color='black'),
            name="1st-Order Taylor"
        ))
    # if show_taylor and show_2nd and Z_t2 is not None and expansion_point is not None:
    #     a, b = expansion_point
    #     distance = np.sqrt((x_vals[:, None] - a)**2 + (y_vals[None, :] - b)**2)
    #     fade_opacity = 0.6 * np.exp(-0.1 * distance)
    #     fig_3d.add_trace(go.Surface(
    #         z=Z_t2, x=x_vals, y=y_vals,
    #         surfacecolor=fade_opacity,
    #         colorscale='Blues',
    #         opacity=0.4,
    #         name="2nd-Order Taylor"
    #     ))

# --- Taylor surfaces with fading ---

    if show_taylor and show_2nd:
        if Z_t2 is not None:
            try:
                if isinstance(Z_t2, (int, float)):
                    raise ValueError(f"Z_t2 is a scalar ({Z_t2}); expected 2D array.")
                Z_t2 = np.array(Z_t2)
                if Z_t2.ndim != 2:
                    raise ValueError(f"Z_t2 is not 2D (shape: {Z_t2.shape})")
                if np.isnan(Z_t2).any():
                    raise ValueError("Z_t2 contains NaNs.")
                if Z_t2.shape != (len(y_vals), len(x_vals)):
                    if Z_t2.shape == (len(x_vals), len(y_vals)):
                        Z_t2 = Z_t2.T
                    else:
                        raise ValueError(f"Z_t2 shape mismatch: {Z_t2.shape} vs expected {(len(y_vals), len(x_vals))}")
                fig_3d.add_trace(go.Surface(
                    z=Z_t2, x=x_vals, y=y_vals,
                    colorscale='Blues',
                    opacity=0.4,
                    # cmin=np.min(Z_t2),
                    # cmax=np.max(Z_t2),
                    # line=dict(width=2, color='black'),
                    name="2nd-Order Taylor"
                ))
            except Exception as e:
                st.warning(f"⚠️ Skipped 2nd-order Taylor surface: {e}")




    # Marker and dashed line from (a,b) to step 1
    if expansion_point is not None and f_func is not None:
        a, b = expansion_point
        z_ab = f_func(a, b)

        fig_3d.add_trace(go.Scatter3d(
            x=[a], y=[b], z=[z_ab],
            mode='markers+text',
            marker=dict(size=7, color='black', symbol='circle'),
            text=["(a, b)"],
            textposition="bottom right",
            textfont=dict(size=12),
            name="Expansion Point".
            hoverinfo='text+x+y+z'
        ))

        if len(path) > 1:
            x1, y1 = path[1]
            z1 = f_func(x1, y1)
            fig_3d.add_trace(go.Scatter3d(
                x=[a, x1], y=[b, y1], z=[z_ab, z1],
                mode='lines',
                line=dict(color='black', width=3, dash='dash'),
                name="Expansion → 1st Step"
            ))

    fig_3d.update_layout(
        title="3D Descent Path + Taylor Approximation",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="f(x, y)"),
        height=600,
        margin=dict(l=60, r=40, b=40, t=50),
        legend=dict(x=0.7, y=0.9),
        hovermode='closest',
        
        # 👉 Customize hoverlabel style
        hoverlabel=dict(
            bgcolor="black",   # background color
            font_size=12,
            font_color="black",  # text color
            bordercolor="white"  # border
        )
    )


    st.markdown("""
    ### 🧠 Teaching Tip
    The dashed vector from (a, b) shows how the Taylor approximation predicts the direction of descent.
    The 2nd-order surface illustrates curvature guidance for Newton's Method.
    """)
    st.plotly_chart(fig_3d, use_container_width=True)

def plot_2d_contour(x_vals, y_vals, Z, path,
                    g_funcs=None, X=None, Y=None,
                    Z_t2=None, show_2nd=False,
                    expansion_point=None):
    xs, ys = zip(*path)

    fig_2d = go.Figure()

    # Taylor contour first to draw under others
    if show_2nd and Z_t2 is not None:
        fig_2d.add_trace(go.Contour(
            z=Z_t2, x=x_vals, y=y_vals,
            showscale=False,
            colorscale='Blues',
            opacity=0.4,
            name="2nd-Order Taylor"
        ))

    # Base contour
    fig_2d.add_trace(go.Contour(
        z=Z, x=x_vals, y=y_vals,
        colorscale='Viridis',
        contours=dict(start=np.min(Z), end=np.max(Z), size=(np.max(Z) - np.min(Z)) / 20),
        name="Function Contour",
        showscale=True
    ))

    # Descent path
    fig_2d.add_trace(go.Scatter(
        x=xs, y=ys,
        mode='lines+markers',
        line=dict(color='red', width=3),
        marker=dict(size=4),
        name="Descent Path"
    ))

    # Constraints
    if g_funcs and X is not None and Y is not None:
        for g_f in g_funcs:
            G = g_f(X, Y)
            fig_2d.add_trace(go.Contour(
                z=G, x=x_vals, y=y_vals,
                contours=dict(showlines=False, coloring='lines'),
                line=dict(color='red', width=2),
                showscale=False,
                name="Constraint"
            ))

    if expansion_point is not None:
        a, b = expansion_point
        fig_2d.add_trace(go.Scatter(
            x=[a], y=[b],
            mode='markers+text',
            marker=dict(color='black', size=9, symbol="circle"),
            text=["(a, b)"],
            textposition="bottom center",
            textfont=dict(size=12),
            name='Expansion Point'
        ))

        if len(path) > 1:
            x1, y1 = path[1]
            fig_2d.add_trace(go.Scatter(
                x=[a, x1], y=[b, y1],
                mode='lines',
                line=dict(color='black', width=2, dash='dash'),
                name="Expansion → 1st Step"
            ))

    fig_2d.update_layout(
        title="2D Contour + Constraints + Taylor Overlay",
        xaxis_title="x", yaxis_title="y",
        height=500,
        margin=dict(l=60, r=40, b=40, t=50),
        legend=dict(x=1.05),
        dragmode='zoom',
        hovermode='closest',
        updatemenus=[dict(
            type="buttons", showactive=False,
            buttons=[dict(label="Reset Zoom", method="relayout",
                         args=[{"xaxis.autorange": True, "yaxis.autorange": True}])],
            x=0, y=0.92, xanchor='left', yanchor='top')]
    )

    st.markdown("""
    ### 🧠 Teaching Tip
    The dashed vector from (a, b) shows how the Taylor approximation predicts the direction of descent.
    The 2nd-order contour shows how curvature affects local optimization.
    """)
    st.plotly_chart(fig_2d, use_container_width=True)
