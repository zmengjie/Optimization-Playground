import streamlit as st

def main():
    st.set_page_config(page_title="Unsupervised Learning Explorer", layout="wide")
    st.title("🧠 Unsupervised Learning Explorer")

    # Sidebar: select module
    module = st.sidebar.radio("Select Module", [
        "Clustering",
        "Dimensionality Reduction",
        "Anomaly Detection",
        "Topic Modeling",
        "Association Rule Mining"
    ])

    if module == "Clustering":
        from clustering import clustering_ui
        clustering_ui()
    elif module == "Dimensionality Reduction":
        from dim_reduction import dim_reduction_ui
        dim_reduction_ui()
    elif module == "Anomaly Detection":
        from anomaly_detection import anomaly_detection_ui
        anomaly_detection_ui()
    elif module == "Topic Modeling":
        from topic_modeling import topic_modeling_ui
        topic_modeling_ui()
    elif module == "Association Rule Mining":
        from association_rules import association_rules_ui
        association_rules_ui()

if __name__ == "__main__":
    main()
