import streamlit as st
import pandas as pd

st.set_page_config(page_title="🩸 DIA", layout="wide")

st.markdown("<h1 style='text-align: center; color: #d7263d; font-size: 2.8rem;'>💊 Diabetes Insight App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555; font-size: 1.2rem;'>Explore and analyze the diabetes dataset interactively.</p>", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/color/96/diabetes.png", width=80)
    st.markdown("<h3 style='color: #4fc3f7;'>D I A - Where data meets diagnosis.</h3>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 1px solid #d7263d;'>", unsafe_allow_html=True)
    st.header("Filter Data")

df = pd.read_csv("diabetes.csv")

filter_cols = st.sidebar.multiselect("Select columns to filter:", df.columns)
filtered_df = df.copy()
for col in filter_cols:
    unique_vals = df[col].unique()
    selected_vals = st.sidebar.multiselect(f"Values for {col}", unique_vals, default=unique_vals, key=f"filter_{col}")
    filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]

tabs = st.tabs(["Dataset Preview", "Summary Stats", "Visualizations", "Correlation", "ML Models", "Download Data"])

with tabs[0]:
    st.markdown("<h3 style='color: #1a659e;'>Dataset Preview</h3>", unsafe_allow_html=True)
    st.dataframe(filtered_df.head(50), use_container_width=True)

with tabs[1]:
    st.markdown("<h3 style='color: #1a659e;'>Summary Statistics</h3>", unsafe_allow_html=True)
    st.dataframe(filtered_df.describe(), use_container_width=True)

with tabs[2]:
    st.markdown("<h3 style='color: #1a659e;'>Visualizations</h3>", unsafe_allow_html=True)
    num_cols = filtered_df.select_dtypes(include="number").columns.tolist()
    cat_cols = filtered_df.select_dtypes(exclude="number").columns.tolist()
    graph_types = ["Scatter", "Line", "Bar", "Box"]
    c1, c2, c3 = st.columns([2,2,2])
    with c1:
        graph_type = st.selectbox("Graph Type", graph_types)
    with c2:
        x_axis = st.selectbox("X-axis", num_cols + cat_cols)
    with c3:
        y_axis = st.selectbox("Y-axis", num_cols, index=1 if len(num_cols) > 1 else 0)
    plot_btn = st.button("Plot Graph", use_container_width=True)
    if plot_btn:
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(8, 5))
        if graph_type == "Scatter":
            sns.scatterplot(data=filtered_df, x=x_axis, y=y_axis, ax=ax)
        elif graph_type == "Line":
            sns.lineplot(data=filtered_df, x=x_axis, y=y_axis, ax=ax)
        elif graph_type == "Bar":
            sns.barplot(data=filtered_df, x=x_axis, y=y_axis, ax=ax)
        elif graph_type == "Box":
            sns.boxplot(data=filtered_df, x=x_axis, y=y_axis, ax=ax)
        st.pyplot(fig)

with tabs[3]:
    st.markdown("<h3 style='color: #1a659e;'>Correlation Heatmap</h3>", unsafe_allow_html=True)
    import seaborn as sns
    import matplotlib.pyplot as plt
    corr = filtered_df.select_dtypes(include="number").corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

with tabs[4]:
    st.markdown("<h3 style='color: #1a659e;'>Machine Learning Models</h3>", unsafe_allow_html=True)
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    import numpy as np
    c1, c2 = st.columns(2)
    with c1:
        target_col = st.selectbox("Select Target Column", filtered_df.columns)
    with c2:
        feature_cols = st.multiselect("Select Feature Columns", [col for col in filtered_df.columns if col != target_col], default=[col for col in filtered_df.columns if col != target_col])
    c3, c4 = st.columns([2,2])
    with c3:
        test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, 0.05)
    with c4:
        algo = st.selectbox("Algorithm", ["KNN", "Random Forest", "Logistic Regression", "Linear Regression"])
    run_ml = st.button("Run Model", use_container_width=True)
    @st.cache_data(show_spinner=False)
    def train_and_evaluate(X, y, algo, test_size):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        is_classification = y.nunique() <= 10 and y.dtype in [int, np.int64, np.int32]
        if algo == "KNN":
            model = KNeighborsClassifier() if is_classification else KNeighborsRegressor()
        elif algo == "Random Forest":
            model = RandomForestClassifier() if is_classification else RandomForestRegressor()
        elif algo == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif algo == "Linear Regression":
            model = LinearRegression()
        else:
            return None, None, None
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if is_classification or algo == "Logistic Regression":
            acc = accuracy_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            return acc, mse, rmse
        else:
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            return None, mse, rmse
    if run_ml and feature_cols:
        X = filtered_df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
        y = filtered_df[target_col].fillna(0)
        if X.shape[1] == 0:
            st.error("Please select at least one numeric feature column.")
        else:
            acc, mse, rmse = train_and_evaluate(X, y, algo, test_size)
            if acc is not None:
                st.success(f"Accuracy: {acc:.3f}")
            st.info(f"MSE: {mse:.3f}")
            st.info(f"RMSE: {rmse:.3f}")

with tabs[5]:
    st.markdown("<h3 style='color: #1a659e;'>Download Filtered Dataset</h3>", unsafe_allow_html=True)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=filtered_df.to_csv(index=False),
        file_name="filtered_diabetes.csv",
        mime="text/csv"
    )
