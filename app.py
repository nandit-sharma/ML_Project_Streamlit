import streamlit as st
import pandas as pd

# Load dataset efficiently
@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv("diabetes.csv")
df = load_data()


st.set_page_config(page_title="🩸 DIA", layout="wide")

st.title("💊 Diabetes Insight App")
st.write("Explore and analyze the diabetes dataset interactively.")

# Sidebar for navigation and filters
st.sidebar.image("https://img.icons8.com/color/96/diabetes.png", width=64)
st.sidebar.markdown("## Navigation")
st.sidebar.markdown("---")

nav_options = [
    "🏠 Dataset Preview",
    "📊 Summary Stats",
    "📈 Visualizations",
    "🔗 Correlation",
    "🤖 ML Models",
    "⬇️ Download Data"
]
section = st.sidebar.radio("", nav_options)

# Map nav_options to section logic
section_map = {
    "🏠 Dataset Preview": "Dataset Preview",
    "📊 Summary Stats": "Summary Stats",
    "📈 Visualizations": "Visualizations",
    "🔗 Correlation": "Correlation",
    "🤖 ML Models": "ML Models",
    "⬇️ Download Data": "Download Data"
}
section = section_map[section]

st.set_page_config(page_title="🧠 DIA", layout="wide")

st.title("🧠 Diabetes Insight App")
st.write("Explore and analyze the diabetes dataset interactively.")

# Sidebar for navigation and filters
st.sidebar.header("Navigation")
section = st.sidebar.radio("Go to:", ["Dataset Preview", "Summary Stats", "Visualizations", "Correlation", "ML Models", "Download Data"])


# Sidebar filters
st.sidebar.header("Filter Data")
filter_cols = st.sidebar.multiselect("Select columns to filter:", df.columns)
filtered_df = df.copy()
for col in filter_cols:
    unique_vals = df[col].unique()
    selected_vals = st.sidebar.multiselect(f"Values for {col}", unique_vals, default=unique_vals, key=f"filter_{col}")
    filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]

if section == "Dataset Preview":
    st.subheader("Dataset Preview")
    st.dataframe(filtered_df.head(50), use_container_width=True)

elif section == "Summary Stats":
    st.subheader("Summary Statistics")
    st.dataframe(filtered_df.describe(), use_container_width=True)

elif section == "Visualizations":
    st.subheader("Visualizations")
    num_cols = filtered_df.select_dtypes(include="number").columns.tolist()
    cat_cols = filtered_df.select_dtypes(exclude="number").columns.tolist()
    graph_types = ["Scatter", "Line", "Bar", "Box"]
    graph_type = st.selectbox("Graph Type", graph_types)
    x_axis = st.selectbox("X-axis", num_cols + cat_cols)
    y_axis = st.selectbox("Y-axis", num_cols, index=1 if len(num_cols) > 1 else 0)
    plot_btn = st.button("Plot Graph")
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

elif section == "Correlation":
    st.subheader("Correlation Heatmap")
    import seaborn as sns
    import matplotlib.pyplot as plt
    # Only use numeric columns for correlation
    corr = filtered_df.select_dtypes(include="number").corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

elif section == "ML Models":
    st.subheader("Machine Learning Models")
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    import numpy as np

    target_col = st.selectbox("Select Target Column", filtered_df.columns)
    feature_cols = st.multiselect("Select Feature Columns", [col for col in filtered_df.columns if col != target_col], default=[col for col in filtered_df.columns if col != target_col])
    test_size = st.slider("Test Size (fraction)", 0.1, 0.5, 0.2, 0.05)
    algo = st.selectbox("Algorithm", ["KNN", "Random Forest", "Logistic Regression", "Linear Regression"])
    run_ml = st.button("Run Model")

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

elif section == "Download Data":
    st.subheader("Download Filtered Dataset")
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=filtered_df.to_csv(index=False),
        file_name="filtered_diabetes.csv",
        mime="text/csv"
    )
