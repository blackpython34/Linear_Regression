import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Linear Regression Learning System", layout="wide", initial_sidebar_state="expanded")

# CSS for exact UI matching
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background-color: #151822; /* Dark background */
    color: #e2e8f0;
}

[data-testid="stSidebar"] {
    background-color: #10121B !important; 
    border-right: 1px solid #1e293b;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="stSidebarNav"] {
    display: none !important; 
}

/* Sidebar button base - Aggressive left alignment */
[data-testid="stSidebar"] .stButton button {
    display: flex !important;
    justify-content: flex-start !important;
    align-items: center !important;
    padding: 0.75rem 1rem !important;
    border-radius: 0 !important;
    width: 100% !important;
    border: none !important;
    text-align: left !important;
}

[data-testid="stSidebar"] .stButton button > div,
[data-testid="stSidebar"] .stButton button div[data-testid="stMarkdownContainer"] {
    display: flex !important;
    justify-content: flex-start !important;
    align-items: center !important;
    width: 100% !important;
    text-align: left !important;
}

[data-testid="stSidebar"] .stButton button p {
    text-align: left !important;
    margin: 0 !important;
    font-size: 0.95rem !important;
    width: 100% !important;
    display: block !important;
    white-space: normal !important; /* allow wrapping but keep left-aligned */
}

/* Primary (Active) Button */
[data-testid="stSidebar"] .stButton button[kind="primary"] {
    background: linear-gradient(90deg, rgba(59, 130, 246, 0.25) 0%, rgba(59, 130, 246, 0.05) 100%) !important; 
    border: 1px solid rgba(59, 130, 246, 0.4) !important;
    border-left: 4px solid #3b82f6 !important;
    border-radius: 8px !important;
    margin-bottom: 4px !important;
}

[data-testid="stSidebar"] .stButton button[kind="primary"] p {
    color: #60a5fa !important;
    font-weight: 700 !important;
}

/* Secondary (Inactive) Button */
[data-testid="stSidebar"] .stButton button[kind="secondary"] {
    background-color: transparent !important;
    border-left: 4px solid transparent !important;
}

[data-testid="stSidebar"] .stButton button[kind="secondary"] p {
    color: #9ca3af !important;
    font-weight: 500 !important;
}

[data-testid="stSidebar"] .stButton button[kind="secondary"]:hover {
    background-color: #1c2333 !important;
}

[data-testid="stSidebar"] .stButton button[kind="secondary"]:hover p {
    color: #e2e8f0 !important;
}

.edu-panel {
    background-color: #1e1b4b; /* Deep purple background */
    border-left: 4px solid #8b5cf6;
    padding: 20px 24px;
    border-radius: 8px;
    margin-bottom: 24px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.edu-title {
    color: #d8b4fe;
    font-weight: 600;
    margin-bottom: 12px;
    font-size: 1.1rem;
    display: flex;
    align-items: center;
}

.edu-content {
    color: #9ca3af;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* Make st.container(border=True) look like cards with colorful accents */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px !important;
    border: 1px solid #2d3748 !important;
    border-top: 4px solid #8b5cf6 !important; /* Vibrant purple accent */
    background: linear-gradient(180deg, #1e2433 0%, #151822 100%) !important;
    padding: 1.5rem !important;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
}

.custom-metric {
    background: linear-gradient(180deg, rgba(30, 41, 59, 1) 0%, rgba(30, 41, 59, 0.4) 100%);
    border-radius: 8px;
    padding: 10px;
    text-align: center;
    border: 1px solid #334155;
    border-top: 3px solid #3b82f6; /* Blue accent */
    display: flex;
    flex-direction: column;
    justify-content: center;
    height: 100%;
}

.metric-label {
    color: #9ca3af;
    font-size: 0.8rem;
    margin-bottom: 4px;
    font-weight: 500;
}

.metric-value {
    color: #60a5fa;
    font-size: 1.5rem;
    font-weight: 700;
}

h1, h2, h3, h4 {
    font-family: 'Inter', sans-serif !important;
}

.gradient-header {
    background: -webkit-linear-gradient(45deg, #4facfe, #00f2fe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 25px !important;
    font-weight: 800;
}

.stButton > button {
    background-color: #3b82f6 !important;
    color: white !important;
    border-radius: 6px !important;
    border: none !important;
    font-weight: 500 !important;
}

.stButton > button:hover {
    background-color: #2563eb !important;
}

.block-container {
    padding-top: 3rem !important;
    padding-bottom: 3rem !important;
    max-width: 1200px !important;
}
</style>
""", unsafe_allow_html=True)

# State initialization
if 'current_step' not in st.session_state:
    st.session_state.current_step = '1. Dataset Input'
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = []
if 'model_params' not in st.session_state:
    st.session_state.model_params = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'cost_history' not in st.session_state:
    st.session_state.cost_history = []

steps = [
    ("🗄️", "1. Dataset Input"),
    ("🎛️", "2. Preprocessing"),
    ("📊", "3. Exploratory Data Analysis"),
    ("🧠", "4. Learning Module"),
    ("▶️", "5. Training Config"),
    ("📈", "6. Prediction & Evaluation")
]

# Sidebar Menu
st.sidebar.markdown("""
<div style='padding: 10px 0px 20px 0px;'>
    <h2 style='background: -webkit-linear-gradient(45deg, #8b5cf6, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0px; font-size: 1.8rem; font-weight: 800;'>Linear Regression</h2>
    <p style='color: #60a5fa; margin-top: 5px; font-size: 0.95rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;'>Learning System</p>
</div>
""", unsafe_allow_html=True)

for icon, step_name in steps:
    is_active = (st.session_state.current_step == step_name)
    if st.sidebar.button(f"{icon}  {step_name}", key=step_name, use_container_width=True, type="primary" if is_active else "secondary"):
        st.session_state.current_step = step_name
        st.rerun()

st.sidebar.markdown("""
<div style='margin-top: 50px; padding-left: 15px; color: #64748b; font-size: 0.75rem;'>
    Developed for interactive learning.
</div>
""", unsafe_allow_html=True)

def edu_panel(title, content):
    st.markdown(f"""
    <div class="edu-panel">
        <div class="edu-title">{title}</div>
        <div class="edu-content">{content}</div>
    </div>
    """, unsafe_allow_html=True)

def step1_dataset():
    st.markdown("<h2 class='gradient-header'>1. Dataset Upload & Preview</h2>", unsafe_allow_html=True)
    edu_panel("💡 Why start here?", "Every Machine Learning pipeline begins with data. Linear Regression models learn relationships from historical data. The quality, size, and format of your dataset directly dictate the success of the model.")
    
    with st.container(border=True):
        if st.session_state.raw_data is None:
            st.markdown("<h3 style='margin-bottom:15px; font-size: 1.2rem;'>📄 Upload Dataset</h3>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv")
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.raw_data = df
                    st.rerun()
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        else:
            df = st.session_state.raw_data
            
            col1, col2, col3, col4 = st.columns([2.5, 1, 1, 1.5])
            with col1:
                st.markdown("<div style='display:flex; align-items:center; height: 100%;'><h3 style='margin:0; font-size:1.3rem;'>📄 Dataset Loaded</h3></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='custom-metric'><div class='metric-label'>Rows</div><div class='metric-value'>{df.shape[0]}</div></div>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"<div class='custom-metric'><div class='metric-label'>Columns</div><div class='metric-value'>{df.shape[1]}</div></div>", unsafe_allow_html=True)
            with col4:
                st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
                if st.button("Upload New Data", use_container_width=True):
                    st.session_state.raw_data = None
                    st.session_state.processed_data = None
                    st.rerun()
                    
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Data Preview (First 5 rows)**")
            st.dataframe(df.head(), use_container_width=True)

def step2_preprocessing():
    st.markdown("<h2 class='gradient-header'>2. Preprocessing Interface</h2>", unsafe_allow_html=True)
    edu_panel("💡 Why Preprocess Data?", "Models only understand numbers. Preprocessing converts raw data into a clean, mathematical format.<br><br>• <b>Missing Values:</b> Models cannot compute with missing data (NaN). We must either drop them or guess (impute) them.<br>• <b>Encoding:</b> Text categories like \"Red\", \"Blue\" must be converted to numbers (e.g., 0, 1).<br>• <b>Scaling:</b> If 'Age' is 0-100 and 'Salary' is 0-100000, Salary will dominate the model. Scaling puts them on the same range.")
    
    if st.session_state.raw_data is None:
        st.warning("Please upload a dataset in Step 1 first.")
        return
        
    df = st.session_state.raw_data.copy()
    
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("<h3 style='font-size: 1.2rem; margin-bottom: 15px;'>Feature Selection</h3>", unsafe_allow_html=True)
            target = st.selectbox("Target Variable (y):", df.columns, index=len(df.columns)-1)
            st.caption("The value you want to predict.")
            features = st.multiselect("Features (X):", [c for c in df.columns if c != target], default=[c for c in df.columns if c != target])
    
    with col2:
        with st.container(border=True):
            st.markdown("<h3 style='font-size: 1.2rem; margin-bottom: 15px;'>Transformations</h3>", unsafe_allow_html=True)
            missing_strategy = st.selectbox("Missing Values Handling:", ["Impute with Mean (Numeric only)", "Drop Rows with Missing Values"])
            scaling_strategy = st.selectbox("Feature Scaling:", ["None", "Standardization (Z-score)", "Normalization (Min-Max)"])
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Apply Preprocessing", use_container_width=True):
                if not features:
                    st.error("Please select at least one feature.")
                    return
                    
                st.session_state.target_col = target
                st.session_state.feature_cols = features
                
                if missing_strategy == "Drop Rows with Missing Values":
                    df = df.dropna()
                else:
                    for col in df.columns:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            df[col] = df[col].fillna(df[col].mean())
                        else:
                            df = df.dropna(subset=[col])
                            
                cols_to_encode = features + [target]
                for col in cols_to_encode:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].astype('category').cat.codes
                        
                if scaling_strategy == "Standardization (Z-score)":
                    for col in features:
                        std = df[col].std()
                        df[col] = (df[col] - df[col].mean()) / (std if std != 0 else 1)
                elif scaling_strategy == "Normalization (Min-Max)":
                    for col in features:
                        min_val, max_val = df[col].min(), df[col].max()
                        if min_val != max_val:
                            df[col] = (df[col] - min_val) / (max_val - min_val)
                        else:
                            df[col] = 0.0
                            
                st.session_state.processed_data = df
                st.success("Preprocessing Applied!")
                
    if st.session_state.processed_data is not None and st.session_state.feature_cols:
        st.markdown("<h3 style='color:#f8fafc; margin-top:30px; margin-bottom:15px;'>Processed Data Preview</h3>", unsafe_allow_html=True)
        with st.container(border=True):
            display_cols = st.session_state.feature_cols + [st.session_state.target_col]
            st.dataframe(st.session_state.processed_data[display_cols].head(10), use_container_width=True)

def step3_eda():
    st.markdown("<h2 class='gradient-header'>3. Exploratory Data Analysis (EDA)</h2>", unsafe_allow_html=True)
    edu_panel("💡 Why Visualize?", "Before building a Linear Regression model, we need to understand our data. Correlation helps identify strong predictors. Distributions show if features are normally distributed. Relationship plots verify if a linear trend exists between features and the target.")
    
    if st.session_state.processed_data is None:
        st.warning("Please complete Step 2 first.")
        return
        
    df = st.session_state.processed_data
    features = st.session_state.feature_cols
    target = st.session_state.target_col
    
    # 1. Feature Distributions
    st.markdown("<h3 style='color:#f8fafc; margin-top:10px; margin-bottom:15px;'>1. Feature Distributions</h3>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("**Analyze the distribution of individual features**")
        st.caption("Check for skewness or outliers using histograms and box plots.")
        dist_feature = st.selectbox("Select Feature to view distribution:", features, key="dist_feat")
        
        col1, col2 = st.columns(2)
        with col1:
            fig_hist = px.histogram(df, x=dist_feature, nbins=30, color_discrete_sequence=['#8b5cf6'])
            fig_hist.update_layout(title="Histogram", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
            st.plotly_chart(fig_hist, use_container_width=True)
        with col2:
            fig_box = px.box(df, y=dist_feature, color_discrete_sequence=['#3b82f6'])
            fig_box.update_layout(title="Box Plot", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
            st.plotly_chart(fig_box, use_container_width=True)

    # 2. Correlation Analysis
    st.markdown("<h3 style='color:#f8fafc; margin-top:30px; margin-bottom:15px;'>2. Correlation Analysis</h3>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("**Correlation Matrix**")
        st.caption("Identify how features relate to each other and to the target. Values near 1 or -1 indicate strong linear relationships.")
        
        corr_matrix = df[features + [target]].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
        fig_corr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
        st.plotly_chart(fig_corr, use_container_width=True)
        
    # 3. Relationship Visualization
    st.markdown("<h3 style='color:#f8fafc; margin-top:30px; margin-bottom:15px;'>3. Relationship Visualization</h3>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(f"**How does each feature affect {target}?**")
        st.caption("Look for clear linear trends. If the data is scattered randomly, the feature might not be a good linear predictor.")
        rel_feature = st.selectbox("Select Feature to plot against Target:", features, key="rel_feat")
        
        fig_scatter = px.scatter(df, x=rel_feature, y=target, trendline="ols" if len(df) > 1 else None, color_discrete_sequence=['#4facfe'])
        fig_scatter.update_layout(title=f"{rel_feature} vs {target}", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
        st.plotly_chart(fig_scatter, use_container_width=True)

def step4_learning():
    st.markdown("<h2 class='gradient-header'>4. Linear Regression Learning Module</h2>", unsafe_allow_html=True)
    if st.session_state.processed_data is None:
        st.warning("Please complete Preprocessing first.")
        return
        
    df = st.session_state.processed_data
    features = st.session_state.feature_cols
    target = st.session_state.target_col
    
    is_simple = len(features) == 1
    
    st.markdown("<h3 style='color:#f8fafc; margin-top:10px; margin-bottom:15px;'>Theoretical Foundations</h3>", unsafe_allow_html=True)
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Simple Linear Regression**")
            st.markdown("Models the relationship between exactly *one* independent variable and the target.")
            st.latex(r"y = \theta_0 + \theta_1 x_1")
        with col2:
            st.markdown("**Multiple Linear Regression**")
            st.markdown("Models the relationship between *multiple* independent variables and the target.")
            st.latex(r"y = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n")
            
        if is_simple:
            st.info(f"You are currently analyzing a **Simple Linear Regression** because you selected 1 feature: `{features[0]}`.")
        else:
            st.info(f"You are currently analyzing a **Multiple Linear Regression** because you selected {len(features)} features.")

    with st.container(border=True):
        st.markdown("<div class='edu-title'>📖 1. Hypothesis Formulation</div>", unsafe_allow_html=True)
        st.markdown("The hypothesis $h_\\theta(x)$ represents our model's prediction.")
        st.latex(r"h_\theta(x) = \theta_0 + \theta_1 x_1 + \dots + \theta_n x_n")
        st.markdown("• **$\\theta_0$**: y-intercept (bias)\n\n• **$\\theta_1 \dots \\theta_n$**: weights for features\n\n• **$x_1 \dots x_n$**: feature values")

    with st.container(border=True):
        st.markdown("<div class='edu-title'>📖 2. Cost Function Definition</div>", unsafe_allow_html=True)
        st.markdown("The Cost Function $J(\\theta)$ measures the average squared difference between predictions and actual values. Minimizing this gives the best model.")
        st.latex(r"J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2")
    
    st.markdown("<h3 style='color:#f8fafc; margin-top:20px; margin-bottom:15px;'>Step-by-Step Computation (Sample 1)</h3>", unsafe_allow_html=True)
    with st.container(border=True):
        sample = df.iloc[0]
        
        st.markdown("**Step A: Intermediate Values Setup**")
        st.markdown("Let's assume initial random weights for our model:")
        st.code("Bias (θ₀) = 0.0\nWeights (θ₁...θₙ) = 0.5", language="text")
        
        st.markdown("Feature values extracted for the very first sample in your dataset:")
        feature_vals = "\n".join([f"{f} (x_{i+1}) = {sample[f]:.4f}" for i, f in enumerate(features)])
        st.code(feature_vals, language="text")
        st.markdown(f"**Actual Target Value (y) = {sample[target]:.4f}**")
        
        st.markdown("---")
        
        st.markdown("**Step B: Hypothesis Computation (Formula Used)**")
        st.markdown("Formula: $h_\\theta(x) = \\theta_0 + (\\theta_1 \\times x_1) + (\\theta_2 \\times x_2) + \\dots$")
        prediction_str = "0.0 + " + " + ".join([f"(0.5 * {sample[f]:.4f})" for f in features])
        prediction_val = 0.0 + sum([0.5 * sample[f] for f in features])
        st.code(f"h_θ(x) = {prediction_str}\nh_θ(x) = {prediction_val:.4f}", language="text")
        
        st.markdown("---")
        
        st.markdown("**Step C: 3. Error Computation**")
        st.markdown("Formula: $Error = h_\\theta(x) - y$")
        error = prediction_val - sample[target]
        st.code(f"Error = {prediction_val:.4f} - {sample[target]:.4f}\nError = {error:.4f}", language="text")
        
    with st.container(border=True):
        st.markdown("<div class='edu-title'>📖 4. Parameter Learning</div>", unsafe_allow_html=True)
        st.markdown("To reduce the error computed above, we use **Gradient Descent** to update the weights iteratively. We subtract a fraction (learning rate $\\alpha$) of the gradient from the current weights.")
        st.latex(r"\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}")

def step5_training():
    st.markdown("<h2 class='gradient-header'>5. Training Configuration & Visualization</h2>", unsafe_allow_html=True)
    if st.session_state.processed_data is None:
        st.warning("Please complete Preprocessing first.")
        return
        
    df = st.session_state.processed_data
    features = st.session_state.feature_cols
    target = st.session_state.target_col
    
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("<h3 style='font-size: 1.2rem; margin-bottom: 15px;'>Hyperparameters</h3>", unsafe_allow_html=True)
            split_ratio = st.slider("Train-Test Split (%)", 50, 90, 80)
            lr = st.number_input("Learning Rate (α)", value=0.01, step=0.001, format="%f")
            epochs = st.number_input("Epochs (Iterations)", value=100, step=10)
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Start Training", use_container_width=True):
                X = df[features].values
                y = df[target].values
                
                m = len(y)
                train_size = int(m * (split_ratio / 100))
                X_train, y_train = X[:train_size], y[:train_size]
                X_test, y_test = X[train_size:], y[train_size:]
                
                w = np.zeros(X.shape[1])
                b = 0.0
                history = []
                
                progress_bar = st.progress(0)
                
                for epoch in range(epochs):
                    y_pred = np.dot(X_train, w) + b
                    error = y_pred - y_train
                    
                    cost = np.sum(error ** 2) / (2 * train_size)
                    history.append(cost)
                    
                    w_grad = np.dot(X_train.T, error) / train_size
                    b_grad = np.sum(error) / train_size
                    
                    w -= lr * w_grad
                    b -= lr * b_grad
                    
                    if epoch % max(1, epochs//100) == 0:
                        progress_bar.progress((epoch + 1) / epochs)
                
                progress_bar.empty()
                
                st.session_state.model_params = {'w': w, 'b': b}
                st.session_state.cost_history = history
                
                if len(y_test) > 0:
                    y_test_pred = np.dot(X_test, w) + b
                    mse = np.mean((y_test_pred - y_test)**2)
                    mae = np.mean(np.abs(y_test_pred - y_test))
                    ss_tot = np.sum((y_test - np.mean(y_test))**2)
                    ss_res = np.sum((y_test - y_test_pred)**2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                else:
                    mse, mae, r2 = 0, 0, 0
                    
                st.session_state.metrics = {'mse': mse, 'mae': mae, 'r2': r2}
                
    with col2:
        with st.container(border=True):
            st.markdown("<h3 style='font-size: 1.2rem; margin-bottom: 15px;'>Learned Parameters</h3>", unsafe_allow_html=True)
            if st.session_state.model_params:
                params = st.session_state.model_params
                st.markdown(f"**Intercept (θ₀):** `{params['b']:.4f}`")
                for i, f in enumerate(features):
                    st.markdown(f"**{f} (θ_{i+1}):** `{params['w'][i]:.4f}`")
                
                st.success("Training Complete!")
            else:
                st.info("Train the model to see parameters.")
                
    if st.session_state.cost_history:
        with st.container(border=True):
            st.markdown("<h3 style='font-size: 1.2rem;'>Cost Convergence</h3>", unsafe_allow_html=True)
            fig = px.line(x=range(1, len(st.session_state.cost_history)+1), y=st.session_state.cost_history, labels={'x': 'Epoch', 'y': 'Cost J(θ)'})
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
            st.plotly_chart(fig, use_container_width=True)

def step6_prediction():
    st.markdown("<h2 class='gradient-header'>6. Prediction & Evaluation</h2>", unsafe_allow_html=True)
    if not st.session_state.metrics:
        st.warning("Please train the model first.")
        return
        
    metrics = st.session_state.metrics
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='custom-metric'><div class='metric-label'>Mean Squared Error (MSE)</div><div class='metric-value'>{metrics['mse']:.4f}</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='custom-metric'><div class='metric-label'>Mean Absolute Error (MAE)</div><div class='metric-value'>{metrics['mae']:.4f}</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='custom-metric'><div class='metric-label'>R² Score</div><div class='metric-value'>{metrics['r2']:.4f}</div></div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown("<h3 style='font-size: 1.2rem; margin-bottom: 15px;'>Make a Prediction</h3>", unsafe_allow_html=True)
        params = st.session_state.model_params
        features = st.session_state.feature_cols
        
        col_input, col_result = st.columns(2)
        
        with col_input:
            inputs = {}
            for f in features:
                inputs[f] = st.number_input(f"{f}", value=0.0)
                
        with col_result:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Predict", use_container_width=True):
                pred = params['b'] + sum([params['w'][i] * inputs[f] for i, f in enumerate(features)])
                st.markdown(f"<div style='background-color:#1e1b4b; padding:20px; border-radius:8px; border-left: 4px solid #8b5cf6; text-align:center;'><div style='color:#d8b4fe; font-size:1.1rem; margin-bottom:10px;'>Predicted {st.session_state.target_col}</div><div style='color:#8b5cf6; font-size:2.5rem; font-weight:bold;'>{pred:.4f}</div></div>", unsafe_allow_html=True)
                
                st.markdown("<br>**Step-by-Step Computation**", unsafe_allow_html=True)
                calc_str = f"{params['b']:.4f} + " + " + ".join([f"({params['w'][i]:.4f} * {inputs[f]:.4f})" for i, f in enumerate(features)])
                st.code(f"y = {calc_str}\n  = {pred:.4f}", language="text")

# Render appropriate step
if st.session_state.current_step == '1. Dataset Input':
    step1_dataset()
elif st.session_state.current_step == '2. Preprocessing':
    step2_preprocessing()
elif st.session_state.current_step == '3. Exploratory Data Analysis':
    step3_eda()
elif st.session_state.current_step == '4. Learning Module':
    step4_learning()
elif st.session_state.current_step == '5. Training Config':
    step5_training()
elif st.session_state.current_step == '6. Prediction & Evaluation':
    step6_prediction()

