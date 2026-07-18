"""
Smart Waste Management using Artificial Intelligence
Streamlit web application: dataset explorer, EDA, predictions, model info.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import streamlit as st

st.set_page_config(
    page_title="Smart Waste Management AI",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURE_COLS = ['Current_Fill_%', 'Weight_kg', 'Temperature_C', 'Humidity_%',
                 'Methane_ppm', 'CO2_ppm', 'Battery_Level_%',
                 'Hours_Since_Last_Collection', 'Sensor_Status',
                 'Motion_Detected', 'Rain_Detected']


@st.cache_resource
def load_model_and_metadata():
    with open('smart_waste_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    return model, metadata


@st.cache_data
def load_default_dataset():
    return pd.read_csv('dataset.csv')


model, metadata = load_model_and_metadata()
encoders = metadata['encoders']  # {'Sensor_Status': {0:'Fault',1:'Healthy',2:'Warning'}, ...}


def label_to_code(col, label):
    """Reverse lookup: category label -> integer code used at training time."""
    for code, lab in encoders[col].items():
        if lab == label:
            return code
    return 0


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🗑️ Smart Waste AI")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 Dataset Explorer", "📈 Data Analysis", "🔮 Predictions", "ℹ️ Model Info"],
)

st.sidebar.markdown("---")
st.sidebar.caption("AI-Powered IoT Solution for Intelligent Waste Collection")

# ---------------------------------------------------------------------------
# HOME
# ---------------------------------------------------------------------------
if page == "🏠 Home":
    st.title("🗑️ Smart Waste Management using Artificial Intelligence")
    st.markdown("### AI-Powered IoT Solution for Intelligent Waste Collection")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model Accuracy", f"{metadata['accuracy']*100:.1f}%")
    col2.metric("Training Records", f"{metadata['n_train']:,}")
    col3.metric("Features Used", len(metadata['feature_cols']))
    col4.metric("Algorithm", "Random Forest")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("What this app does")
        st.markdown("""
        - 📊 **Dataset Explorer** — upload and browse IoT sensor data
        - 📈 **Data Analysis** — interactive EDA visualizations
        - 🔮 **Predictions** — real-time bin overflow prediction
        - ℹ️ **Model Info** — feature importance & performance metrics
        """)
    with c2:
        st.subheader("How it works")
        st.markdown("""
        ```
        IoT Sensors (Bins)
              ↓
        Real-time Data Collection
              ↓
        ML Model (Random Forest)
              ↓
        Overflow Prediction
              ↓
        Optimal Collection Schedule
        ```
        """)

    st.info("👈 Use the sidebar to explore the dataset, view analysis, or make a live prediction.")

# ---------------------------------------------------------------------------
# DATASET EXPLORER
# ---------------------------------------------------------------------------
elif page == "📊 Dataset Explorer":
    st.title("📊 Dataset Explorer")

    uploaded = st.file_uploader("Upload your own CSV (optional)", type="csv")
    df = pd.read_csv(uploaded) if uploaded is not None else load_default_dataset()

    st.success(f"Loaded dataset with **{df.shape[0]:,}** rows and **{df.shape[1]}** columns.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", f"{df.shape[0]:,}")
    c2.metric("Total Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.subheader("Preview")
    n_rows = st.slider("Rows to display", 5, 100, 10)
    st.dataframe(df.head(n_rows), use_container_width=True)

    st.subheader("Column Summary")
    st.dataframe(df.describe(include='all').T, use_container_width=True)

    if 'Overflow' in df.columns:
        st.subheader("Overflow Distribution")
        st.bar_chart(df['Overflow'].value_counts())

# ---------------------------------------------------------------------------
# DATA ANALYSIS (EDA)
# ---------------------------------------------------------------------------
elif page == "📈 Data Analysis":
    st.title("📈 Exploratory Data Analysis")
    df = load_default_dataset()

    tab1, tab2, tab3 = st.tabs(["Distributions", "Correlation", "Categorical Breakdown"])

    with tab1:
        st.subheader("Numeric Feature Distributions")
        numeric_cols = ['Current_Fill_%', 'Weight_kg', 'Temperature_C', 'Humidity_%',
                         'Methane_ppm', 'CO2_ppm', 'Battery_Level_%', 'Hours_Since_Last_Collection']
        col_choice = st.selectbox("Choose a feature", numeric_cols)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df[col_choice], bins=30, kde=True, color="steelblue", ax=ax)
        ax.set_title(f"Distribution of {col_choice}")
        st.pyplot(fig)

    with tab2:
        st.subheader("Correlation Heatmap")
        enc_df = df.copy()
        for col in ['Sensor_Status', 'Motion_Detected', 'Rain_Detected', 'Overflow']:
            enc_df[col] = enc_df[col].astype('category').cat.codes
        corr_cols = FEATURE_COLS + ['Overflow']
        corr = enc_df[corr_cols].corr()
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    square=True, linewidths=0.5, ax=ax)
        st.pyplot(fig)

    with tab3:
        st.subheader("Overflow Rate by Category")
        cat_col = st.selectbox("Choose a categorical feature",
                                ['Waste_Type', 'Location', 'Sensor_Status', 'AI_Priority', 'Route_Assigned'])
        rate = df.groupby(cat_col)['Overflow'].apply(lambda s: (s == 'Yes').mean() * 100)
        st.bar_chart(rate)

# ---------------------------------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------------------------------
elif page == "🔮 Predictions":
    st.title("🔮 Real-Time Overflow Prediction")
    st.markdown("Adjust the sensor readings below, then click **Predict Overflow**.")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Container**")
        fill_pct = st.slider("Current Fill %", 0, 100, 75)
        weight = st.slider("Weight (kg)", 0.0, 100.0, 60.0)
        hours_since = st.slider("Hours Since Last Collection", 0, 72, 30)

    with c2:
        st.markdown("**Environment**")
        temp = st.slider("Temperature (°C)", 0.0, 50.0, 28.0)
        humidity = st.slider("Humidity %", 0, 100, 65)
        rain = st.selectbox("Rain Detected", encoders['Rain_Detected'].values())

    with c3:
        st.markdown("**Gas & Device**")
        methane = st.slider("Methane (ppm)", 0, 1000, 350)
        co2 = st.slider("CO2 (ppm)", 0, 2000, 700)
        battery = st.slider("Battery Level %", 0, 100, 75)
        sensor_status = st.selectbox("Sensor Status", encoders['Sensor_Status'].values())
        motion = st.selectbox("Motion Detected", encoders['Motion_Detected'].values())

    if st.button("🚀 Predict Overflow", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            'Current_Fill_%': fill_pct,
            'Weight_kg': weight,
            'Temperature_C': temp,
            'Humidity_%': humidity,
            'Methane_ppm': methane,
            'CO2_ppm': co2,
            'Battery_Level_%': battery,
            'Hours_Since_Last_Collection': hours_since,
            'Sensor_Status': label_to_code('Sensor_Status', sensor_status),
            'Motion_Detected': label_to_code('Motion_Detected', motion),
            'Rain_Detected': label_to_code('Rain_Detected', rain),
        }])[FEATURE_COLS]

        pred = model.predict(row)[0]
        proba = model.predict_proba(row)[0]
        overflow_label = encoders['Overflow'][pred]

        st.markdown("---")
        if overflow_label == 'Yes':
            st.error(f"⚠️ **OVERFLOW DETECTED** — Confidence: {max(proba)*100:.1f}%")
        else:
            st.success(f"✅ **No Overflow** — Confidence: {max(proba)*100:.1f}%")

        prob_df = pd.DataFrame({
            'Outcome': [encoders['Overflow'][0], encoders['Overflow'][1]],
            'Probability': proba,
        }).set_index('Outcome')
        st.bar_chart(prob_df)

# ---------------------------------------------------------------------------
# MODEL INFO
# ---------------------------------------------------------------------------
elif page == "ℹ️ Model Info":
    st.title("ℹ️ Model Information")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Algorithm", "Random Forest")
    c2.metric("Trees", 100)
    c3.metric("Accuracy", f"{metadata['accuracy']*100:.2f}%")
    c4.metric("Features", len(metadata['feature_cols']))

    st.subheader("Feature Importance")
    imp_df = pd.DataFrame(metadata['feature_importance'])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(imp_df['Feature'], imp_df['Importance'], color="steelblue")
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    st.pyplot(fig)
    st.dataframe(imp_df, use_container_width=True)

    st.subheader("Confusion Matrix")
    cm = np.array(metadata['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Overflow', 'Overflow'],
                yticklabels=['No Overflow', 'Overflow'], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.caption(f"Trained on {metadata['n_train']:,} records · Tested on {metadata['n_test']:,} records")
