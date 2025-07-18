import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

st.title("📊 Telecom Churn Prediction App")

# Upload cleaned dataset
uploaded_file = st.file_uploader("Upload your telecom customer CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("✅ Dataset Preview:")
    st.dataframe(df.head())

    # Encoding
    df_encoded = df.copy()
    cat_cols = df.select_dtypes(include="object").columns
    le = LabelEncoder()
    for col in cat_cols:
        df_encoded[col] = le.fit_transform(df_encoded[col])

    # Train a simple XGBoost model (for demo)
    if "Churn" in df_encoded.columns:
        X = df_encoded.drop("Churn", axis=1)
        y = df_encoded["Churn"]

        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        model.fit(X, y)

        # SHAP Explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        st.subheader("🔍 Feature Importance (SHAP)")
        shap.summary_plot(shap_values, X, show=False)
        st.pyplot(bbox_inches='tight')
    else:
        st.warning("Please include 'Churn' column in your dataset to train the model.")
