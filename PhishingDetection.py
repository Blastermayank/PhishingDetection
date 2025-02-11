import streamlit as st
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from FeatureExtraction import extract_features

# Set Streamlit page title and layout
st.set_page_config(page_title="Phishing Detection Tool", layout="wide")

# Paths to folders
IMAGES_DIR = "Images"
MODELS_DIR = "Models"

# ----------------------------------
# 🔹 PROJECT DESCRIPTION
# ----------------------------------
st.title("🔍 Phishing Detection Tool")
st.markdown("""
### **Project Overview**
Phishing is a cyber attack where fraudulent websites trick users into providing sensitive information like passwords and credit card details.  
This tool uses **Machine Learning models** to detect whether a URL is **Legitimate or Phishing**.  

### **How it Works?**
1. Enter a URL in the input box below.
2. Select a **Machine Learning Model** to classify the URL.
3. The system will analyze the URL and **predict whether it's safe or a phishing attempt**.

Below, you can also explore dataset insights and model performance!
""")

st.divider()  # Adds a horizontal line

# ----------------------------------
# 🔹 DATASET INSIGHTS (Collapsible Section)
# ----------------------------------
with st.expander("📊 Dataset Insights", expanded=False):
    st.subheader("Feature Correlation Heatmap & Phishing Distribution")

    # Layout: 2 columns for Heatmap and Pie Chart
    col1, col2 = st.columns(2)

    # Feature Correlation Heatmap
    heatmap_path = os.path.join(IMAGES_DIR, "heatmap.png")
    if os.path.exists(heatmap_path):
        with col1:
            st.image(heatmap_path, caption="Feature Correlation Heatmap", use_container_width=True)
    else:
        with col1:
            st.warning("Heatmap not found. Please run model training script.")

    # Phishing vs. Legitimate Pie Chart
    piechart_path = os.path.join(IMAGES_DIR, "pie_chart.png")
    if os.path.exists(piechart_path):
        with col2:
            st.image(piechart_path, caption="Phishing vs. Legitimate Distribution", use_container_width=True)
    else:
        with col2:
            st.warning("Pie chart not found. Please run model training script.")

st.divider()

# ----------------------------------
# 🔹 MODEL PERFORMANCE COMPARISON (Collapsible Section)
# ----------------------------------
with st.expander("📊 Model Performance Comparison", expanded=False):
    st.subheader("Compare Different Models")

    # Layout: 3 columns for comparison charts
    col1, col2, col3 = st.columns(3)

    # Accuracy Comparison
    accuracy_path = os.path.join(IMAGES_DIR, "accuracy_comparison.png")
    if os.path.exists(accuracy_path):
        with col1:
            st.image(accuracy_path, caption="Model Accuracy Comparison", use_container_width=True)

    # R-Squared Comparison
    r2_path = os.path.join(IMAGES_DIR, "r_squared_comparison.png")
    if os.path.exists(r2_path):
        with col2:
            st.image(r2_path, caption="Model R-Squared Comparison", use_container_width=True)

    # Adjusted R-Squared Comparison
    adj_r2_path = os.path.join(IMAGES_DIR, "adjusted_r_squared_comparison.png")
    if os.path.exists(adj_r2_path):
        with col3:
            st.image(adj_r2_path, caption="Model Adjusted R-Squared Comparison", use_container_width=True)

    # Additional Performance Metrics
    col1, col2, col3 = st.columns(3)

    # Precision Comparison
    precision_path = os.path.join(IMAGES_DIR, "precision_comparison.png")
    if os.path.exists(precision_path):
        with col1:
            st.image(precision_path, caption="Model Precision Comparison", use_container_width=True)

    # Recall Comparison
    recall_path = os.path.join(IMAGES_DIR, "recall_comparison.png")
    if os.path.exists(recall_path):
        with col2:
            st.image(recall_path, caption="Model Recall Comparison", use_container_width=True)

    # F1-Score Comparison
    f1_path = os.path.join(IMAGES_DIR, "f1-score_comparison.png")
    if os.path.exists(f1_path):
        with col3:
            st.image(f1_path, caption="Model F1-Score Comparison", use_container_width=True)

st.divider()

# ----------------------------------
# 🔍 URL PHISHING DETECTION SECTION
# ----------------------------------
st.subheader("🔬 Detect Phishing URL")

# Input URL
url = st.text_input("Enter a URL to analyze:", placeholder="e.g. https://www.google.com")

# Dropdown to select a model
model_options = [
    "Decision Tree",
    "KNN",
    "Logistic Regression",
    "Random Forest",
    "SVM",
    "Naive Bayes"
]

selected_model = st.selectbox("Choose a Machine Learning Model", model_options)

# Load model and scaler
model_filename = f"{MODELS_DIR}/{selected_model.replace(' ', '_')}.joblib"
scaler_filename = f"{MODELS_DIR}/scaler.joblib"

# Button to analyze the URL
if st.button("🔍 Detect Phishing"):
    if url:
        if os.path.exists(model_filename) and os.path.exists(scaler_filename):
            # Load model and scaler
            model = joblib.load(model_filename)
            scaler = joblib.load(scaler_filename)

            # Extract features from the URL
            features = extract_features(url)
            features_df = pd.DataFrame([features])

            # Standardize features
            features_scaled = scaler.transform(features_df)

            # Predict using the model
            prediction = model.predict(features_scaled)

            # Display result
            if prediction[0] == 1:
                st.error("🚨 **Phishing Detected!** This URL is unsafe.")
            else:
                st.success("✅ **Legitimate Website!** This URL appears to be safe.")

        else:
            st.warning("⚠️ Model or Scaler not found. Please train models first.")
    else:
        st.warning("⚠️ Please enter a valid URL.")

st.divider()

# ----------------------------------
# 🔎 CONFUSION MATRIX FOR SELECTED MODEL
# ----------------------------------
st.subheader(f"🔎 Confusion Matrix for {selected_model}")

cm_img_path = f"{IMAGES_DIR}/confusion_matrix_{selected_model.replace(' ', '_')}.png"

if os.path.exists(cm_img_path):
    st.image(cm_img_path, caption=f"Confusion Matrix - {selected_model}", use_container_width=False, width=500)
else:
    st.warning("Confusion matrix not found. Please train models first.")

st.divider()

# ----------------------------------
# Footer
# ----------------------------------
st.markdown(
    """
    Developed by **Mayank, Sneha & Saumya, PG-DASSD, CDAC-Hyderabad** | Machine Learning-based Phishing Detection 🚀  
    """
)
