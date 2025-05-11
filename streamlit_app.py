
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import shap

st.set_page_config(page_title="German Credit Risk Analysis", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("german_credit_data.csv")
    if 'risk' not in df.columns:
        st.error("⚠️ The dataset is missing the 'risk' column required for modeling. Please upload the correct version.")
        st.stop()
    df.columns = [col.lower() for col in df.columns]  # standardize column names
    return df

df = load_data()

st.title("German Credit Risk Analysis Dashboard")

# Display available columns for debugging
st.write("Available columns:", df.columns.tolist())

# Sidebar Filters with safer access
with st.sidebar:
    st.header("Filter Data")

    sex_col = "sex"
    job_col = "job"
    housing_col = "housing"

    sex = st.selectbox("Select Sex", options=["All"] + sorted(df[sex_col].dropna().unique().tolist()) if sex_col in df else ["N/A"])
    job = st.selectbox("Select Job", options=["All"] + sorted(df[job_col].dropna().unique().tolist()) if job_col in df else ["N/A"])
    housing = st.selectbox("Select Housing", options=["All"] + sorted(df[housing_col].dropna().unique().tolist()) if housing_col in df else ["N/A"])

    filtered_df = df.copy()
    if sex_col in df and sex != "All":
        filtered_df = filtered_df[filtered_df[sex_col] == sex]
    if job_col in df and job != "All":
        filtered_df = filtered_df[filtered_df[job_col] == job]
    if housing_col in df and housing != "All":
        filtered_df = filtered_df[filtered_df[housing_col] == housing]

# Show filtered dataset
with st.expander("📊 View Filtered Dataset"):
    st.dataframe(filtered_df)

# Encode categorical columns
df_model = df.copy()
categorical_cols = df_model.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df_model[col] = LabelEncoder().fit_transform(df_model[col])

# Model training
X = df_model.drop('risk', axis=1)
y = df_model['risk'].apply(lambda x: 1 if x == 'good' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Model Evaluation
st.subheader("📈 Model Performance")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Confusion Matrix**")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    st.pyplot(fig)

with col2:
    st.markdown("**Classification Report**")
    st.text(classification_report(y_test, y_pred))

# ROC Curve
st.markdown("**ROC Curve**")
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend()
st.pyplot(fig)

# Precision-Recall Curve
st.markdown("**Precision-Recall Curve**")
precision, recall, _ = precision_recall_curve(y_test, y_proba)
fig, ax = plt.subplots()
ax.plot(recall, precision, marker='.')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
st.pyplot(fig)

# SHAP Feature Importance
st.subheader("🔍 SHAP Feature Importance")
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test[:100])
fig = shap.plots.bar(shap_values, show=False)
st.pyplot(bbox_inches='tight')
