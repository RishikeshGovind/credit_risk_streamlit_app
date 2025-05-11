
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
    return pd.read_csv("german_credit_data.csv")

df = load_data()

st.title("German Credit Risk Analysis Dashboard")

# Sidebar Filters
with st.sidebar:
    st.header("Filter Data")
    sex = st.selectbox("Select Sex", options=["All"] + list(df['Sex'].unique()))
    job = st.selectbox("Select Job", options=["All"] + list(df['Job'].unique()))
    housing = st.selectbox("Select Housing", options=["All"] + list(df['Housing'].unique()))

    filtered_df = df.copy()
    if sex != "All":
        filtered_df = filtered_df[filtered_df['Sex'] == sex]
    if job != "All":
        filtered_df = filtered_df[filtered_df['Job'] == job]
    if housing != "All":
        filtered_df = filtered_df[filtered_df['Housing'] == housing]

# Show dataset
with st.expander("📊 View Filtered Dataset"):
    st.dataframe(filtered_df)

# Encode categorical columns
df_model = df.copy()
categorical_cols = df_model.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df_model[col] = LabelEncoder().fit_transform(df_model[col])

X = df_model.drop('Risk', axis=1)
y = df_model['Risk'].apply(lambda x: 1 if x == 'good' else 0)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model training
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

# SHAP
st.subheader("🔍 SHAP Feature Importance")
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test[:100])
fig = shap.plots.bar(shap_values, show=False)
st.pyplot(bbox_inches='tight')
