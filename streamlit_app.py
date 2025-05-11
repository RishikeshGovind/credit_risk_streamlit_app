
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import shap

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('https://raw.githubusercontent.com/RishikeshGovind/Credit-card-risk-modelling/main/german_credit_data.csv')

df = load_data()

# Preprocessing
df = df.dropna()
df = pd.get_dummies(df, drop_first=True)

# Feature selection
X = df.drop('Risk_good', axis=1) if 'Risk_good' in df.columns else df.iloc[:, :-1]
y = df['Risk_good'] if 'Risk_good' in df.columns else df.iloc[:, -1]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Streamlit UI
st.title("Credit Risk Modeling Dashboard")

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
st.pyplot(fig)

st.subheader("ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic')
ax.legend()
st.pyplot(fig)

st.subheader("Precision-Recall Curve")
precision, recall, _ = precision_recall_curve(y_test, y_proba)
fig, ax = plt.subplots()
ax.plot(recall, precision, marker='.')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
st.pyplot(fig)

st.subheader("Feature Importance (SHAP)")
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test[:100])
st.set_option('deprecation.showPyplotGlobalUse', False)
shap.summary_plot(shap_values, X_test[:100], plot_type="bar", show=False)
st.pyplot(bbox_inches='tight')
