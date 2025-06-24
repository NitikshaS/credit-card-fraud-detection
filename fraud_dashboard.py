import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Load and clean data
df = pd.read_csv('creditcard.csv')
df.columns = df.columns.str.strip()  # clean column names

# Prepare label
df = df.rename(columns={'Class': 'is_fraud'})
fraud = df[df.is_fraud == 1]
normal = df[df.is_fraud == 0].sample(n=len(fraud), random_state=42)
df_balanced = pd.concat([fraud, normal])

# Split features/labels
X = df_balanced.drop('is_fraud', axis=1)
y = df_balanced['is_fraud']

# Scale
scaler = StandardScaler()
X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
roc_score = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

# ---- Streamlit UI ----
st.title("Credit Card Fraud Detection Dashboard")

# Fraud distribution
st.subheader("1. Fraud vs Normal Transaction Count")
fig1 = px.histogram(df_balanced, x="is_fraud", color="is_fraud",
                    labels={"is_fraud": "Transaction Type"},
                    title="Fraud vs Normal Count")
st.plotly_chart(fig1)

# Transaction amount boxplot
st.subheader("2. Transaction Amount by Class")
fig2 = px.box(df_balanced, x="is_fraud", y="Amount",
              labels={"is_fraud": "Transaction Type"},
              title="Transaction Amounts by Class")
st.plotly_chart(fig2)

# ROC curve
st.subheader("3. ROC Curve & AUC Score")
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig3 = plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
st.pyplot(fig3)

st.metric("ROC AUC Score", f"{roc_score:.2f}")

# Confusion Matrix
st.subheader("4. Confusion Matrix")
fig4 = plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig4)

# Show raw data (optional)
if st.checkbox("Show Raw Data"):
    st.write(df_balanced.sample(10))
