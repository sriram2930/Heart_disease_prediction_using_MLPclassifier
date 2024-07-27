import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Function to load and preprocess the dataset
@st.cache
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
    df = pd.read_csv(url, names=column_names)
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.apply(pd.to_numeric)
    return df

# Load dataset
df = load_data()

# Split data into features and target
X = df.drop('target', axis=1)
y = df['target']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30), max_iter=1000)
mlp.fit(X_train, y_train)

# Predict
y_pred = mlp.predict(X_test)

# Define the possible class labels
class_labels = [0, 1]

# Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred, labels=class_labels)
class_report = classification_report(y_test, y_pred, labels=class_labels)

# Streamlit app
st.title("Heart Disease Prediction using MLP Classifier")

st.header("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
st.pyplot(fig)

st.header("Classification Report")
st.text(class_report)

# Display dataset
st.header("Dataset")
st.write(df)
