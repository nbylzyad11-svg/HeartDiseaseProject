import streamlit as st
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# إعداد الصفحة
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Risk Predictor")

st.write(
    "Machine Learning app to predict heart disease risk using Random Forest."
)

# مسارات الملفات
model_path = "heart_model.pkl"
columns_path = "model_columns.pkl"
metrics_path = "model_metrics.pkl"

# تحميل أو تدريب النموذج
if os.path.exists(model_path):

    model = joblib.load(model_path)
    model_columns = joblib.load(columns_path)
    metrics = joblib.load(metrics_path)

    st.success("Model loaded successfully")
    st.info(f"📊 Model Accuracy: **{metrics['accuracy']:.2%}** | F1 Score: **{metrics['f1']:.2%}**")

else:

    st.info("Training model... please wait")

    data = pd.read_csv("cardio_train.csv", sep=";")

    data = data.drop("id", axis=1)

    # تحويل العمر
    data["age"] = data["age"] / 365

    # تنظيف البيانات
    data = data[(data["ap_hi"] > 80) & (data["ap_hi"] < 200)]
    data = data[(data["ap_lo"] > 50) & (data["ap_lo"] < 150)]
    data = data[data["ap_hi"] > data["ap_lo"]]

    # Feature Engineering
    data["bmi"] = data["weight"] / ((data["height"] / 100) ** 2)
    data["pulse_pressure"] = data["ap_hi"] - data["ap_lo"]

    X = data.drop("cardio", axis=1)
    y = data["cardio"]

    model_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    # ✅ حساب الـ Accuracy و F1 Score
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics = {"accuracy": acc, "f1": f1}

    print(f"\n✅ Model Accuracy: {acc:.2%}")
    print(f"✅ F1 Score:       {f1:.2%}")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, model_path)
    joblib.dump(model_columns, columns_path)
    joblib.dump(metrics, metrics_path)

    st.success(f"✅ Model trained! Accuracy: **{acc:.2%}** | F1 Score: **{f1:.2%}**")

# Sidebar إدخال البيانات
st.sidebar.header("Enter Patient Data")

age = st.sidebar.slider("Age", 18, 100, 50)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

height = st.sidebar.number_input(
    "Height (cm)",
    120, 220, 170
)

weight = st.sidebar.number_input(
    "Weight (kg)",
    40, 200, 70
)

ap_hi = st.sidebar.number_input(
    "Systolic BP",
    80, 200, 120
)

ap_lo = st.sidebar.number_input(
    "Diastolic BP",
    50, 150, 80
)

cholesterol = st.sidebar.selectbox(
    "Cholesterol",
    ["Normal", "Above normal", "High"]
)

gluc = st.sidebar.selectbox(
    "Glucose",
    ["Normal", "Above normal", "High"]
)

smoke = st.sidebar.selectbox(
    "Smoker?",
    ["No", "Yes"]
)

alco = st.sidebar.selectbox(
    "Alcohol intake?",
    ["No", "Yes"]
)

active = st.sidebar.selectbox(
    "Physically active?",
    ["No", "Yes"]
)

# تحويل القيم
gender = 1 if gender == "Male" else 2
cholesterol = {"Normal": 1, "Above normal": 2, "High": 3}[cholesterol]
gluc = {"Normal": 1, "Above normal": 2, "High": 3}[gluc]
smoke = 1 if smoke == "Yes" else 0
alco = 1 if alco == "Yes" else 0
active = 1 if active == "Yes" else 0

# حساب الميزات الجديدة
bmi = weight / ((height / 100) ** 2)
pulse_pressure = ap_hi - ap_lo

# التنبؤ
if st.button("Predict"):

    patient = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "height": [height],
        "weight": [weight],
        "ap_hi": [ap_hi],
        "ap_lo": [ap_lo],
        "cholesterol": [cholesterol],
        "gluc": [gluc],
        "smoke": [smoke],
        "alco": [alco],
        "active": [active],
        "bmi": [bmi],
        "pulse_pressure": [pulse_pressure]
    })

    prob = model.predict_proba(patient)[0][1]

    st.subheader("Prediction Result")

    if prob < 0.3:
        st.success(f"Low Risk ({prob:.2%})")
    elif prob < 0.7:
        st.warning(f"Medium Risk ({prob:.2%})")
    else:
        st.error(f"High Risk ({prob:.2%})")

    # الرسم البياني
    fig, ax = plt.subplots()

    sns.barplot(
        x=["No Disease", "Disease"],
        y=[1 - prob, prob],
        palette=["green", "red"],
        ax=ax
    )

    ax.set_ylabel("Probability")

    st.pyplot(fig)

    # المؤشرات الصحية
    st.subheader("Health Indicators")

    st.metric("BMI", round(bmi, 2))
    st.metric("Pulse Pressure", pulse_pressure)
