import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import xgboost as xgb 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Загрузка моделей
@st.cache_resource
def load_models():
    xgb = pickle.load(open('notebooks/models/xgb_v2.pkl', 'rb'))
    scaler = pickle.load(open('notebooks/models/scaler.pkl', 'rb'))
    tokenizer = pickle.load(open('notebooks/models/tokenizer.pkl', 'rb'))
    text_cnn = load_model('notebooks/models/text_cnn.h5')
    return xgb, scaler, tokenizer, text_cnn 

xgb_model, scaler_model, text_tokenizer, text_cnn = load_models()

st.title("🎯 Churn Prediction Dashboard")
st.markdown("### Предсказание оттока клиентов + анализ сентимента отзывов")
st.write("hello")


# Sidebar для ввода
st.sidebar.header("📊 Введите данные клиента")
gender = st.sidebar.selectbox("Пол", ['Male', 'Female'])
senior = st.sidebar.selectbox("Senior Citizen", ['0', '1'])
partner = st.sidebar.selectbox("Partner", ['Yes', 'No'])
dependents = st.sidebar.selectbox("Dependents", ['Yes', 'No'])
tenure = st.sidebar.slider("Tenure (месяцев)", 0, 72, 12)
phone = st.sidebar.selectbox("PhoneService", ['Yes', 'No'])
internet = st.sidebar.selectbox("InternetService", ['DSL', 'Fiber optic', 'No'])
contract = st.sidebar.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
monthly = st.sidebar.slider("Monthly Charges ($)", 18.0, 118.0, 50.0)
review = st.sidebar.text_area("Отзыв клиента", "хороший сервис")
multiple = st.sidebar.selectbox("Multiple Lines", ['No', 'Yes', 'No phone service'])
backup = st.sidebar.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
protection = st.sidebar.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
support = st.sidebar.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
payment = st.sidebar.selectbox("Payment Method", [
    'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 
    'Credit card (automatic)'
])
online_security = st.sidebar.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
paperless = st.sidebar.selectbox("Paperless Billing", ['Yes', 'No'])
streaming_tv = st.sidebar.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])
total_charges = st.sidebar.slider("Total Charges ($)", 0.0, 8684.0, 1000.0)

if st.sidebar.button("🔮 Предсказать отток", type="primary"):
    # Сентимент (вычисляем отдельно)
    review_seq = text_tokenizer.texts_to_sequences([review])
    review_pad = pad_sequences(review_seq, maxlen=20)
    sentiment = text_cnn.predict(review_pad)[0][0]
    
    
    input_data = pd.DataFrame([{
        'gender_Male': 1 if gender == 'Male' else 0,
        'SeniorCitizen': int(senior),
        'Partner_Yes': 1 if partner == 'Yes' else 0,
        'Dependents_Yes': 1 if dependents == 'Yes' else 0,
        'tenure': tenure,
        'PhoneService_Yes': 1 if phone == 'Yes' else 0,
        'MultipleLines_No phone service': 1 if multiple == 'No phone service' else 0,
        'MultipleLines_Yes': 1 if multiple == 'Yes' else 0,
        'InternetService_Fiber optic': 1 if internet == 'Fiber optic' else 0,
        'InternetService_No': 1 if internet == 'No' else 0,
        'OnlineBackup_No internet service': 1 if backup == 'No internet service' else 0,
        'OnlineBackup_Yes': 1 if backup == 'Yes' else 0,
        'DeviceProtection_No internet service': 1 if protection == 'No internet service' else 0,
        'DeviceProtection_Yes': 1 if protection == 'Yes' else 0,
        'TechSupport_No internet service': 1 if support == 'No internet service' else 0,
        'TechSupport_Yes': 1 if support == 'Yes' else 0,
        'OnlineSecurity_No internet service': 1 if online_security == 'No internet service' else 0,
        'OnlineSecurity_Yes': 1 if online_security == 'Yes' else 0,
        'PaperlessBilling_Yes': 1 if paperless == 'Yes' else 0,
        'Contract_One year': 1 if contract == 'One year' else 0,
        'Contract_Two year': 1 if contract == 'Two year' else 0,
        'PaymentMethod_Electronic check': 1 if payment == 'Electronic check' else 0,
        'PaymentMethod_Mailed check': 1 if payment == 'Mailed check' else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment == 'Credit card (automatic)' else 0,
        'MonthlyCharges': monthly,
        'TotalCharges': total_charges,
        'StreamingTV_No internet service': 1 if streaming_tv == 'No internet service' else 0,
        'StreamingTV_Yes': 1 if streaming_tv == 'Yes' else 0,
        'StreamingMovies_No internet service': 1 if streaming_movies == 'No internet service' else 0,
        'StreamingMovies_Yes': 1 if streaming_movies == 'Yes' else 0
    }])

    booster = xgb_model.get_booster()
    dmatrix = xgb.DMatrix(input_data)
    prob = booster.predict(dmatrix)[0]
    
    # input_scaled = scaler_model.transform(input_data)
    # prob = xgb_model.predict_proba(input_scaled)[0][0]
    
    
    st.metric("🎲 Вероятность оттока", f"{prob:.1%}")
    st.success("✅ Низкий риск" if prob < 0.5 else "⚠️ Высокий риск")
    st.info(f"💭 Сентимент: {sentiment:.2f}")



# Загрузка данных для графиков
df_test = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
col1, col2 = st.columns(2)
col1.metric("👥 Всего клиентов", len(df_test))
col1.metric("📉 % оттока", f"{df_test['Churn'].value_counts(normalize=True)['Yes']:.1%}")

# Feature importance
st.subheader("📈 Топ факторов оттока")
importances_df = pd.read_csv('notebooks/feature_importance.csv')  # сохрани ранее
fig = px.bar(importances_df.head(10), x='importance', y='feature', orientation='h')
st.plotly_chart(fig)