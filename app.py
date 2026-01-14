import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# --- 1. ЗАГРУЗКА ДАННЫХ ---
@st.cache_data(ttl=600)
def load_data(url):
    df = pd.read_csv(url, decimal=',')
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.dropna(subset=['SMOOTHED FINAL'])
    return df

# --- 2. СТАТИСТИЧЕСКИЕ ТЕСТЫ ---
def detect_anomalies(data, method, param):
    series = data['SMOOTHED FINAL']
    if method == "IQR Rule":
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        return (series < (Q1 - param * IQR)) | (series > (Q3 + param * IQR))
    
    elif method == "Z-Score":
        z = np.abs(stats.zscore(series))
        return z > param
    
    elif method == "Moving Average Dist":
        ma = series.rolling(window=7, center=True).mean()
        dist = np.abs(series - ma) / ma
        return dist > (param / 100)
    
    elif method == "Grubbs Test":
        # Упрощенная реализация через отклонение от среднего
        std_dev = np.abs(series - series.mean())
        return std_dev > (param * series.std())

# --- 3. ИНТЕРФЕЙС ---
st.set_page_config(layout="wide", page_title="Mining BI")
st.title("🛰️ Weyland-Yutani Operations Dashboard")

# Ссылка из ваших Secrets
SHEET_URL = st.secrets["gsheets_url"]

try:
    df = load_data(SHEET_URL)
    val_col = 'SMOOTHED FINAL'

    # --- САЙДБАР (Настройки) ---
    st.sidebar.header("Analysis Settings")
    chart_type = st.sidebar.selectbox("Chart Type", ["Line", "Bar", "Stacked Area"])
    poly_deg = st.sidebar.slider("Trendline Polynomial Degree", 1, 4, 1)
    
    test_method = st.sidebar.selectbox("Anomaly Test", ["IQR Rule", "Z-Score", "Moving Average Dist", "Grubbs Test"])
    test_param = st.sidebar.number_input("Test Sensitivity (Sigma/Factor)", value=1.5 if test_method=="IQR Rule" else 3.0)

    # --- СТАТИСТИКА (KPI) ---
    st.subheader("Production Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Mean Daily", round(df[val_col].mean(), 2))
    with col2: st.metric("Std Dev", round(df[val_col].std(), 2))
    with col3: st.metric("Median", round(df[val_col].median(), 2))
    with col4: st.metric("IQR", round(df[val_col].quantile(0.75) - df[val_col].quantile(0.25), 2))

    # --- РАСЧЕТ ТРЕНДА И АНОМАЛИЙ ---
    df['is_anomaly'] = detect_anomalies(df, test_method, test_param)
    
    # Регрессия для тренда
    X = np.array(range(len(df))).reshape(-1, 1)
    y = df[val_col].values
    poly = PolynomialFeatures(degree=poly_deg)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    df['trend'] = model.predict(X_poly)

    # --- ГРАФИК ---
    fig = go.Figure()
    
    # Выбор типа графика
    if chart_type == "Line":
        fig.add_trace(go.Scatter(x=df['Date'], y=df[val_col], name="Production", line=dict(color='#00d4ff')))
    elif chart_type == "Bar":
        fig.add_trace(go.Bar(x=df['Date'], y=df[val_col], name="Production", marker_color='#00d4ff'))
    else:
        fig.add_trace(go.Scatter(x=df['Date'], y=df[val_col], name="Production", fill='tozeroy', line=dict(color='#00d4ff')))

    # Тренд
    fig.add_trace(go.Scatter(x=df['Date'], y=df['trend'], name="Trendline", line=dict(color='yellow', dash='dash')))

    # Аномалии
    anoms = df[df['is_anomaly']]
    fig.add_trace(go.Scatter(x=anoms['Date'], y=anoms[val_col], mode='markers', 
                             name="Anomaly", marker=dict(color='red', size=12, symbol='x')))

    st.plotly_chart(fig, use_container_width=True)

    # --- ОТЧЕТ ---
    if st.button("Generate PDF Report"):
        st.warning("PDF Generation requires 'fpdf' or 'reportlab' library. Add to requirements.txt.")
        # Тут логика формирования PDF (пока заглушка)
        st.info("Section for Anomaly Details: Detected " + str(len(anoms)) + " events.")

except Exception as e:
    st.error(f"Error: {e}")

