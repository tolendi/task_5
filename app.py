import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from io import BytesIO

# --- 1. ЗАГРУЗКА ДАННЫХ ---
@st.cache_data(ttl=600)
def load_data(url):
    # Принудительно конвертируем ссылку в формат экспорта CSV
    if "edit?usp=sharing" in url:
        url = url.replace("edit?usp=sharing", "export?format=csv&gid=0")
    
    df = pd.read_csv(url, decimal=',')
    
    # Проверка наличия колонки
    if 'SMOOTHED FINAL' not in df.columns:
        st.error(f"Колонка 'SMOOTHED FINAL' не найдена. Доступные колонки: {df.columns.tolist()}")
        st.stop()
        
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
        std_dev = np.abs(series - series.mean())
        return std_dev > (param * series.std())

# --- 3. ИНТЕРФЕЙС ---
st.set_page_config(layout="wide", page_title="Weyland-Yutani Mining BI")
st.title("🛰️ Weyland-Yutani | Mining Operations Dashboard")

# Прямая вставка ссылки (чтобы избежать KeyError в secrets)
SHEET_URL = "https://docs.google.com/spreadsheets/d/1O3PPHYZDVzHoa_AamKwv-4y1GRfpII4XzuRVURvK4RY/export?format=csv&gid=1541532661"

try:
    df = load_data(SHEET_URL)
    val_col = 'SMOOTHED FINAL'

    # --- САЙДБАР ---
    st.sidebar.header("Control Panel")
    chart_type = st.sidebar.selectbox("Chart Type", ["Line", "Bar", "Stacked Area"])
    poly_deg = st.sidebar.slider("Trendline Polynomial Degree", 1, 4, 1)
    
    st.sidebar.subheader("Anomaly Detection")
    test_method = st.sidebar.selectbox("Test Method", ["IQR Rule", "Z-Score", "Moving Average Dist", "Grubbs Test"])
    test_param = st.sidebar.number_input("Sensitivity Factor", value=1.5 if test_method=="IQR Rule" else 3.0)

    # --- СТАТИСТИКА (KPI) ---
    mean_val = df[val_col].mean()
    std_val = df[val_col].std()
    med_val = df[val_col].median()
    iqr_val = df[val_col].quantile(0.75) - df[val_col].quantile(0.25)

    st.subheader("Mine Performance Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Output", f"{mean_val:.2f}")
    c2.metric("Std Deviation", f"{std_val:.2f}")
    c3.metric("Median", f"{med_val:.2f}")
    c4.metric("IQR", f"{iqr_val:.2f}")

    # --- ТРЕНД И АНОМАЛИИ ---
    df['is_anomaly'] = detect_anomalies(df, test_method, test_param)
    
    X = np.array(range(len(df))).reshape(-1, 1)
    y = df[val_col].values
    poly = PolynomialFeatures(degree=poly_deg)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    df['trend'] = model.predict(X_poly)

    # --- ГРАФИК ---
    fig = go.Figure()
    if chart_type == "Line":
        fig.add_trace(go.Scatter(x=df['Date'], y=df[val_col], name="Output", line=dict(color='#00d4ff', width=2)))
    elif chart_type == "Bar":
        fig.add_trace(go.Bar(x=df['Date'], y=df[val_col], name="Output", marker_color='#00d4ff'))
    else:
        fig.add_trace(go.Scatter(x=df['Date'], y=df[val_col], name="Output", fill='tozeroy', line=dict(color='#00d4ff')))

    fig.add_trace(go.Scatter(x=df['Date'], y=df['trend'], name=f"Trend (Poly {poly_deg})", line=dict(color='yellow', dash='dot')))
    
    anoms = df[df['is_anomaly']]
    fig.add_trace(go.Scatter(x=anoms['Date'], y=anoms[val_col], mode='markers', name="🚨 Anomaly", 
                             marker=dict(color='red', size=10, symbol='diamond')))

    st.plotly_chart(fig, use_container_width=True)

    # --- PDF ОТЧЕТ (Упрощенная генерация через CSV/Table) ---
    st.divider()
    if st.button("Generate Detailed Report"):
        report_df = df[df['is_anomaly']][['Date', val_col]]
        st.subheader("Anomaly Log")
        st.table(report_df)
        
        # Скачивание лога как CSV (самый быстрый аналог PDF для анализа)
        csv = report_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Log as CSV", csv, "anomaly_report.csv", "text/csv")

except Exception as e:
    st.error(f"Critical System Error: {e}")

