import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from fpdf import FPDF
import tempfile

# --- ЗАГРУЗКА ДАННЫХ ---
@st.cache_data(ttl=600)
def load_data(url):
    # Авто-замена на формат CSV
    if "edit?usp=sharing" in url:
        url = url.replace("edit?usp=sharing", "export?format=csv&gid=1054366367") # ЗАМЕНИТЕ GID НА ВАШ
    df = pd.read_csv(url, decimal=',')
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.dropna(subset=['SMOOTHED FINAL'])
    return df

# --- ТЕСТЫ АНОМАЛИЙ ---
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

# --- ФУНКЦИЯ ГЕНЕРАЦИИ PDF ---
def create_pdf(df, stats_dict, method_name):
    pdf = FPDF()
    pdf.add_page()
    
    # Заголовок в стиле Weyland-Yutani
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "WEYLAND-YUTANI | MINING OPERATIONS REPORT", ln=True, align='C')
    pdf.ln(10)
    
    # Секция статистики
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "1. Executive Summary (Statistics):", ln=True)
    pdf.set_font("Arial", "", 10)
    for key, val in stats_dict.items():
        pdf.cell(200, 7, f"- {key}: {val:.2f}", ln=True)
    
    pdf.ln(5)
    
    # Секция аномалий
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, f"2. Anomaly Detection (Method: {method_name}):", ln=True)
    
    anoms = df[df['is_anomaly']]
    if not anoms.empty:
        pdf.set_font("Arial", "B", 10)
        pdf.cell(60, 10, "Date", border=1)
        pdf.cell(60, 10, "Output Value", border=1)
        pdf.ln()
        pdf.set_font("Arial", "", 10)
        for i, row in anoms.iterrows():
            pdf.cell(60, 10, str(row['Date'].date()), border=1)
            pdf.cell(60, 10, str(round(row['SMOOTHED FINAL'], 2)), border=1)
            pdf.ln()
    else:
        pdf.cell(200, 10, "No anomalies detected in this period.", ln=True)

    return pdf.output()

# --- ИНТЕРФЕЙС ---
st.title("Weyland-Yutani | Operations Center")

# !!! ОБЯЗАТЕЛЬНО ПРОВЕРЬТЕ GID ТУТ !!!
SHEET_URL = "https://docs.google.com/spreadsheets/d/1O3PPHYZDVzHoa_AamKwv-4y1GRfpII4XzuRVURvK4RY/export?format=csv&gid=1541532661"

try:
    df = load_data(SHEET_URL)
    val_col = 'SMOOTHED FINAL'

    # Сайдбар
    st.sidebar.header("Analysis Parameters")
    chart_type = st.sidebar.selectbox("Chart Type", ["Line", "Bar"])
    poly_deg = st.sidebar.slider("Trend Degree", 1, 4, 1)
    test_method = st.sidebar.selectbox("Test", ["IQR Rule", "Z-Score", "Moving Average Dist", "Grubbs Test"])
    test_param = st.sidebar.number_input("Threshold", value=1.5 if test_method=="IQR Rule" else 3.0)

    # Расчеты
    df['is_anomaly'] = detect_anomalies(df, test_method, test_param)
    
    # KPI для отчета
    stats_dict = {
        "Mean Daily Output": df[val_col].mean(),
        "Standard Deviation": df[val_col].std(),
        "Median": df[val_col].median(),
        "Interquartile Range (IQR)": df[val_col].quantile(0.75) - df[val_col].quantile(0.25)
    }

    # Вывод метрик на экран
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean", round(stats_dict["Mean Daily Output"], 2))
    c2.metric("Std Dev", round(stats_dict["Standard Deviation"], 2))
    c3.metric("Median", round(stats_dict["Median"], 2))
    c4.metric("IQR", round(stats_dict["Interquartile Range (IQR)"], 2))

    # График (Trendline)
    X = np.array(range(len(df))).reshape(-1, 1)
    poly = PolynomialFeatures(degree=poly_deg)
    model = LinearRegression().fit(poly.fit_transform(X), df[val_col])
    df['trend'] = model.predict(poly.fit_transform(X))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df[val_col], name="Output", line=dict(color='#00d4ff')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['trend'], name="Trend", line=dict(color='yellow', dash='dot')))
    
    anoms = df[df['is_anomaly']]
    fig.add_trace(go.Scatter(x=anoms['Date'], y=anoms[val_col], mode='markers', name="🚨 Alert", marker=dict(color='red', size=10)))
    
    st.plotly_chart(fig, use_container_width=True)

    # --- КНОПКА PDF ---
    st.divider()
    if st.button("🛠️ Generate Detailed PDF Report"):
        pdf_data = create_pdf(df, stats_dict, test_method)
        st.download_button(
            label="💾 Download PDF Report",
            data=pdf_data,
            file_name="Weyland_Yutani_Report.pdf",
            mime="application/pdf"
        )
        st.success("Report generated successfully.")

except Exception as e:
    st.error(f"Waiting for Data Feed... (Error: {e})")
