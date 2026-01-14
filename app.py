import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import gspread
import tempfile
from datetime import datetime
from fpdf import FPDF

# --- КОНФИГУРАЦИЯ СТРАНИЦЫ ---
st.set_page_config(
    page_title="Weyland-Yutani Corp | Mining BI",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ФУНКЦИИ ДЕТЕКЦИИ АНОМАЛИЙ ---
def detect_iqr(series, factor=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    return (series < lower) | (series > upper), lower, upper

def detect_zscore(series, threshold=3):
    mean = series.mean()
    std = series.std()
    z_scores = (series - mean) / std
    return np.abs(z_scores) > threshold

def detect_ma(series, window=7, threshold_pct=0.2):
    ma = series.rolling(window=window, center=True).mean()
    # Заполняем пропуски (края окна) оригинальными значениями, чтобы не было ошибок
    ma_filled = ma.fillna(series)
    diff_pct = np.abs(series - ma_filled) / ma_filled
    return diff_pct > threshold_pct

# --- ЗАГРУЗКА ДАННЫХ ---
@st.cache_data(ttl=300)
def load_data():
    try:
        # АВТОРИЗАЦИЯ
        gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
        
        # ОТКРЫТИЕ ТАБЛИЦЫ (Вставьте вашу ссылку!)
        sh = gc.open_by_url("https://docs.google.com/spreadsheets/d/1QPf81XTT-WoAFquzTYeMDO27VnC_mLvoCFfUO1rQj_8/edit?usp=sharing") 
        worksheet = sh.worksheet("Data")
        
        # ЧТЕНИЕ
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        
        # ОБРАБОТКА
        # Предполагаем, что даты в колонке 'Date', а значения в 'SMOOTHED FINAL'
        # Если имена колонок отличаются, поправьте их здесь
        df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
        
        # Преобразуем числа (заменяем запятые на точки, если есть, и убираем пробелы)
        col_name = 'SMOOTHED FINAL' # Имя колонки из Google Sheet
        df[col_name] = df[col_name].astype(str).str.replace(',', '.', regex=False).str.replace(r'[^\d\.]', '', regex=True)
        df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        
        df = df.dropna(subset=['Date', col_name])
        return df.sort_values('Date')
    except Exception as e:
        st.error(f"🔴 Критическая ошибка загрузки: {e}")
        return pd.DataFrame()

# --- КЛАСС ДЛЯ PDF ОТЧЕТА ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Weyland-Yutani Corp: Daily Extraction Report', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
        self.ln(5)

    def chapter_title(self, label):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 6, f"  {label}", 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, text):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, text)
        self.ln()

# --- ОСНОВНОЙ UI ---
def main():
    # Sidebar
    st.sidebar.title("⚙️ Config Panel")
    st.sidebar.subheader("Anomaly Detection Parameters")
    
    iqr_factor = st.sidebar.slider("IQR Factor", 1.0, 3.0, 1.5, 0.1)
    z_thresh = st.sidebar.slider("Z-Score Threshold", 2.0, 5.0, 3.0, 0.1)
    ma_window = st.sidebar.slider("Moving Avg Window", 3, 30, 7)
    ma_thresh = st.sidebar.slider("MA % Deviation", 0.05, 0.50, 0.20, 0.01)
    
    st.sidebar.markdown("---")
    st.sidebar.info("Data source: Google Sheets (Live)")

    # Load Data
    df = load_data()
    
    if df.empty:
        st.warning("Нет данных для отображения. Проверьте подключение и структуру таблицы.")
        return

    # Подготовка данных
    value_col = 'SMOOTHED FINAL'
    series = df[value_col]
    
    # 1. KPI SECTION
    st.title("🏭 Mining Operations Dashboard")
    
    stats_cols = st.columns(4)
    curr_mean = series.mean()
    curr_std = series.std()
    curr_med = series.median()
    curr_iqr = series.quantile(0.75) - series.quantile(0.25)
    
    stats_cols[0].metric("Mean Output", f"{curr_mean:,.0f} t")
    stats_cols[1].metric("Std Deviation", f"{curr_std:,.0f} t")
    stats_cols[2].metric("Median", f"{curr_med:,.0f} t")
    stats_cols[3].metric("IQR", f"{curr_iqr:,.0f} t")
    
    st.markdown("---")

    # 2. ANOMALY CALCULATIONS
    mask_iqr, _, _ = detect_iqr(series, iqr_factor)
    mask_z = detect_zscore(series, z_thresh)
    mask_ma = detect_ma(series, ma_window, ma_thresh)
    
    # Объединяем все аномалии (OR logic)
    df['Is_Anomaly'] = mask_iqr | mask_z | mask_ma
    
    anomalies_count = df['Is_Anomaly'].sum()
    st.subheader(f"📉 Trend Analysis & Anomalies ({anomalies_count} detected)")
    
    # 3. CHART CONTROLS
    c1, c2 = st.columns([1, 4])
    with c1:
        trend_degree = st.selectbox("Trendline Degree", [1, 2, 3, 4], index=0)
        show_anomalies = st.checkbox("Highlight Anomalies", value=True)
    
    # 4. PLOTTING
    # Тренд
    x_nums = np.arange(len(df))
    coeffs = np.polyfit(x_nums, df[value_col], trend_degree)
    poly_func = np.poly1d(coeffs)
    df['Trend'] = poly_func(x_nums)
    
    fig = go.Figure()
    
    # Основная линия
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df[value_col], 
        mode='lines', name='Output', 
        line=dict(color='#2E86C1', width=2)
    ))
    
    # Тренд
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Trend'], 
        mode='lines', name=f'Trend (Deg {trend_degree})',
        line=dict(color='orange', width=2, dash='dash')
    ))
    
    # Аномалии
    if show_anomalies and anomalies_count > 0:
        anom_df = df[df['Is_Anomaly']]
        fig.add_trace(go.Scatter(
            x=anom_df['Date'], y=anom_df[value_col],
            mode='markers', name='Anomalies',
            marker=dict(color='red', size=8, symbol='x')
        ))

    fig.update_layout(
        title="Daily Extraction Output vs Trend",
        xaxis_title="Date",
        yaxis_title="Tons",
        template="plotly_white",
        height=500,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    st.plotly_chart(fig, width="stretch")
    
    # 5. DETAILED ANOMALY TABLE
    with st.expander("🔍 Detailed Anomaly List", expanded=False):
        if anomalies_count > 0:
            # Подготовка красивой таблицы
            detail_df = df[df['Is_Anomaly']].copy()
            detail_df['Reason'] = ""
            
            # Векторизированное добавление причин (упрощенно через apply)
            def get_reasons(idx):
                reasons = []
                if mask_iqr[idx]: reasons.append("IQR Rule")
                if mask_z[idx]: reasons.append("Z-Score")
                if mask_ma[idx]: reasons.append(f"MA Dist > {int(ma_thresh*100)}%")
                return ", ".join(reasons)
                
            detail_df['Reason'] = detail_df.index.to_series().apply(get_reasons)
            
            st.dataframe(
                detail_df[['Date', value_col, 'Reason']].style.format({value_col: "{:.1f}"}),
                width="stretch"
            )
        else:
            st.info("No anomalies detected with current parameters.")

    # 6. PDF GENERATION
    st.markdown("---")
    if st.button("📄 Download PDF Report"):
        with st.spinner("Generating PDF... (this may take a moment)"):
            try:
                # 1. Сохраняем график во временный файл
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                    fig.write_image(tmpfile.name, width=1200, height=600, scale=2)
                    tmp_img_path = tmpfile.name
                
                # 2. Создаем PDF
                pdf = PDFReport()
                pdf.add_page()
                
                # Статистика
                pdf.chapter_title("1. Performance Summary")
                pdf.set_font('Arial', '', 11)
                pdf.cell(50, 10, f"Mean: {curr_mean:.1f}", 1)
                pdf.cell(50, 10, f"Median: {curr_med:.1f}", 1)
                pdf.cell(50, 10, f"Std Dev: {curr_std:.1f}", 1)
                pdf.ln(15)
                
                # График
                pdf.chapter_title("2. Visual Analysis")
                pdf.image(tmp_img_path, x=10, w=190)
                pdf.ln(5)
                
                # Аномалии
                pdf.add_page()
                pdf.chapter_title("3. Detected Anomalies")
                
                if anomalies_count > 0:
                    pdf.set_font('Arial', 'B', 10)
                    # Заголовки таблицы
                    pdf.cell(40, 8, "Date", 1)
                    pdf.cell(40, 8, "Value", 1)
                    pdf.cell(100, 8, "Detection Method", 1)
                    pdf.ln()
                    
                    pdf.set_font('Arial', '', 10)
                    for idx, row in detail_df.iterrows():
                        date_str = row['Date'].strftime('%Y-%m-%d')
                        val_str = f"{row[value_col]:.1f}"
                        reason_str = row['Reason']
                        
                        pdf.cell(40, 8, date_str, 1)
                        pdf.cell(40, 8, val_str, 1)
                        pdf.cell(100, 8, reason_str, 1)
                        pdf.ln()
                else:
                    pdf.chapter_body("No anomalies detected within the selected parameters.")

                # 3. Вывод
                pdf_bytes = bytes(pdf.output(dest="S"))
                
                st.download_button(
                    label="💾 Save Report",
                    data=pdf_bytes,
                    file_name="weyland_report.pdf",
                    mime="application/pdf"
                )
                
            except Exception as e:
                st.error(f"Error generating PDF: {e}")
                st.info("Ensure 'kaleido' is installed: pip install kaleido")

if __name__ == "__main__":

    main()


