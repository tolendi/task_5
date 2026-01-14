import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import gspread
import tempfile
from datetime import datetime
from fpdf import FPDF
from scipy import stats

# --- ФУНКЦИИ ДЕТЕКЦИИ ---

def detect_grubbs(series, alpha=0.05):
    """Тест Граббса на одиночный самый значимый выброс."""
    n = len(series)
    if n < 3: return pd.Series([False]*n, index=series.index)
    t_dist = stats.t.ppf(1 - alpha / (2 * n), n - 2)
    g_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_dist**2 / (n - 2 + t_dist**2))
    z_scores = np.abs(series - series.mean()) / series.std()
    return (z_scores == z_scores.max()) & (z_scores > g_crit)

# --- PDF КЛАСС С ИСПРАВЛЕНИЕМ ENCODING ---
class WeylandPDF(FPDF):
    def header(self):
        self.set_fill_color(30, 30, 30)
        self.rect(0, 0, 210, 30, 'F')
        self.set_text_color(255, 255, 255)
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'WEYLAND-YUTANI CORP: MINING REPORT', 0, 1, 'C')
        self.ln(10)

# --- MAIN APP ---
def main():
    # Настройка стилей для устранения "плоских" графиков
    st.sidebar.title("Configuration")
    trend_deg = st.sidebar.slider("Trendline Degree", 1, 4, 1)
    chart_type = st.sidebar.selectbox("View Mode", ["Line", "Bar", "Stacked"])

    # Загрузка
    try:
        gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
        sh = gc.open_by_url("https://docs.google.com/spreadsheets/d/1QPf81XTT-WoAFquzTYeMDO27VnC_mLvoCFfUO1rQj_8/edit?usp=sharing")
        df = pd.DataFrame(sh.worksheet("Data").get_all_records())
        
        # Очистка и фикс инициализации (убираем "drop" в начале)
        val_col = 'SMOOTHED FINAL'
        df[val_col] = pd.to_numeric(df[val_col].astype(str).str.replace(',', '.'), errors='coerce')
        df = df.dropna(subset=[val_col])
        
        # Статистика (Mean, Median, Std, IQR)
        mean_v = df[val_col].mean()
        std_v = df[val_col].std()
        
        # Детекция (Z-Score, IQR, Grubbs)
        df['is_z'] = np.abs(stats.zscore(df[val_col])) > 3
        df['is_grubbs'] = detect_grubbs(df[val_col])
        df['Is_Anomaly'] = df['is_z'] | df['is_grubbs']

        # Визуализация
        fig = go.Figure()
        if chart_type == "Line":
            fig.add_trace(go.Scatter(x=df['Date'], y=df[val_col], name="Output"))
        else:
            fig.add_trace(go.Bar(x=df['Date'], y=df[val_col], name="Output"))
        
        # Трендлайн (Polynomial)
        x_idx = np.arange(len(df))
        p = np.poly1d(np.polyfit(x_idx, df[val_col], trend_deg))
        fig.add_trace(go.Scatter(x=df['Date'], y=p(x_idx), name="Trend", line=dict(dash='dash')))
        
        st.plotly_chart(fig, use_container_width=True)

        # PDF Кнопка (без Unicode символов)
        if st.button("Download PDF"):
            pdf = WeylandPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 10, f"Mean Output: {mean_v:.2f}", 0, 1)
            pdf.cell(0, 10, f"Anomalies Found: {df['Is_Anomaly'].sum()}", 0, 1)
            
            # Сохранение графика
            with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
                fig.write_image(tmp.name)
                pdf.image(tmp.name, x=10, w=180)
            
            st.download_button("Save PDF", pdf.output(dest='S').encode('latin-1'), "Report.pdf")

    except Exception as e:
        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
