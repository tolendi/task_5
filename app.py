import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import gspread
import tempfile
from datetime import datetime
from fpdf import FPDF
from scipy import stats

# --- КОНФИГУРАЦИЯ СТРАНИЦЫ ---
st.set_page_config(
    page_title="Weyland-Yutani Corp | Mining BI",
    page_icon="⛏️",
    layout="wide"
)

# --- ФУНКЦИИ АНАЛИЗА ---

def detect_iqr(series, factor=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return (series < (Q1 - factor * IQR)) | (series > (Q3 + factor * IQR))

def detect_zscore(series, threshold=3):
    return np.abs(stats.zscore(series)) > threshold

def detect_ma(series, window=7, threshold_pct=0.2):
    ma = series.rolling(window=window, center=True).mean().fillna(series)
    return (np.abs(series - ma) / ma) > threshold_pct

def detect_grubbs(series, alpha=0.05):
    n = len(series)
    if n < 3: return pd.Series([False]*n, index=series.index)
    t_dist = stats.t.ppf(1 - alpha / (2 * n), n - 2)
    g_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_dist**2 / (n - 2 + t_dist**2))
    z_scores = np.abs(series - series.mean()) / series.std()
    is_grubbs = (z_scores == z_scores.max()) & (z_scores > g_crit)
    return is_grubbs

# --- ЗАГРУЗКА ДАННЫХ ---
@st.cache_data(ttl=300)
def load_data():
    try:
        gc = gspread.service_account_from_dict(st.secrets["gcp_service_account"])
        sh = gc.open_by_url("https://docs.google.com/spreadsheets/d/1QPf81XTT-WoAFquzTYeMDO27VnC_mLvoCFfUO1rQj_8/edit?usp=sharing") 
        df = pd.DataFrame(sh.worksheet("Data").get_all_records())
        
        df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
        val_col = 'SMOOTHED FINAL'
        df[val_col] = pd.to_numeric(df[val_col].astype(str).str.replace(',', '.'), errors='coerce')
        return df.dropna(subset=['Date', val_col]).sort_values('Date')
    except Exception as e:
        st.error(f"🔴 Connection Error: {e}")
        return pd.DataFrame()

# --- PDF REPORT CLASS ---
class WeylandPDF(FPDF):
    def header(self):
        self.set_fill_color(30, 30, 30)
        self.rect(0, 0, 210, 40, 'F')
        self.set_text_color(255, 255, 255)
        self.set_font('Arial', 'B', 20)
        self.cell(0, 20, 'WEYLAND-YUTANI CORP', 0, 1, 'C')
        self.set_font('Arial', '', 10)
        self.cell(0, 5, 'INTERNAL ANALYTICS DIVISION - MINING REPORT', 0, 1, 'C')
        self.ln(15)

    def section_header(self, title):
        self.set_text_color(0, 0, 0)
        self.set_font('Arial', 'B', 14)
        self.set_draw_color(200, 150, 0)
        self.cell(0, 10, title, 'B', 1, 'L')
        self.ln(5)

# --- MAIN APP ---
def main():
    df = load_data()
    if df.empty: return

    val_col = 'SMOOTHED FINAL'
    
    # Sidebar
    st.sidebar.title("🛠️ Analysis Control")
    chart_type = st.sidebar.selectbox("Chart Style", ["Line", "Bar", "Stacked"])
    trend_deg = st.sidebar.slider("Polynomial Trend Degree", 1, 4, 1)
    
    st.sidebar.subheader("Anomaly Sensitivity")
    z_sens = st.sidebar.slider("Z-Score Threshold", 2.0, 5.0, 3.0)
    iqr_sens = st.sidebar.slider("IQR Multiplier", 1.0, 3.0, 1.5)
    ma_sens = st.sidebar.slider("MA Deviation %", 0.05, 0.5, 0.2)

    # 1. Stats Calculation
    stats_data = {
        "Mean": df[val_col].mean(),
        "Std Dev": df[val_col].std(),
        "Median": df[val_col].median(),
        "IQR": df[val_col].quantile(0.75) - df[val_col].quantile(0.25)
    }

    st.title("⛏️ Extraction Dashboard")
    cols = st.columns(4)
    for i, (k, v) in enumerate(stats_data.items()):
        cols[i].metric(k, f"{v:,.1f}")

    # 2. Anomaly Detection Logic
    m_iqr = detect_iqr(df[val_col], iqr_sens)
    m_z = detect_zscore(df[val_col], z_sens)
    m_ma = detect_ma(df[val_col], threshold_pct=ma_sens)
    m_grubbs = detect_grubbs(df[val_col])
    
    df['Is_Anomaly'] = m_iqr | m_z | m_ma | m_grubbs
    
    # 3. Plotting
    x_range = np.arange(len(df))
    poly = np.poly1d(np.polyfit(x_range, df[val_col], trend_deg))
    df['Trend'] = poly(x_range)

    fig = go.Figure()
    if chart_type == "Bar" or chart_type == "Stacked":
        fig.add_trace(go.Bar(x=df['Date'], y=df[val_col], name="Output", marker_color='#2E86C1'))
        if chart_type == "Stacked": fig.update_layout(barmode='stack')
    else:
        fig.add_trace(go.Scatter(x=df['Date'], y=df[val_col], name="Output", line=dict(color='#2E86C1')))

    fig.add_trace(go.Scatter(x=df['Date'], y=df['Trend'], name="Trendline", line=dict(color='orange', dash='dash')))
    
    anoms = df[df['Is_Anomaly']]
    fig.add_trace(go.Scatter(x=anoms['Date'], y=anoms[val_col], mode='markers', name="Outliers", marker=dict(color='red', size=10, symbol='x')))
    
    st.plotly_chart(fig, use_container_width=True)

    # 4. Detailed List & PDF
    with st.expander("📝 Anomaly Breakdown"):
        anom_list = []
        for idx, row in anoms.iterrows():
            reasons = []
            if m_iqr[idx]: reasons.append("IQR")
            if m_z[idx]: reasons.append("Z-Score")
            if m_ma[idx]: reasons.append("MA")
            if m_grubbs[idx]: reasons.append("Grubbs")
            type_anom = "Spike 🚀" if row[val_col] > stats_data["Mean"] else "Drop 🔻"
            anom_list.append({"Date": row['Date'].date(), "Value": row[val_col], "Type": type_anom, "Methods": ", ".join(reasons)})
        
        anom_report_df = pd.DataFrame(anom_list)
        st.table(anom_report_df)

    if st.button("📊 Generate Executive PDF Report"):
        pdf = WeylandPDF()
        pdf.add_page()
        
        pdf.section_header("1. Operational Statistics")
        for k, v in stats_data.items():
            pdf.cell(0, 8, f"- {k}: {v:,.2f}", 0, 1)
        
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            fig.write_image(tmp.name)
            pdf.ln(5)
            pdf.section_header("2. Trend Visualization")
            pdf.image(tmp.name, x=10, w=190)
        
        pdf.add_page()
        pdf.section_header("3. Anomaly Intelligence Report")
        pdf.set_font('Arial', '', 10)
        for item in anom_list:
            pdf.multi_cell(0, 10, f"EVENT ID-{np.random.randint(100,999)}: On {item['Date']}, a {item['Type']} was detected. Value: {item['Value']}. Identified via: {item['Methods']}.", border='B')
            
        st.download_button("Download Report", pdf.output(dest='S').encode('latin-1'), "W_Y_Mining_Report.pdf", "application/pdf")

if __name__ == "__main__":
    main()
