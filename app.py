import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def load_and_process_data(sheet_url):
    # Читаем лист Data
    df = pd.read_csv(sheet_url)
    
    # 1. Приводим форматы в порядок
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    val_col = 'SMOOTHED FINAL'
    
    # 2. Умная детекция аномалий
    # Считаем среднее и отклонение для каждого дня недели отдельно
    # Это нужно, чтобы субботнее снижение не считалось аномалией
    df['day_of_week'] = df['Date'].dt.dayofweek
    
    df['is_anomaly'] = False
    for day in range(7):
        day_data = df[df['day_of_week'] == day][val_col]
        mean = day_data.mean()
        std = day_data.std()
        # Если значение отклоняется более чем на 3 сигмы от среднего ДЛЯ ЭТОГО ДНЯ
        df.loc[df['day_of_week'] == day, 'is_anomaly'] = np.abs(df[val_col] - mean) > (3 * std)
    
    return df

# --- ИНТЕРФЕЙС ---
st.title("🛰️ Weyland-Yutani | Mining Operations Center")

# Ссылка на вашу таблицу (экспорт в CSV)
SHEET_ID = "1O3PPHYZDVzHoa_AamKwv-4y1GRfpII4XzuRVURvK4RY"
DATA_GID = "1541532661" # Обычно 0 для первого листа, или число из ссылки gid=...
csv_url = f"https://docs.google.com/spreadsheets/d/e/2PACX-1vQwLRedMgwJUgBxq-349qrMcbrOA4oKtpnSc5YoVa3KaBaaB67MUZTeL5yvY-PKgn2pn3rSjSb2fbtX/pub?gid=1541532661&single=true&output=csv"

try:
    df = load_and_process_data(csv_url)
    
    # График
    fig = go.Figure()
    # Основная линия
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMOOTHED FINAL'], name="Production Output", line=dict(color='#00d4ff')))
    
    # Аномалии (только те точки, где True)
    anoms = df[df['is_anomaly']]
    fig.add_trace(go.Scatter(x=anoms['Date'], y=anoms['SMOOTHED FINAL'], 
                             mode='markers', name="🚨 System Alert", 
                             marker=dict(color='red', size=10, symbol='circle-open')))
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.success("Data Feed: Active. All sensors operational.")

except Exception as e:
    st.info("Please connect the Google Sheets data source to begin analysis.")

