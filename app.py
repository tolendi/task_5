import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def load_and_process_data(sheet_url):
    # Читаем CSV, указывая, что десятичный разделитель — запятая
    # Это превратит '3399,96' в число 3399.96
    df = pd.read_csv(sheet_url, decimal=',')
    
    # Удаляем пустые строки, если они есть
    df = df.dropna(subset=['Date', 'SMOOTHED FINAL'])
    
    # Приводим даты в порядок
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    val_col = 'SMOOTHED FINAL'
    
    # Принудительно конвертируем колонку в числа (на случай, если закрался текст)
    df[val_col] = pd.to_numeric(df[val_col], errors='coerce')
    
    # 2. Умная детекция аномалий
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['is_anomaly'] = False
    
    for day in range(7):
        day_mask = df['day_of_week'] == day
        day_data = df.loc[day_mask, val_col]
        
        if len(day_data) > 0:
            mean = day_data.mean()
            std = day_data.std()
            # Защита от деления на ноль, если std = 0
            if std > 0:
                anomalies = np.abs(df.loc[day_mask, val_col] - mean) > (3 * std)
                df.loc[day_mask, 'is_anomaly'] = anomalies
    
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
    st.error(f"❌ Ошибка подключения:")
    st.write(e) # Это покажет технический текст ошибки
    st.info(f"Проверьте ссылку. Сейчас код использует: {csv_url}")


