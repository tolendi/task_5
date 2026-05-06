# ⛏️ Task 5 — Data Generation & Analytics Dashboard

## 📌 Overview
This project consists of two parts:
1. A Google Sheets-based data generator simulating mining output for Weyland-Yutani Corporation.
2. An interactive analytics dashboard built with Streamlit for anomaly detection and statistical analysis.

---

## ⚙️ Tech Stack
- Google Sheets (formula-based data generation)
- Python
- Streamlit
- Pandas / NumPy
- Statistical methods (z-score, IQR, Grubbs test)
- PDF report generation

---

## 🧪 Part I — Data Generator (Google Sheets)

A fully dynamic spreadsheet-based simulation system that generates realistic mining output data.

### Features:
- Configurable mine names
- Adjustable date range
- Distribution selection:
  - Uniform
  - Normal
- Dynamic distribution parameters
- Smoothing (time correlation between values)
- Day-of-week effects (e.g. Sunday -40%)
- Long-term trend component
- Event system:
  - spikes and drops
  - duration, magnitude, probability
  - bell-curve shaped impact

### Visualization:
- Auto-updating chart reflecting all parameter changes in real time

---

## 📊 Part II — Streamlit Analytics Dashboard

Interactive web dashboard built with Streamlit for exploration and anomaly detection.

### Key Metrics:
- Mean daily output
- Standard deviation
- Median
- Interquartile range (IQR)

### Anomaly Detection Methods:
- IQR rule
- Z-score method
- Distance from moving average
- Grubbs’ test

All methods support adjustable parameters.

---

### 📈 Visualization Features:
- Line / bar / stacked charts (user selectable)
- Trendlines:
  - polynomial regression (degree 1–4)
- Outliers highlighted on charts

Built using Streamlit for interactive exploration.

---

## 📄 PDF Report Generator
The dashboard includes a feature to export a full analytical report:

- Summary statistics
- Charts
- Anomaly breakdown sections (spikes & drops)
- Structured and formatted PDF output

---

## 🚀 How to Run Dashboard

```bash
streamlit run app.py
```

## 🔗 Access

### 📊 Data Source
- Google Sheets (Generator): [link](https://docs.google.com/spreadsheets/d/1O3PPHYZDVzHoa_AamKwv-4y1GRfpII4XzuRVURvK4RY/edit?usp=sharing)

### 🌐 Dashboard
- Streamlit App: [link](https://f6gnvdsrmfpc49cfzvlxy5.streamlit.app/)

### 📄 Report Output
- PDF reports are generated directly from the dashboard
