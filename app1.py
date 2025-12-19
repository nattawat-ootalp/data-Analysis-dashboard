import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from model_logic import train_waste_model

# 1. ตั้งค่า Page Configuration
st.set_page_config(page_title="Eco-Smart Waste Dashboard", layout="wide")

# 2. ใส่ CSS แบบจัดเต็มเพื่อพื้นหลังเรืองแสงและลูกเล่น Modern UI
st.markdown("""
    <style>
    /* พื้นหลังแบบเรืองแสงไล่เฉดสี */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        background-attachment: fixed;

    /* ตกแต่ง Metric Card ให้ดูเหมือนกระจก (Glassmorphism) */
    [data-testid="stMetricValue"] {
        color: #00ffcc !important;
        font-size: 2.5rem !important;
        text-shadow: 0 0 10px rgba(0, 255, 204, 0.5);
    }
    
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* ปรับแต่ง Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px 10px 0px 0px;
        color: white;
        padding: 10px 20px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #00ffcc !important;
        color: #0f172a !important;
        font-weight: bold;
    }

    /* ปรับแต่งหัวข้อ */
    h1, h2, h3 {
        color: #00ffcc !important;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. ส่วนหัวของเว็บ
st.markdown("<h1 style='text-align: center;'>🌐 SMART WASTE ANALYTICS 2024</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8892b0;'>ระบบวิเคราะห์และทำนายปริมาณขยะอัจฉริยะด้วย AI ปัญญาประดิษฐ์</p>", unsafe_allow_html=True)

DATA_PATH = 'sustainable_waste_management_dataset_2024.csv'

try:
    df, X_test, y_test, y_pred, metrics = train_waste_model(DATA_PATH)
    residuals = y_test - y_pred

    # 4. แสดง Metrics แบบเรืองแสง
    st.write("---")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(label="Model Accuracy (R²)", value=f"{metrics['R2']:.4f}")
    with m2:
        st.metric(label="Avg. Error (MAE)", value=f"{metrics['MAE']:.2f} kg")
    with m3:
        st.metric(label="System Status", value="ONLINE", delta="Stable")

    # 5. กราฟและข้อมูล
    tab1, tab2 = st.tabs(["✨ Visualization Hub", "📂 Data Exploration"])

    with tab1:
        c1, c2 = st.columns(2)
        
        # ปรับสีเส้นกราฟให้เข้ากับ Theme (Neon Style)
        plt.style.use('dark_background')
        
        with c1:
            st.subheader("🎯 Prediction Accuracy")
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            ax1.scatter(y_test, y_pred, alpha=0.5, color='#00ffcc', edgecolors='white')
            ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '#ff007f', lw=3, ls='--')
            ax1.set_xlabel('Actual (kg)', color='#8892b0')
            ax1.set_ylabel('Predicted (kg)', color='#8892b0')
            fig1.patch.set_facecolor('none') # ให้พื้นหลังกราฟโปร่งใส
            st.pyplot(fig1)

        with c2:
            st.subheader("📉 Error Distribution")
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.hist(residuals, bins=25, color='#ff007f', alpha=0.8, edgecolor='white')
            ax2.set_xlabel('Residual Value', color='#8892b0')
            fig2.patch.set_facecolor('none')
            st.pyplot(fig2)

    with tab2:
        st.subheader("📋 Data Overview")
        st.dataframe(df.style.background_gradient(cmap='Blues'), use_container_width=True)

except Exception as e:
    st.error(f"เกิดข้อผิดพลาด: {e}")
    st.info("กรุณาตรวจสอบว่าไฟล์ 'sustainable_waste_management_dataset_2024.csv' และ 'model_logic.py' อยู่ในโฟลเดอร์เดียวกัน")