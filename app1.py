import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import base64
from model_logic import train_waste_model

# Function to convert image to base64
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None

# Page Configuration
st.set_page_config(
    page_title="Eco-Smart Waste Analytics",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load background image
BACKGROUND_IMAGE = "studio-background-concept-dark-gradient-purple-studio-room-background-product.jpg"  # เปลี่ยนชื่อไฟล์ตามรูปของคุณ
bg_image = get_base64_image(BACKGROUND_IMAGE)

# Modern CSS Styling
st.markdown(f"""
    <style>
    /* Import Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {{
        font-family: 'Inter', sans-serif;
    }}
    
    /* Main Background with Image or Gradient */
    .stApp {{
        {'background-image: url("data:image/jpeg;base64,' + bg_image + '");' if bg_image else 'background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);'}
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    
    /* Optional: Add overlay for better readability */
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.3); /* ปรับความมืดได้ที่นี่ */
        z-index: -1;
        pointer-events: none;
    }}
    
    /* Header Styling */
    .main-header {{
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }}
    
    .main-title {{
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 0;
        letter-spacing: -1px;
    }}
    
    .subtitle {{
        text-align: center;
        color: #ffffff;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }}
    
    /* Metric Cards - Glassmorphism */
    div[data-testid="metric-container"] {{
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1rem;
        border-radius: 12px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }}
    
    div[data-testid="metric-container"]:hover {{
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2);
    }}
    
    [data-testid="stMetricLabel"] {{
        font-size: 0.8rem !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    [data-testid="stMetricValue"] {{
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }}
    
    [data-testid="stMetricDelta"] {{
        font-size: 0.8rem !important;
        color: #ffffff !important;
    }}
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: rgba(255, 255, 255, 0.95);
        padding: 0.5rem;
        border-radius: 12px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        background: transparent;
        border-radius: 8px;
        color: #000000 !important;
        font-weight: 600;
        padding: 0 24px;
        transition: all 0.3s ease;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background: rgba(102, 126, 234, 0.1);
        color: #000000 !important;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: black !important;
    }}
    
    /* Dataframe Styling */
    .dataframe {{
        border-radius: 12px;
        overflow: hidden;
    }}
    
    /* Subheader Styling */
    .stSubheader, h2, h3, h4 {{
        color: #ffffff !important;
        font-weight: 600 !important;
        margin-bottom: 1rem !important;
    }}
    
    /* Paragraph and text */
    p, span, div {{
        color: #ffffff !important;
    }}
    
    /* Spinner text */
    .stSpinner > div {{
        color: #ffffff !important;
    }}
    
    /* Text input */
    input {{
        color: #1e293b !important;
    }}
    .stAlert {{
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }}
    
    /* Divider */
    hr {{
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: rgba(255, 255, 255, 0.2);
    }}
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
    <div class="main-header">
        <h1 class="main-title">♻️ Eco-Smart Waste Analytics</h1>
        <p class="subtitle">ระบบวิเคราะห์และทำนายปริมาณขยะอัจฉริยะด้วย AI</p>
    </div>
""", unsafe_allow_html=True)

DATA_PATH = 'sustainable_waste_management_dataset_2024.csv'

try:
    # Load and train model
    with st.spinner('🔄 กำลังโหลดและวิเคราะห์ข้อมูล...'):
        df, X_test, y_test, y_pred, metrics = train_waste_model(DATA_PATH)
        residuals = y_test - y_pred

    # Metrics Section
    st.markdown("### 📊 ผลการวิเคราะห์")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🎯 ความแม่นยำของโมเดล",
            value=f"{metrics['R2']:.2%}",
            delta="Excellent" if metrics['R2'] > 0.9 else "Good"
        )
    
    with col2:
        st.metric(
            label="📉 ค่าเฉลี่ยความคาดเคลื่อน",
            value=f"{metrics['MAE']:.1f} kg",
            delta="Low Error" if metrics['MAE'] < 50 else "Moderate"
        )
    
    with col3:
        st.metric(
            label="💚 สถานะระบบ",
            value="Online",
            delta="Ready"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Content Tabs
    tab1, tab2, tab3 = st.tabs(["📈 การวิเคราะห์เชิงลึก", "📊 การทำนาย", "🗂️ ข้อมูลดิบ"])

    with tab1:
        st.markdown("#### 📈 วิเคราะห์ประสิทธิภาพ Model แบบละเอียด")
        
        # แถวที่ 1: Actual vs Predicted และ Residual vs Predicted
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 🎯 Actual vs. Predicted")
            fig1, ax1 = plt.subplots(figsize=(4, 3))
            ax1.scatter(y_test, y_pred, alpha=0.6, color='#667eea', edgecolors='white')
            ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='#ff007f', ls='--', lw=2)
            ax1.set_xlabel('Actual Waste (kg)')
            ax1.set_ylabel('Predicted Waste (kg)')
            st.pyplot(fig1, use_container_width=False)

        with c2:
            st.markdown("##### 📉 Residuals vs. Predicted")
            fig2, ax2 = plt.subplots(figsize=(4, 3.))
            ax2.scatter(y_pred, residuals, alpha=0.6, color='#764ba2')
            ax2.axhline(0, linestyle='--', color='white', alpha=0.5)
            ax2.set_xlabel('Predicted Waste (kg)')
            ax2.set_ylabel('Residuals (kg)')
            st.pyplot(fig2, use_container_width=False)

        st.divider()

        # แถวที่ 2: Distribution และ Actual vs Predicted Time Series
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("##### 📊 Distribution of Residuals")
            fig3, ax3 = plt.subplots(figsize=(4.4, 3))
            ax3.hist(residuals, bins=30, color='#667eea', alpha=0.7, edgecolor='white')
            ax3.set_xlabel('Residual (kg)')
            ax3.set_ylabel('Frequency')
            st.pyplot(fig3, use_container_width=False)

        with c4:
            st.markdown("##### 🕒 Actual vs Predicted Over Samples")
            fig4, ax4 = plt.subplots(figsize=(4, 3))
            ax4.plot(y_test.values[:50], label='Actual', color='#00ffcc', linewidth=2)
            ax4.plot(y_pred[:50], label='Predicted', color='#ff007f', linestyle='--')
            ax4.legend()
            ax4.set_xlabel('Sample Index (First 50)')
            st.pyplot(fig4, use_container_width=False)

        st.divider()

        # แถวที่ 3: Percentage Error และ Boxplot
        c5, c6 = st.columns(2)
        with c5:
            st.markdown("##### 💹 Percentage Error vs Actual")
            percentage_error = (residuals / y_test) * 100
            fig5, ax5 = plt.subplots(figsize=(4.3, 3))
            ax5.scatter(y_test, percentage_error, alpha=0.6, color='#f1c40f')
            ax5.axhline(0, linestyle='--', color='white', alpha=0.5)
            ax5.set_xlabel('Actual Waste (kg)')
            ax5.set_ylabel('Percentage Error (%)')
            st.pyplot(fig5, use_container_width=False)

        with c6:
            st.markdown("##### 📦 Boxplot of Residuals")
            fig6, ax6 = plt.subplots(figsize=(4, 3))
            ax6.boxplot(residuals, vert=False, patch_artist=True, 
                        boxprops=dict(facecolor='#667eea', color='white'),
                        whiskerprops=dict(color='white'),
                        capprops=dict(color='white'))
            ax1.tick_params(colors='white')
            ax6.set_xlabel('Residual (kg)')
            st.pyplot(fig6, use_container_width=False)

    with tab2:
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("#### 📊 การกระจายของค่าคลาดเคลื่อน")
            
            fig2, ax2 = plt.subplots(figsize=(4, 2.5), facecolor='white')
            ax2.hist(residuals, bins=30, color='#667eea', alpha=0.7, edgecolor='white', linewidth=1.2)
            ax2.axvline(x=0, color='#764ba2', linestyle='--', linewidth=2, label='Zero Error')
            ax2.set_xlabel('Prediction Error (kg)', fontsize=8, color='#1e293b', fontweight='600')
            ax2.set_ylabel('Frequency', fontsize=8, color='#1e293b', fontweight='600')
            ax2.grid(True, alpha=0.2, axis='y')
            ax2.legend()
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            st.pyplot(fig2)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_b:
            st.markdown("#### 📈 สถิติเชิงลึก")
            
            stats_df = pd.DataFrame({
                'Metric': ['Mean Error', 'Std Dev', 'Min Error', 'Max Error', 'Q1', 'Median', 'Q3'],
                'Value': [
                    f"{residuals.mean():.2f} kg",
                    f"{residuals.std():.2f} kg",
                    f"{residuals.min():.2f} kg",
                    f"{residuals.max():.2f} kg",
                    f"{residuals.quantile(0.25):.2f} kg",
                    f"{residuals.median():.2f} kg",
                    f"{residuals.quantile(0.75):.2f} kg"
                ]
            })
            
            st.dataframe(
                stats_df,
                use_container_width=True,
                hide_index=True
            )

    with tab3:
        st.markdown("#### 🗂️ ข้อมูลต้นฉบับ")
        st.markdown(f"แสดงข้อมูลทั้งหมด {len(df):,} แถว")
        
        # Add search functionality
        search = st.text_input("🔍 ค้นหาข้อมูล", placeholder="พิมพ์เพื่อค้นหา...")
        
        if search:
            mask = df.astype(str).apply(lambda x: x.str.contains(search, case=False, na=False)).any(axis=1)
            filtered_df = df[mask]
            st.info(f"พบ {len(filtered_df)} แถวที่ตรงกับการค้นหา")
            st.dataframe(filtered_df, use_container_width=True, height=400)
        else:
            st.dataframe(df, use_container_width=True, height=400)

    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center; color: rgba(255,255,255,0.7); padding: 2rem;'>
            <p>🌿 Powered by AI & Machine Learning | Made with Streamlit ❤️</p>
        </div>
    """, unsafe_allow_html=True)

except FileNotFoundError:
    st.error("❌ ไม่พบไฟล์ข้อมูล")
    st.info("📁 กรุณาตรวจสอบว่าไฟล์ 'sustainable_waste_management_dataset_2024.csv' อยู่ในโฟลเดอร์เดียวกัน")
except ImportError:
    st.error("❌ ไม่พบไฟล์ model_logic.py")
    st.info("📄 กรุณาตรวจสอบว่าไฟล์ 'model_logic.py' อยู่ในโฟลเดอร์เดียวกัน")
except Exception as e:
    st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
    st.info("💡 ลองรีเฟรชหน้าเว็บหรือตรวจสอบข้อมูลอีกครั้ง")