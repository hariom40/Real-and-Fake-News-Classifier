import streamlit as st
import pickle
from sklearn.linear_model import LogisticRegression

# Load the pre-trained model and vectorizer
model = pickle.load(open('pred.pkl', 'rb'))
vector = pickle.load(open('tfidf.pkl', 'rb'))

# Set Streamlit page config
st.set_page_config(
    page_title="Fake News Detector - AI Powered",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Apply Custom CSS for Modern Styling
st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        
        /* Main App Background */
        .stApp {
            background: black;
            font-family: 'Poppins', sans-serif;
        }
        
        /* Main Container */
        .main {
            background-color: rgba(255, 255, 255, 0.95);
            padding: 3rem 2.5rem;
            border-radius: 30px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
            margin-top: 2rem;
        }
        
        /* Hide Streamlit Branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Title Styling */
        .title {
            font-size: 48px !important;
            font-weight: 700 !important;
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px !important;
            font-family: 'Poppins', sans-serif;
        }
        
        /* Subtitle */
        .subtitle {
            font-size: 18px;
            color: #718096;
            text-align: center;
            margin-bottom: 30px;
            font-weight: 400;
        }
        
        /* Info Box */
        .info-box {
            background: linear-gradient(135deg, #edf2f7 0%, #e2e8f0 100%);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 25px;
            border-left: 5px solid #667eea;
        }
        
        .info-box p {
            color: #1a202c;
            font-size: 15px;
            margin: 0;
            line-height: 1.6;
            font-weight: 500;
        }
        
        /* Text Area Styling */
        .stTextArea label {
            font-size: 16px !important;
            font-weight: 600 !important;
            color: #1a202c !important;
            margin-bottom: 10px !important;
        }
        
        .stTextArea textarea {
            border: 2px solid #e2e8f0 !important;
            border-radius: 15px !important;
            padding: 20px !important;
            font-size: 16px !important;
            font-family: 'Poppins', sans-serif !important;
            background-color: #ffffff !important;
            color: #1a202c !important;
            transition: all 0.3s ease !important;
        }
        
        .stTextArea textarea:focus {
            border-color: #667eea !important;
            background-color: white !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        }
        
        /* Button Styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border-radius: 15px !important;
            padding: 18px 24px !important;
            font-size: 18px !important;
            font-weight: 600 !important;
            border: none !important;
            width: 100% !important;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3) !important;
            transition: all 0.3s ease !important;
            font-family: 'Poppins', sans-serif !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4) !important;
        }
        
        .stButton > button:active {
            transform: translateY(0) !important;
        }
        
        /* Result Box - True News */
        .result-true {
            background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
            color: white;
            font-size: 28px;
            font-weight: 700;
            text-align: center;
            padding: 30px;
            border-radius: 20px;
            margin-top: 30px;
            box-shadow: 0 10px 30px rgba(86, 171, 47, 0.3);
            animation: fadeIn 0.5s ease;
        }
        
        .result-true-subtitle {
            font-size: 16px;
            font-weight: 400;
            margin-top: 10px;
            color: rgba(255, 255, 255, 0.95);
        }
        
        /* Result Box - Fake News */
        .result-fake {
            background: linear-gradient(135deg, #fc5c7d 0%, #e91e63 100%);
            color: white;
            font-size: 28px;
            font-weight: 700;
            text-align: center;
            padding: 30px;
            border-radius: 20px;
            margin-top: 30px;
            box-shadow: 0 10px 30px rgba(252, 92, 125, 0.3);
            animation: fadeIn 0.5s ease;
        }
        
        .result-fake-subtitle {
            font-size: 16px;
            font-weight: 400;
            margin-top: 10px;
            color: rgba(255, 255, 255, 0.95);
        }
        
        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: scale(0.95);
            }
            to {
                opacity: 1;
                transform: scale(1);
            }
        }
        
        /* Warning Message */
        .stAlert {
            border-radius: 15px !important;
            border-left: 5px solid #f59e0b !important;
        }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("<h1 class='title'>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Powered by Machine Learning & AI</p>", unsafe_allow_html=True)

# Info Box
st.markdown("""
    <div class='info-box'>
        <p>🔍 <strong>How it works:</strong> Enter or paste a news article below to analyze its authenticity. 
        Our AI model will evaluate the content and determine if it's likely to be genuine or fake news.</p>
    </div>
""", unsafe_allow_html=True)

# Input Section
st.markdown("### 📝 News Article Text")
message = st.text_area(
    "", 
    placeholder="Paste the news article or headline here...",
    height=200,
    label_visibility="collapsed"
)

# Analyze Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("🔎 Analyze News Article")

# Prediction Logic
if analyze_button:
    if message.strip():
        with st.spinner("🤖 Analyzing with AI..."):
            transformed_message = vector.transform([message])
            output = model.predict(transformed_message)[0]
            
            # Display result
            if output == 1:
                st.markdown("""
                    <div class='result-true'>
                        ✅ Likely Authentic News
                        <div class='result-true-subtitle'>
                            Our AI model has identified this content as likely to be true and reliable.
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown("""
                    <div class='result-fake'>
                        ⚠️ Fake News Detected!
                        <div class='result-fake-subtitle'>
                            Our AI model has identified this content as potentially false or misleading.
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                st.snow()
    else:
        st.warning("⚠️ Please enter news content before analyzing")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: #718096; font-size: 14px; padding: 20px;'>
        💡 <em>Note: Results are based on machine learning predictions and should be verified with multiple sources.</em>
    </div>
""", unsafe_allow_html=True)