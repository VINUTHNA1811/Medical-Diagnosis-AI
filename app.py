import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from datetime import datetime
from streamlit_lottie import st_lottie
import requests

# Load pre-trained deep learning models
pneumonia_model = tf.keras.models.load_model('Chest-xray_model.h5')
brain_model = tf.keras.models.load_model('brain_tumour_model.h5')

# Set up Streamlit page settings
st.set_page_config(page_title="Medical AI Dashboard", layout="wide", initial_sidebar_state="expanded")

# Initialize history in session state for storing recent predictions
if 'history' not in st.session_state:
    st.session_state.history = []

# Load a Lottie animation from URL
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    return None

lottie_dashboard = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json")

st.markdown("""
<style>
    .stApp { background-color: #121212; color: #f1f1f1; }
    .main-header { text-align: center; color: #00bcd4; font-size: 2.5rem; font-weight: 600; padding: 1rem 0; }
    .subtitle { text-align: center; color: #b0bec5; font-size: 1.1rem; margin-bottom: 2rem; }
    .dashboard-card, .result-card { background: #1e1e1e; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); margin: 1rem 0; }
    .metric-card { background: linear-gradient(135deg, #2196f3, #00bcd4); color: white; padding: 1.5rem; border-radius: 12px; text-align: center; margin: 0.5rem 0; }
    .upload-container { background: #2c2c2c; border: 2px dashed #00bcd4; border-radius: 12px; padding: 2rem; text-align: center; margin: 1rem 0; }
    .positive-result { border-left: 6px solid #ef5350; background: #2c2c2c; }
    .negative-result { border-left: 6px solid #66bb6a; background: #2c2c2c; }
    .progress-bar { width: 100%; height: 8px; background: #424242; border-radius: 4px; overflow: hidden; margin: 1rem 0; }
    .progress-fill { height: 100%; background: linear-gradient(90deg, #00bcd4, #4caf50); border-radius: 4px; transition: width 0.8s ease; }
</style>
""", unsafe_allow_html=True)

#  Sidebar navigation
app_mode = st.sidebar.radio("Medical Scan Menu", ["🏠 Dashboard", "🫁 Pneumonia Scan", "🧠 Brain MRI Scan"])

#  Dynamically update browser tab title based on selected page
page_titles = {
    "🏠 Dashboard": "Medical AI Dashboard",
    "🫁 Pneumonia Scan": "Pneumonia Scan - Chest X-ray",
    "🧠 Brain MRI Scan": "Brain MRI Scan - Tumor Detection"
}
page_title = page_titles.get(app_mode, "Medical AI Dashboard")
st.markdown(f"<script>document.title = '{page_title}';</script>", unsafe_allow_html=True)

# Image preprocessing
def preprocess_image(image, target_size, mode='RGB'):
    try:
        img = image.convert(mode)
        img = img.resize(target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize pixels
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        st.error(f"Image processing error: {e}")
        return None

#  Displaying the result box with progress bar and color-coded result
def display_result(result, confidence, status):
    result_class = "positive-result" if status == "positive" else "negative-result"
    icon = "🚨" if status == "positive" else "✅"
    color = "#ef5350" if status == "positive" else "#66bb6a"
    confidence = round(confidence, 1)

    st.markdown(f"""
    <div class="result-card {result_class}">
        <h3 style="color: {color};">{icon} {result}</h3>
        <p style="font-size: 1.1rem;">Confidence: {confidence:.1f}%</p>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {confidence}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

#  Predicting pneumonia using chest X-ray image
def predict_pneumonia(image):
    target_size = pneumonia_model.input_shape[1:3]
    mode = 'L' if pneumonia_model.input_shape[3] == 1 else 'RGB'
    img_array = preprocess_image(image, target_size, mode)
    if img_array is None:
        return "Error", 0, "error"
    prediction = pneumonia_model.predict(img_array, verbose=0)[0][0]
    result = "Pneumonia Detected" if prediction > 0.5 else "Normal Chest X-ray"
    status = "positive" if prediction > 0.5 else "negative"
    return result, prediction * 100, status

#  Predicting brain tumor using MRI scan
def predict_brain_tumor(image):
    target_size = brain_model.input_shape[1:3]
    mode = 'RGB'
    img_array = preprocess_image(image, target_size, mode)
    if img_array is None:
        return "Error", 0, "error"
    prediction = brain_model.predict(img_array, verbose=0)[0][0]
    result = "No Tumor Detected" if prediction > 0.5 else "Brain Tumor Detected"
    status = "negative" if prediction > 0.5 else "positive"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return result, confidence * 100, status

#  Emoji icons for page headings
page_icons = {
    "🏠 Dashboard": "📊",
    "🫁 Pneumonia Scan": "🫁",
    "🧠 Brain MRI Scan": "🧠"
}
page_icon = page_icons.get(app_mode, "")

# ------------> DASHBOARD PAGE
if app_mode == "🏠 Dashboard":
    # App name and tagline (changed only here)
    st.markdown("<div class='main-header'>🩺 Medical Diagnosis AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Your smart assistant for medical diagnosis.</div>", unsafe_allow_html=True)

    st_lottie(lottie_dashboard, speed=1, height=250, key="dashboard_lottie")

    st.subheader("📊 Dashboard Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="metric-card"><h3>🫁 Pneumonia Model</h3><p>Binary classifier</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card"><h3>🧠 Brain Tumor Model</h3><p>Binary classifier</p></div>""", unsafe_allow_html=True)

    #  Displaying the latest 5 predictions from session history
    if st.session_state.history:
        st.markdown("<h4>🕘 Recent History</h4>", unsafe_allow_html=True)
        for record in reversed(st.session_state.history[-5:]):
            st.markdown(f"""
            <div class="dashboard-card">
                <strong>Type:</strong> {record['type']}<br>
                <strong>Result:</strong> {record['result']}<br>
                <strong>Confidence:</strong> {record['confidence']}%<br>
                <strong>Time:</strong> {record['timestamp']}
            </div>
            """, unsafe_allow_html=True)

# ------------> PNEUMONIA SCAN PAGE
elif app_mode == "🫁 Pneumonia Scan":
    st.subheader("🫁 Upload Chest X-ray")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Chest X-ray", use_column_width=True)
        result, confidence, status = predict_pneumonia(image)
        if status != "error":
            display_result(result, confidence, status)
            st.session_state.history.append({
                "type": "Pneumonia",
                "result": result,
                "confidence": round(confidence, 1),
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        else:
            st.error("❌ Unable to analyze the X-ray. Please upload a valid image.")

# ------------> BRAIN MRI SCAN PAGE
elif app_mode == "🧠 Brain MRI Scan":
    st.subheader("🧠 Upload Brain MRI Image")
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Brain MRI", use_column_width=True)
        result, confidence, status = predict_brain_tumor(image)
        if status != "error":
            display_result(result, confidence, status)
            st.session_state.history.append({
                "type": "Brain MRI",
                "result": result,
                "confidence": round(confidence, 1),
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        else:
            st.error("❌ Unable to analyze the brain scan. Please upload a valid MRI image.")

# 🔹 Footer with medical disclaimer
st.markdown("""
---
<div style="text-align: center; color: #b0bec5; padding: 1rem;">
    <p><strong>Medical Disclaimer:</strong> This AI tool is for screening purposes only. Always consult healthcare professionals for medical diagnosis.</p>
</div>
""", unsafe_allow_html=True)
