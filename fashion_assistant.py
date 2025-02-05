import sys
import os
import subprocess

# Force ChromaDB to use a compatible SQLite version
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass  # Will use the system default if pysqlite3 isn't installed

# Ensure required packages are installed
required_packages = ["streamlit", "langchain", "langchain-openai", "langchain-chroma", "chromadb", "pysqlite3-binary"]
for package in required_packages:
    try:
        __import__(package)
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import streamlit as st
from fashion_recommend import get_fashion_recommendation, refine_fashion_recommendation

# Initialize session state
if "recommendation_generated" not in st.session_state:
    st.session_state.recommendation_generated = False
original_recommendation = ""

# Page Config
st.set_page_config(page_title="AI Fashion Assistant", layout="wide")

# Sidebar for Inputs
with st.sidebar:
    st.header("🎨 Customize Your Outfit")
    gender = st.selectbox("Select your gender", ["Male", "Female"], index=0)
    occasion = st.text_input("Enter the occasion (e.g., Wedding, Business Meeting, Casual)")
    season = st.selectbox("Select the season", ["Spring", "Summer", "Autumn", "Winter"], index=0)
    style = st.text_input("Enter your preferred style (e.g., Formal, Casual, Streetwear)")
    detail = st.text_area("Additional details (e.g., It's a Japanese wedding!)")
    
    if st.button("👕 Generate Outfit Recommendations"):
        with st.spinner("AI is analyzing your preferences..."):
            text_recommendation, img_response = get_fashion_recommendation(gender, occasion, season, style, detail)
            original_recommendation = text_recommendation
            st.session_state.recommendation_generated = True
            st.session_state.recommendation_text = text_recommendation
            st.session_state.recommendation_img = img_response

# Main Display Section
st.title("👗 AI Fashion Assistant")
st.markdown("<h2 style='font-size:24px;'>💡 Enter your outfit preferences in the sidebar, and our AI assistant will generate personalized outfit recommendations!</h2>", unsafe_allow_html=True)

if st.session_state.recommendation_generated:
    st.markdown(f"<p style='font-size:20px;'>{st.session_state.recommendation_text}</p>", unsafe_allow_html=True)
    st.image(st.session_state.recommendation_img, width=500)
    
    # Follow-up query
    follow_up = st.text_input("Not satisfied? Add more details or ask a follow-up question:")
    if st.button("🔄 Refine Recommendation"):
        if follow_up:
            with st.spinner("Refining recommendations with additional details..."):
                follow_up_text_response, follow_up_img_response = refine_fashion_recommendation(follow_up, original_recommendation)
                st.session_state.recommendation_text = follow_up_text_response
                st.session_state.recommendation_img = follow_up_img_response
                st.markdown(f"<p style='font-size:20px;'>{follow_up_text_response}</p>", unsafe_allow_html=True)
                st.image(follow_up_img_response, width=500)
