import streamlit as st

def apply_custom_css():
    st.markdown("""
        <style>
        .main {
            background-color: #f5f5f5;
        }
        .stTextInput {
            background-color: white;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)