import streamlit as st
from detect_page import show_detect_page
from explore_page import show_explore_page

st.set_page_config(page_title=None, page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)

pageval=st.sidebar.selectbox("Explore or Detect", ("Explore","Detect"))


if(pageval=="Detect"):
    show_detect_page()
else:
    show_explore_page()    
