import streamlit as st

st.title("EyeGuard Dashboard")

st.metric("Eye Status", "Eyes Open")
st.metric("Blink Count", "25")
st.metric("Screen Time", "10 minutes")
st.metric("Next Break", "9:45")

st.warning("Blink rate is low")