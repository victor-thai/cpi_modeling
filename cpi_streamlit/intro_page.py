import pandas as pd
import streamlit as st

st.set_page_config(layout="wide")
st.header('CPI Modeling')

st.markdown("# What will our economy look like in the next month?")

intro_left_col, intro_right_col = st.columns(2)

with intro_right_col:
	st.write("This project focues on analyzing the trends of Consumer \
			Price Index, also known as CPI, and applies a predictive VAR model \
			 to predict the next month's CPI.")

with intro_left_col:
	st.image('cpi_intro_photo.jpeg')