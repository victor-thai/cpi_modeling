import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import itertools


st.write("## Predictive Modeling: Consumer Price Index")

df = pd.read_csv('cpi_w_gold_oil.csv', index_col = 0).dropna(axis=1)
df.index = pd.to_datetime(df.index, infer_datetime_format = True)

all_categories = df.columns.values
category = st.selectbox(
        'Choose a category',
        all_categories)


st.line_chart(df[category])

st.write("*Insert disclaimers/information on data collection*")


decom_all = seasonal_decompose(df[category], model='additive')

st.write(decom_all.plot())

if st.button("Try adfuller test"):
        st.write("adfuller test")
        dftest = adfuller(df[category])
        st.subheader(f'ADF: {dftest[0]}')
        st.subheader(f'p-value: {dftest[1]}')

if st.button("Try adfuller test after differencing"):
        st.write("adfuller test after differencing")
        df_diff = (((df.diff()).dropna()).diff()).dropna()
        dftest = adfuller(df_diff[category])
        st.subheader(f'ADF: {dftest[0]}')
        st.subheader(f'p-value: {dftest[1]}')
