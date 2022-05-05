import pandas as pd
import streamlit as st
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.varmax import VARMAX
from timeit import default_timer as timer
from sklearn import metrics

#make page wider
st.set_page_config(layout="wide")

#import the data that we need for model
cpi = pd.read_csv('../CPI_Data/cpi_w_gold_oil.csv', index_col = 0)
cpi.index = pd.to_datetime(cpi.index, infer_datetime_format = True)

# there are null column values that we cannot fix
cpi = cpi.dropna(axis = 1)

st.markdown('# VARIMA Model')
st.markdown('Since CPI data can be interpretted as a type of time series data, we decided to proceed with a VARIMA model\
	 also known as a Vector Auto Regression Integrated Moving Average Model.')
st.write(cpi)

decomp = sm.tsa.seasonal_decompose(cpi['All items'], model = 'additive')
fig = decomp.plot()
plt.xlabel('Year')
plt.rcParams["figure.figsize"] = (30,10)
plt.show()