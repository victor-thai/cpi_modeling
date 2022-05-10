import base64
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import time
from PIL import Image

plt.style.use('dark_background')

raw_cpi = pd.read_csv('CPI_Data/Clean_Data.csv')
raw_cpi = raw_cpi.iloc[2:, :]
raw_cpi = raw_cpi.rename(columns={'Unnamed: 0': 'Date'})
raw_cpi = raw_cpi.set_index('Date')
raw_cpi.index = pd.to_datetime(raw_cpi.index)


st.title('Consumer Price Index EDA')
st.markdown("""
Consumer Price Index is an important economic index that measures the level of the price of goods and services. It also \
measures the level of the inflation rate of a country. Government pay close attention to the inflation rate because it \
indicates the level of economic growth. On the other hand, people care about the index as well because it indicates their \
buying power. Therefore, being able to predict the future value of the index is important, as it will help people and the \
government to make informed decisions. With that being said, let's first explore the data and do some EDA.
""")

st.header('Processed CPI index data')
st.write('Data Dimensions: ' + str(raw_cpi.shape[0]) + ' rows and ' + str(raw_cpi.shape[1]) + ' columns')
st.dataframe(raw_cpi)


def download_csv(df):
    csv_file = df.to_csv(index=False)
    b64 = base64.b64encode(csv_file.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="raw_cpi.csv">Download csv file</a>'
    return href


st.markdown(download_csv(raw_cpi), unsafe_allow_html=True)


def plot_cpi(df, options):
    st.header('Plot of CPI index')
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('index', fontsize=12)
    ax.legend(options, loc=2, fontsize=12)
    st.pyplot(fig)


plot_cpi(raw_cpi['All items'], ['All items'])

if st.checkbox('Plot categories'):
    option = st.multiselect('Select categories', ['Food', 'Energy', 'Apparel','New vehicles','Medical care commodities','Rent of primary residence','Transportation services'])
    st.write('You selected:', option)
    plot_cpi(raw_cpi[option], option)

st.header('Correlation')
st.write('As some index move in slightly different directions, we want to see if there is any correlation between each category')
st.image('output.png')
