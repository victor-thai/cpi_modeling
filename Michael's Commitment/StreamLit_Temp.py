import base64
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import time
from PIL import Image

raw_cpi = pd.read_csv('CPI_Data/Clean_Data.csv')
raw_cpi = raw_cpi.iloc[2:, :]
raw_cpi = raw_cpi.rename(columns={'Unnamed: 0': 'Date'})
raw_cpi = raw_cpi.set_index('Date')
raw_cpi.index = pd.to_datetime(raw_cpi.index)
prediction = pd.read_csv('CPI_Data/SARIMAX_prediction.csv', index_col=0)
prediction.index = pd.to_datetime(prediction.index, format='%Y-%m-%d')

st.title('Consumer Price Index Analysis')
st.markdown("""
Consumer Price Index is an important economic index that measures the level of the price of goods and services. It also \
measures the level of the inflation rate of a country. Government pay close attention to the inflation rate because it \
indicates the level of economic growth. On the other hand, people care about the index as well because it indicates their \
buying power. Therefore, being able to predict the future value of the index is important, as it will help people and the \
government to make informed decisions.
""")

st.header('Processed CPI index data')
st.write('Data Dimensions: ' + str(raw_cpi.shape[0]) + ' rows and ' + str(raw_cpi.shape[1]) + ' columns')
st.dataframe(raw_cpi.head(10))


def download_csv(df):
    csv_file = df.to_csv(index=False)
    b64 = base64.b64encode(csv_file.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="raw_cpi.csv">Download csv file</a>'
    return href


st.markdown(download_csv(raw_cpi), unsafe_allow_html=True)


def plot_cpi(df):
    st.header('Plot of CPI index')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df, color='blue', label='CPI')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('index', fontsize=12)
    ax.legend()
    st.pyplot(fig)


if st.checkbox('Plot CPI index'):
    option = st.selectbox('Select a category', ('All items', 'Food', 'Energy', 'Apparel','New vehicles','Medical care commodities','Rent of primary residence','Transportation services'))
    st.write('You selected:', option)
    plot_cpi(raw_cpi[option])

st.header('Data Transformation')
st.write(
    'The data needs to be stational enough for model training. Therefore we did some kind of transformations to prepare our datasets.')
st.subheader('Before transformation')
image = Image.open('decomp1.JPG')
st.image(image, caption='Seasonal Decomposition', use_column_width=True)
st.subheader('After transformation')
image2 = Image.open('decomp2.JPG')
st.image(image2, caption='Seasonal Decomposition', use_column_width=True)
image3 = Image.open('Trend_roll.JPG')
st.image(image3, caption='Trend Seasonality Double Check', use_column_width=True)
st.write(
    "The data is transformed into a stationary form after we eliminate trend. To include seasonality, we used SARIMA "
    "(Seasonal Auto Regressive Integrated Moving Average) model rather tha regular AR, MA and ARMA models. ")
st.write(
    "Then we used the AIC (Akaike's Information Criterion), which indicates the goodness of fit of the model, to decide the best"
    " parameters p, d, and q.")

st.header('Training')
st.write('The selected model for training based on AIC is SARIMAX (0, 0, 1) x (0, 1, 1, 12). '
         'Underneath is a diagnose of how the model fitted training set')
image4 = Image.open('diagnose.JPG')
st.image(image4, caption='Diagnose', use_column_width=True)
st.write('The data is normal distributed around 0 with a standard deviation of 1 illustrated by the four graphs. '
         'It proves that the model is able to fit the data well. Therefore, we fed 80% of the data to the model and use the'
         ' remaining 20% to test the model.')


def plot_prediction(df):
    st.header('Validation')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['index'], color='blue', label='CPI')
    ax.plot(df['prediction'][-15:], color='red', label='Prediction')
    ax.vlines(x=pd.to_datetime('2020-02-20', format='%Y-%m-%d'), ymin=230, ymax=280, colors='orange', label='Covid')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('index', fontsize=12)
    ax.legend()
    st.pyplot(fig)


plot_prediction(prediction[['index', 'prediction']])
st.markdown('* RMSE: **59.15**')
st.markdown('* Next Month CPI (2022-01-01): **263.35**')

st.write('Although some unpredictable events such as the Covid would cause deviation of the prediction from actual CPI, '
         'the model captured the trend of the CPI index well.')


