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

train = cpi.iloc[:96]
test = cpi.iloc[96::]

st.markdown('# VARIMA Model')
st.markdown("""
	Since CPI data can be interpretted as a type of time series data, we decided to proceed with a VARIMA model
	also known as a Vector Auto Regression Integrated Moving Average Model.

	The VARIMA model is a multivariate forecasting algorithm that is used when two or more time series data influence each
	other. In our model, we used the features of past CPI data, PPI data, US Crude Oil prices, and US Gold prices. It is
	modeled as a function of the past values, that is the predictors are nothing but the lags (time delayed value)
	of the series. Compared to ARIMA models, this model is bi-directional, meaning all parameters can be used to
	influence oneother.""")
st.write(cpi)



st.markdown("""
	In order for the VARIMA model to work, we must test for stationarity. Within modeling,
	 stationarity is when the mean, variance, and covariance are constant and not dependent on time. In other
	 words, our data used for modeling cannot show any clear trends over time. Among all four elements of our
	 data we gathered, they all showed signs of stationarity. 
	""")

option = st.selectbox('What data would you like to look at?',('US CPI All items', 'US Crude Oil Prices', 'US Gold Prices'))
st.header(option + " seasonal decomposition plot")
select_col1, select_col2 = st.columns([6.5,3.5])
if option == 'US CPI All items':
	with select_col1:
		st.image('cpi_streamlit_photos/all_items_seasonal_decomp.jpeg')
	with select_col2:
		st.write('The graph here shows the seasonal decomposition for ' + option +
			'. When looking at these decomps, we are given useful information on the trend, seasonality, and residual information!')
elif option == 'US Crude Oil Prices':
	with select_col1:
		st.image('cpi_streamlit_photos/crude_oil_seasonal_decomp.jpeg')
	with select_col2:
		st.write('The graph here shows the seasonal decomposition for ' + option +
			'. When looking at these decomps, we are given useful information on the trend, seasonality, and residual information!')
elif option == 'US Gold Prices':
	with select_col1:
		st.image('cpi_streamlit_photos/gold_seasonal_decomp.jpeg')
	with select_col2:
		st.write('The graph here shows the seasonal decomposition for ' + option +
			'. When looking at these decomps, we are given useful information on the trend, seasonality, and residual information!')


st.markdown("""
	To get rid of stationarity, it is important to
	 difference the data which would transform our data into new data that doesn’t show any trend, but rather
	 the difference one value subtracted by another. Through differencing our data to get stationary data, we
	 plotted an Autocorrelation Function plot. This gives insight into the parameters that we should use for our
	 VARIMA model.""")

st.markdown("""
	Let's predict next month's cpi!
	""")

model = VARMAX(train[['All items', 'Crude Oil Price', 'Gold US dollar per oz']], order=(4,0)).fit( disp=False)
result = model.forecast(steps = 24)

predict_button = st.button('Predict')
if predict_button:
	st.write(result[0:1])

	for i in ['All items', 'Crude Oil Price', 'Gold US dollar per oz']:
	    
	    plt.rcParams["figure.figsize"] = [10,7]
	    plt.plot( train[str(i)], label='Train '+str(i))
	    plt.plot(test[str(i)], label='Test '+str(i))
	    plt.plot(result[str(i)], label='Predicted '+str(i))
	    plt.legend(loc='best')
	    plt.show()

st.markdown("""
	As we plot our model, these are the results that we yield.
	""")

results_option = st.selectbox("What results would you like to look?", ('All items', 'Crude Oil Price', 'Gold US dollar per oz'))
results_left, results_right = st.columns([3,7])
with results_left:
	st.write('Observing the results that we get from the shown plots, it can be seen that the predictions yield results that are quite \
		accurate given the data that is receives to be used for modeling. Since our testing data was surrounded around a pandemic, our predictions\
		 are not as accurate as we would like them to be. ')
with results_right:
	plt.rcParams["figure.figsize"] = [10,7]
	plt.title(results_option + ' Predictions', loc='center')
	plt.plot(train[results_option], label='Train '+str(results_option))
	plt.plot(test[results_option], label='Test '+results_option)
	plt.plot(result[results_option], label='Predicted '+results_option)
	plt.legend(loc='best')
	st.pyplot(plt)#.show()

# for i in ['All items', 'Crude Oil Price', 'Gold US dollar per oz']:
	    
#     plt.rcParams["figure.figsize"] = [10,7]
#     plt.plot( train[str(i)], label='Train '+str(i))
#     plt.plot(test[str(i)], label='Test '+str(i))
#     plt.plot(result[str(i)], label='Predicted '+str(i))
#     plt.legend(loc='best')
#     st.pyplot(plt)#.show()