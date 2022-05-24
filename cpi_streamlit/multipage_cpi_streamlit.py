import streamlit as st
import pandas as pd
import numpy as np
import time
import base64
import matplotlib.pyplot as plt
from PIL import Image
import multipage_template_streamlit as multipage

#varima 
from statsmodels.tsa.statespace.varmax import VARMAX
from timeit import default_timer as timer
import statsmodels.api as sm

#linear regression 
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#ridge regression
from sklearn.linear_model import Ridge

#sarima model
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools


#wide layout
st.set_page_config(layout='wide')

#Clears the cache when the app is started
multipage.start_app()

#multipage object
app = multipage.MultiPage()
app.start_button = "Let's explore this!"
app.navbar_name = "Table of Contents"
app.next_page_button = "Next Page"
app.previous_page_button = "Previous Page"


# intro page function to start
def intropage():
	st.header('CPI Modeling')

	st.markdown("# What will our economy look like in the next month?")

	intro_left_col, intro_right_col = st.columns(2)

	with intro_right_col:
		st.write("This project focues on analyzing the trends of Consumer \
				Price Index, also known as CPI, and applies a predictive VAR model \
				 to predict the next month's CPI.")

	with intro_left_col:
		st.image('cpi_streamlit_photos/cpi_intro_photo.jpeg')

# home page that needs to be exactly like intropage func
def homepage(prevpage):
	st.header('CPI Modeling')

	st.markdown("# What will our economy look like in the next month?")

	intro_left_col, intro_right_col = st.columns(2)

	with intro_right_col:
		st.write("This project focues on analyzing the trends of Consumer \
				Price Index, also known as CPI, and applies a predictive VAR model \
				 to predict the next month's CPI.")

	with intro_left_col:
		st.image('cpi_streamlit_photos/cpi_intro_photo.jpeg')


# eda page
def edapage(prevpage):
	plt.style.use('dark_background')

	raw_cpi = pd.read_csv('../CPI_Data/Clean_data.csv')
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
	st.write('Our Data Dimensions consist of ' + str(raw_cpi.shape[0]) + ' rows and ' + str(raw_cpi.shape[1]) + ' columns.')
	st.dataframe(raw_cpi)


	def download_csv(df):
	    csv_file = df.to_csv(index=False)
	    b64 = base64.b64encode(csv_file.encode()).decode()
	    href = f'<a href="data:file/csv;base64,{b64}" download="raw_cpi.csv">Click here to download this csv file!</a>'
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
	st.image('cpi_streamlit_photos/output.png')


# varima model page
def modelpage(prevpage):
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
		plt.plot(test[results_option], label='Test '+str(results_option))
		plt.plot(result[results_option], label='Predicted '+str(results_option))
		plt.legend(loc='best')
		st.pyplot(plt)#.show()


# other models page
def othermodels(prevpage):
	st.header("Other noteable models")

	st.markdown("""
		Although we ended up using the VARIMA model as our final model for our predictions, other models led us to build up to the final model.
		Initially, we looked into a simple multiple linear regression model for our predictions...
		""")

	# linear regression model
	CPI = pd.read_csv('../CPI_Data/Cleaned_CPI_data.csv')
	CPI.index = pd.to_datetime(CPI.index, infer_datetime_format = True)
	CPI = CPI.dropna(axis = 1)
	CPI = CPI[CPI['Unnamed: 0'].str.contains("Percent")==False]
	CPI = CPI.set_index('Unnamed: 0')

	merged = CPI

	merged = merged.replace('2012-01',0) #could not convert string to float: '2012-01'

	merged = merged.apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)


	# remove energy related categories
	x = merged.drop(['Household energy', 'Fuel oil', 'Fuel oil', 'Propane, kerosene, and firewood', 'Energy services', 'Electricity', 'Utility ',
	'Fuel oil and other fuels', 'Motor fuel', 'Other motor fuels','Transportation commodities less motor fuel'
	,'Energy', 'Energy commodities', 'Energy services', 'Fuels and utilities', 'Gasoline '], axis=1)


	#separte the predicting attribute into Y for model training 
	y = merged.get(['Energy'])

	# manually split test and training data
	x_train = x.iloc[:96]
	x_test = x.iloc[96:]
	y_train = y.iloc[:96]
	y_test = y.iloc[96:]

	#create the linear regression environment
	LinearRegression_model = LinearRegression()

	#fit the data that we want to use in the model
	LinearRegression_model.fit(x_train,y_train)

	# predict using the X values to get Y predictions
	pred_test_lr= LinearRegression_model.predict(x_test)

	
	st.markdown("""
		### Linear Regression
		""")

	# prediction output series to plot
	predict_date = pd.date_range(y_test.index[0], y_test.index[-1], freq = '1MS')
	predictions = pd.Series(list(pred_test_lr), index = predict_date)

	# get first elem in list for predictions
	def first_elem(elem):
		return elem[0]
	predictions = predictions.apply(first_elem)
	predictions.name = 'Energy'

	CPI.index = pd.to_datetime(CPI.index)
	y_test.index = pd.to_datetime(y_test.index)


	# write linear regression results
	linearregression = st.button('Predict using Linear Regresssion!')
	if linearregression:
		st.session_state.load_state = True
		st.write("RMSE for Linear Regression is " ,np.sqrt(metrics.mean_squared_error(y_test,pred_test_lr))) 



	# Ridge Regression
	st.markdown("""
		### Ridge Regression
		""")

	# creating ridge regression 
	rr = Ridge(alpha = 1)
	# fit with testing data
	rr.fit(x_train, y_train) 
	# predict using training data
	pred_train_rr= rr.predict(x_train)

	# predict using test data
	pred_test_rr= rr.predict(x_test)

	ridgeregression = st.button('Predict using Ridge Regression!')
	if ridgeregression:	
		st.session_state.load_state = True
		st.write("RMSE for Ridge Regression is ", np.sqrt(metrics.mean_squared_error(y_test,pred_test_rr))) 



	# SARIMA model
	st.markdown("""
		### SARIMA model
		""")

	CPI = pd.read_csv("../CPI_Data/Merged_Data/Merged_CPI.csv")
	CPI.set_index('Unnamed: 0', inplace = True)
	CPI.index = pd.to_datetime(CPI.index, format = '%Y%m')
	CPI.index = pd.DatetimeIndex(CPI.index.values,
	                               freq=CPI.index.inferred_freq)
	X = CPI['All items']
	X = X.dropna()

	X = X.diff().dropna()

	train_all = X[:-15]
	test_all = X[-15:]

	model_all = SARIMAX(train_all, order = (0,0,1), seasonal_order = (0,1,1,12))
	model_all = model_all.fit()

	pred_all = model_all.predict(start=0,end=train_all.shape[0],typ='levels').rename('SARIMAX predictions')

	#Reverse Prediction
	pred_t_all = model_all.predict(start=train_all.shape[0],end=train_all.shape[0]+test_all.shape[0]-1,typ='levels').rename('SARIMAX predictions')
	last_month_train = CPI['All items'][-16]
	# cum sum from last month to initial month
	pred_t_all[0] = pred_t_all[0] + last_month_train
	pred_t_all = pd.DataFrame(pred_t_all.cumsum())
	predicted_index = pd.DataFrame({'index':CPI['All items'].dropna()})
	SARIMAX_pred_test = np.concatenate([predicted_index[:-15], pred_t_all], axis=0)
	predicted_index['prediction'] = SARIMAX_pred_test
	
	y = predicted_index['index'].iloc[train_all.shape[0]:]
	y_pred = predicted_index['prediction'].iloc[train_all.shape[0]:]

	#function to calculate rmse
	def evaluation(y, prediction):
		return np.sqrt(sum(y ** 2 - prediction ** 2) / y.shape[0])

	sarima = st.button('Predict using SARIMA model!')
	if sarima:
		st.write("RMSE for SARIMA: ", evaluation(y,y_pred))


# discussion page about overall project and models
def discussion(prevpage):
	st.header("Discusion")

	st.markdown("""
		This will be the discussion page that talks about our overall project.
		* problems that we had
		* relevance to the real world
		* why we did what we did
""")


# about us page
def aboutuspage(prev_page):
	st.title("About Us")

	banner = Image.open("cpi_streamlit_photos/ds3_banner.PNG")
	st.image(banner, width = 700)

	st.write("We are a group of students from UCSD's Data Science Student Society. As members of the CPI project team, we show passion, initiative, and competence to contribute to this 2-quarter long project. Our team has a diverse background but we are brought together by a common love for data science.")

	st.title("Project Members")

	v_col1, v_col2 = st.columns(2)
	with v_col1:
	    st.subheader("Project Lead: Victor Thai")
	    st.write("Victor is a second-year at UC San Diego, majoring in Data Science and minor in Cognitive Science. He is a member of CASA at UCSD and enjoys weightlifting to relieve stress.")
	with v_col2:
	    st.subheader("")
	    v_img = Image.open("cpi_streamlit_photos/victor_headshot.jpeg")
	    st.image(v_img, width = 200)

	m_col1, m_col2 = st.columns(2)
	with m_col1:
	    st.subheader("Michael Chen")    
	    st.write("Michael is a second-year student at UC San Diego, majoring in Data Science and Minor in Management Science. Michael was in the Chinese Union. A fun fact about him is that he gets excited when facing a hard task to be solved.")
	with m_col2:
	    st.subheader("")
	    m_img = Image.open("cpi_streamlit_photos/michael_headshot.jpeg")
	    st.image(m_img, width = 200)
	    
	s_col1, s_col2 = st.columns(2)
	with s_col1:
	    st.subheader("Stephanie Chavez")
	    st.write("Stephanie is a second-year student at UC San Diego, majoring in Data Science. Stephanie is a sister of Sigma Pi Alpha Sorority, Inc. and is a First-Gen Student Success Peer Coach. Some of her interests outside of data science include musical theatre, fantasy books, and dogs.")
	with s_col2:
		st.subheader("")
		steph_img = Image.open("cpi_streamlit_photos/steph_headshot.jpg")
		st.image(steph_img, width = 200)

	z_col1, z_col2 = st.columns(2)
	with z_col1:
	    st.subheader("Zed Siyed")
	    st.write("Zed is a first-year student at UC San Diego, majoring in Computer Science. Zed is also a member of Triton Consulting Group as the VP Tech Consulting. Some of Zed's hobbies are sports analytics and weightlifting.")
	with z_col2:
	    st.subheader("")
	    zed_img = Image.open("cpi_streamlit_photos/zed_headshot.PNG")
	    st.image(zed_img, width = 200)


app.set_initial_page(intropage)
app.add_app("Home", homepage)
app.add_app("EDA/Dataset", edapage)
app.add_app("VARIMA model", modelpage)
app.add_app("Other models", othermodels)
app.add_app("Discussion", discussion)
app.add_app("About Us", aboutuspage)

app.run()