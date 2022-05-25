import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from PIL import Image
from statsmodels.tsa.statespace.varmax import VARMAX
from timeit import default_timer as timer
from sklearn import metrics
from pmdarima import auto_arima
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import base64
import statsmodels.api as sm
import itertools
import warnings
warnings.filterwarnings("ignore")


st.set_page_config(layout="wide")

st.header("CPI Predictive Modeling")
st.caption("By Victor Thai, Michael Chen, Stephanie Chavez, and Zed Siyed")

st.sidebar.write("## CPI Modeling")

others = 0

#Intro Page
if st.sidebar.checkbox("Intro"):

        st.title("Intro")

        st.subheader("What is CPI?")
        st.write("Consumer Price Index (CPI) is an index used to measure the average change \
                in the price of a market basket of goods. CPI is often used to measure inflation, \
                and is also used to determine deflation of the consumer dollar.")

        st.subheader("Why CPI?")
        st.write("Recent events make CPI a very important and interesting metric. \
                The COVID-19 pandemic massively impacted inflation...")

        st.subheader("Project Timeline")
        st.write("*DISCUSS TIMELINE AS A GROUP. WHAT PARTS ARE RELEVANT?*")


#EDA Page
if st.sidebar.checkbox("Data Cleaning and EDA"):
        raw_cpi = pd.read_csv('clean_data.csv')
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


#Model Page
if st.sidebar.checkbox("Model"):

        cpi = pd.read_csv('cpi_w_gold_oil.csv', index_col = 0)
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

        st.write("""
                ### All items seasonal decomposition plot
                """)

        st.image('all_items_seasonal_decomp.jpeg')

        st.write("""
                ### US Crude Oil prices seasonal decomposition plot
                """)

        st.image('crude_oil_seasonal_decomp.jpeg')

        st.write("""
                ### US Gold prices seasonal decomposition plot
                """)
        st.image('gold_seasonal_decomp.jpeg')

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

        #predict_button = st.button('Predict')
        st.write("Predition:")
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



#Discussion Page
if st.sidebar.checkbox("Discussion"):

        st.title("Discussion")

        st.subheader("Why would a predictive model useful?")
        st.write("Our model allows us to predict CPI changes and gives \
                the average consumer a good idea of what the economy may look like \
                in the future. It could also give economists a picture of what to \
                look out for, and thus take any precautionary measures necessary.")

        st.subheader("Why VARIMA?")
        st.write("We ultimately chose to use VARIMA to make our model. This model \
                turned out to be accurate, efficient, and easy to explain.")

        st.subheader("Limiations")
        st.write("Our model is not robust to extreme worldly events. Sudden recessions \
                and booms have an impact the accuracy of our model. For example, \
                the pandemic lockdowns caused erratic changes in CPI and high levels \
                of inflation. External factors such as the Russia-Ukraine conflict also \
                introduce even more entropy to our forecasting.")


#Other Models Page
if st.sidebar.checkbox("Other Models"):
        st.title("Other Models")

        st.subheader("Prophet")
        df = pd.read_csv('Monthly_CPI_data_w_percent_change.csv', index_col = 0)
        all_categories = df.columns.values
        category = st.selectbox(
                'Choose a category',
                all_categories)

        cat = df.reset_index()
        cat = cat.loc[~df.index.str.contains("Percent Change"), [category, "index"]]
        cat["index"] = pd.to_datetime(cat["index"])
        cat.rename(columns={category:"y", "index":"ds"}, inplace = True)

        m = Prophet()
        model = m.fit(cat)

        future = m.make_future_dataframe(periods = 2, freq = 'y')
        cat_model = m.predict(future)

        st.write(m.plot(cat_model))

        se = (cat_model.loc[:, 'yhat'] - cat['y'])**2
        rmse = np.sqrt(np.mean(se))
        st.write("Root Mean Square Error: " + str(rmse))


        st.write("Despite the accuracy of this model, we decided against using \
                it because the actual implemenation of the program was entirely hidden. \
                We wanted to be able to describe the processes behind the model, and \
                this program would not have allowed us to do that.")


        st.subheader("")

        st.subheader("SARIMA")

        st.write("The reason we decided to not use SARIMA is beacuse it is very similar \
                to the model we ended up going with. We chose VARIMA because seasonality \
                (the 'S' in 'SARIMA') was already taken care of through differencing. We also \
                preferred the fact that VARIMA works by using multiple varibles, thus giving us a \
                more accurate forecast.")


        st.subheader("Lasso Regression")

        st.write('Unlike the previous models, the Lasso Regression cannot forecast values, \
                it can only predict a variable based on other index values. For this reason, \
                we decided against it as our final model because our goal was to forecast the \
                future values of CPI. However, it still provided useful information on how \
                indexes are related to one another. ')




#About Us Page
if st.sidebar.checkbox("About Us"):

        banner = Image.open("ds3_banner.PNG")
        st.image(banner, width = 700)

        st.title("About Us")

        st.write("We are a group of students from UCSD's Data Science Student Society. \
                As members of the CPI project team, we show passion, initiative, \
                and competence to contribute to this 2-quarter long project. Our team \
                has a diverse background but we are brought together by a common love \
                for data science.")

        st.title("Project Members")

        v_col1, v_col2 = st.columns(2)
        with v_col1:
            st.subheader("Project Lead: Victor Thai")
            st.write("Victor is a second-year at UC San Diego, majoring in Data Science \
                and minor in Cognitive Science. He is a member of CASA at UCSD and enjoys \
                weightlifting to relieve stress.")
        with v_col2:
                st.subheader("")
                v_img = Image.open("victor_headshot.JPEG")
                st.image(v_img, width = 200)

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.subheader("Michael Chen")
            st.write("Michael is a second-year student at UC San Diego, majoring in Data Science \
                and Minor in Management Science. Michael is also involved in the Finance Committee \
                in UTS at UCSD.")
        with m_col2:
                st.subheader("")
                m_img = Image.open("michael_headshot.JPEG")
                st.image(m_img, width = 200)

        s_col1, s_col2 = st.columns(2)
        with s_col1:
            st.subheader("Stephanie Chavez")
            st.write("Stephanie is a second-year student at UC San Diego and is majoring in Data Science. \
                Stephanie is a sister of Sigma Pi Alpha Sorority, Inc. and is a First-Gen Student Success \
                Peer Coach. Some of her interests outside of data science include musical theatre, \
                fantasy books, and dogs.")
        with s_col2:
            st.subheader("")
            steph_img = Image.open("Steph's Pic.jpg")
            st.image(steph_img, width = 200)


        z_col1, z_col2 = st.columns(2)
        with z_col1:
            st.subheader("Zed Siyed")
            st.write("Zed is a first-year student at UC San Diego, majoring in Computer Science. \
                Zed is also a member of Triton Consulting Group as the VP Tech Consulting. Some of \
                Zed's hobbies are sports analytics and weightlifting.")
        with z_col2:
            st.subheader("")
            zed_img = Image.open("zed_headshot.PNG")
            st.image(zed_img, width = 200)
