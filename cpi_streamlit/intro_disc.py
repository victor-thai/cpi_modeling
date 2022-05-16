import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from prophet import Prophet
from PIL import Image
from statsmodels.tsa.statespace.varmax import VARMAX
from timeit import default_timer as timer
from sklearn import metrics
from pmdarima import auto_arima
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
if st.sidebar.button("Intro"):

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
st.sidebar.button("Data Cleaning and EDA")


#Model Page
if st.sidebar.button("Model"):

        st.title("Model")

        #import the data that we need for model
        cpi = pd.read_csv('cpi_w_gold_oil.csv', index_col = 0)
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


#Discussion Page
if st.sidebar.button("Discussion"):

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
if st.sidebar.button("Other Models"):
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

        '''
        CPI = pd.read_csv("Merged_CPI.csv")
        CPI.set_index('Unnamed: 0', inplace = True)
        CPI.index = pd.to_datetime(CPI.index, format = '%Y%m')
        CPI.index = pd.DatetimeIndex(CPI.index.values,
                                       freq=CPI.index.inferred_freq)
        X = CPI[cat]
        X = X.dropna()
        X = X.diff().dropna()
        train_all = X[:-15]
        test_all = X[-15:]

        p = d = q = range(0, 2)
        pdq = list(itertools.product(p, d, q))
        pdq_s = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
        param_model = []
        param_s_model = []
        AIC_model = []
        model_select = pd.DataFrame()
        for param in pdq:
            for param_s in pdq_s:
                model = sm.tsa.statespace.SARIMAX(train_all,order=param,seasonal_order=param_s,enforce_stationarity=False,enforce_invertibility=False)
                model = model.fit()
                print('ARIMA{}x{} - AIC:{}'.format(param, param_s, model.aic))
                param_model.append(param)
                param_s_model.append(param_s)
                AIC_model.append(model.aic)
        model_select['pdq'] = param_model
        model_select['pdq_x_PDQs'] = param_s_model
        model_select['aic'] = AIC_model

        pred_all = model_all.predict(start=0,end=train_all.shape[0],typ='levels').rename('SARIMAX predictions')
        fig_a, ax_a = plt.subplots(1,1, figsize=(15,15))
        X.plot(legend = True, ax = ax_a)
        pred_all.plot(legend=True, ax=ax_a)

        pred_f_a = model_all.predict(start=train_all.shape[0]-1,end=train_all.shape[0]+10,typ='levels').rename('SARIMAX predictions')
        st.write(pred_f_a.plot())
        '''

        st.write("The reason we decided to not use this model is beacuse it is very similar \
                to the model we ended up going with. We chose VARIMA because seasonality \
                (the 'S' in 'SARIMA') was already taken care of through differencing. We also \
                prefered the fact that VARIMA works by using multiple varibles, thus giving us a \
                more accurate forecast.")





#About Us Page
if st.sidebar.button("About Us"):

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

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.subheader("Michael Chen")
            st.write("Michael is a second-year student at UC San Diego, majoring in Data Science \
                and Minor in Management Science. Michael is also involved in the Finance Committee \
                in UTS at UCSD.")

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
