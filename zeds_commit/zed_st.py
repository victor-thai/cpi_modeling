# import packages and modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
# from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import streamlit as st

# %%
CPI = pd.read_csv('/Users/zedsiyed/downloads/cpi_modeling/CPI_Data/Cleaned_CPI_data.csv')
PPI = pd.read_csv('/Users/zedsiyed/downloads/cpi_modeling/CPI_Data/PPI_data.csv')
CPI.index = pd.to_datetime(CPI.index, infer_datetime_format=True)
CPI = CPI.dropna(axis=1)

CPI = CPI[CPI['Unnamed: 0'].str.contains("Percent") == False]
CPI = CPI.set_index('Unnamed: 0')

merged = CPI

# %%
fig = merged['Energy'].plot(figsize=(20, 10), color='r')
fig.yaxis.label.set_color('blue')  # setting up Y-axis label color to blue

fig.tick_params(axis='x', colors='red')  # setting up X-axis tick color to red
fig.tick_params(axis='y', colors='red')  # setting up Y-axis tick color to black

fig.spines['left'].set_color('red')  # setting up Y-axis tick color to red
fig.spines['top'].set_color('red')  # setting up above X-axis tick c
fig.set_title("Energy").set_color('red')

# %%
merged = merged.replace('2012-01', 0)  # could not convert string to float: '2012-01'

merged = merged.apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)

# remove energy related categories
x = merged.drop(
    ['Household energy', 'Fuel oil', 'Fuel oil', 'Propane, kerosene, and firewood', 'Energy services', 'Electricity',
     'Utility ',
     'Fuel oil and other fuels', 'Motor fuel', 'Other motor fuels', 'Transportation commodities less motor fuel'
        , 'Energy', 'Energy commodities', 'Energy services', 'Fuels and utilities', 'Gasoline '], axis=1)

# separte the predicting attribute into Y for model training
y = merged.get(['Energy'])

# manually split test and training data
x_train = x.iloc[:96]
x_test = x.iloc[96:]
y_train = y.iloc[:96]
y_test = y.iloc[96:]

# # Lasso Regression

#### Use cross validation to find the most optimal alpha value for the Lasso Regression based on a coordinate descent solver

# %%
from sklearn.linear_model import LassoCV

# Lasso with 5 fold cross-validation
lasso_cv = LassoCV(cv=5, random_state=0, max_iter=10000)
# Fit model
lasso_cv.fit(x_train, y_train)
optimal_alpha = lasso_cv.alpha_

# streamlit code
st.header('Lasso Regression')
optimal_alpha = st.number_input('Input alpha value', min_value = 0.0, value = optimal_alpha)
# streamlit code




# %%
from sklearn.linear_model import Lasso
# creating lasso model
model_lasso = Lasso(alpha=optimal_alpha)
model_lasso.fit(x_train, y_train)
pred_train_lasso = model_lasso.predict(x_train)
# generating predictions
pred_test_lasso = model_lasso.predict(x_test)
# assign predictions 
preds = y_test.assign(Lasso_Prediction=pred_test_lasso)
preds = preds.sort_index()
# convert indexes to datetime
preds.index = pd.to_datetime(preds.index, infer_datetime_format=True)
CPI.index = pd.to_datetime(CPI.index, infer_datetime_format=True)

# streamlit code
fig, ax = plt.subplots(1, 1, sharex=False, sharey=True, figsize=(10, 7))
preds['Lasso_Prediction'].plot(legend=True)
CPI['Energy'].sort_index().plot(legend=True)

# formatting
fig.autofmt_xdate()
ax.set_xlabel('Date')
ax.set_title("Energy")
st.pyplot(fig)



# %%
feature_df = pd.DataFrame({'category': model_lasso.feature_names_in_, 'weight': model_lasso.coef_},
                          columns=['category', 'weight'])
feature_df = feature_df.set_index('category').sort_values(by='weight', ascending=False)
feature_df = feature_df[feature_df.get('weight') != 0]
st.subheader('Feature Weights Dataframe')
feature_df = feature_df.sort_values(by = 'weight', ascending = False)
feature_df

# %%
st.subheader('Feature Weights')
st.bar_chart(data = feature_df, width = 10, height = 600)
fig, ax2 = plt.subplots(1, 1, sharex=False, sharey=True, figsize=(20, 50))
feature_df.plot(kind='bar', ax=ax2)

# formatting
fig.autofmt_xdate()
ax2.set_title("Weight of Features")
ax2.title.set_color("red")
plt.show()

# %%
