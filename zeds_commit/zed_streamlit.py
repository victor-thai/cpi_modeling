# import packages and modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

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

x_train

# # Linear Regression

# %%
# create the linear regression environment
LinearRegression_model = LinearRegression()

# fit the data that we want to use in the model
LinearRegression_model.fit(x_train, y_train)

# predict using the X values to get Y predictions
pred_test_lr = LinearRegression_model.predict(x_test)

print("MSE is ", np.sqrt(mean_squared_error(y_test, pred_test_lr)))
print("r2 score is ", r2_score(y_test, pred_test_lr))

# %%
from sklearn.linear_model import Ridge

# creating ridge regression 
rr = Ridge(alpha=1)
# fit with testing data
rr.fit(x_train, y_train)
# predict using training data
pred_train_rr = rr.predict(x_train)

# predict using test data
pred_test_rr = rr.predict(x_test)

print("MSE is ", np.sqrt(mean_squared_error(y_test, pred_test_rr)))
print("r2 is ", r2_score(y_test, pred_test_rr))

# # Lasso Regression

#### Use cross validation to find the most optimal alpha value for the Lasso Regression based on a coordinate descent solver

# %%
from sklearn.linear_model import LassoCV

# Lasso with 5 fold cross-validation
lasso_cv = LassoCV(cv=5, random_state=0, max_iter=10000)

# Fit model
lasso_cv.fit(x_train, y_train)
optimal_alpha = lasso_cv.alpha_
optimal_alpha

# %% [markdown]
# #### Fit model with optimal alpha

# %%
from sklearn.linear_model import Lasso

# creating lasso model
model_lasso = Lasso(alpha=optimal_alpha)
model_lasso.fit(x_train, y_train)
pred_train_lasso = model_lasso.predict(x_train)

# generating predictions
pred_test_lasso = model_lasso.predict(x_test)

# scoring predications against test data
print("MSE is ", np.sqrt(mean_squared_error(y_test, pred_test_lasso)))
print("r2 is ", r2_score(y_test, pred_test_lasso))

# %% [markdown]
# #### Create dataframe with Energy predictions, set indexes to datetime

# %%
# assign predictions 
preds = y_test.assign(Lasso_Prediction=pred_test_lasso)
preds = preds.sort_index()

# convert indexes to datetime
preds.index = pd.to_datetime(preds.index, infer_datetime_format=True)
CPI.index = pd.to_datetime(CPI.index, infer_datetime_format=True)

preds

# CPI = CPI.merge(preds, left_index = True, right_index= True)
# CPI

# %% [markdown]
# #### Visualize dataset and prediction

# %%

fig, ax = plt.subplots(1, 1, sharex=False, sharey=True, figsize=(20, 10))
preds['Lasso_Prediction'].plot(legend=True)
CPI['Energy'].sort_index().plot(legend=True)

# formatting
fig.autofmt_xdate()
ax.xaxis.label.set_color('yellow')  # setting up X-axis label color to yellow
ax.yaxis.label.set_color('blue')  # setting up Y-axis label color to blue
ax.set_xlabel('Date')
ax.tick_params(axis='x', colors='red')  # setting up X-axis tick color to red
ax.tick_params(axis='y', colors='red')  # setting up Y-axis tick color to black

ax.spines['left'].set_color('red')  # setting up Y-axis tick color to red
ax.spines['top'].set_color('red')  # setting up above X-axis tick c
ax.set_title("Energy")
ax.title.set_color("red")
plt.show()

# %%
feature_df = pd.DataFrame({'category': model_lasso.feature_names_in_, 'weight': model_lasso.coef_},
                          columns=['category', 'weight'])
feature_df = feature_df.set_index('category').sort_values(by='weight', ascending=False)
feature_df = feature_df[feature_df.get('weight') != 0]
feature_df

# %%
fig, ax2 = plt.subplots(1, 1, sharex=False, sharey=True, figsize=(20, 10))
feature_df.plot(kind='bar', ax=ax2)

# formatting
fig.autofmt_xdate()
ax2.xaxis.label.set_color('yellow')  # setting up X-axis label color to yellow
ax2.yaxis.label.set_color('red')  # setting up Y-axis label color to blue
ax2.set_xlabel('Category')
ax2.set_ylabel('Weights')
ax2.tick_params(axis='x', colors='red')  # setting up X-axis tick color to red
ax2.tick_params(axis='y', colors='red')  # setting up Y-axis tick color to black
plt.axhline(y=0, color='r', linestyle='-')
ax2.spines['left'].set_color('red')  # setting up Y-axis tick color to red
ax2.spines['top'].set_color('red')  # setting up above X-axis tick c
ax2.set_title("Weight of Features")
ax2.title.set_color("red")
plt.show()

# %%
