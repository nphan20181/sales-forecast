from module.es_model import ExponentialSmoothing
from module.holt_trend_es import Holt
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import module.constants as const
import numpy as np
import pandas as pd
import datetime
import os

##################################################################################################
# Start functions for HOLIDAY NAME

# create a data frame of holidays
holidays = pd.DataFrame(dict({'Super Bowl': ['2010-02-12', '2011-02-11', '2012-02-10', '2012-02-08'],
                              'Labor Day': ['2010-09-10', '2011-09-09', '2012-09-07', '2013-09-06'],
                              'Thanksgiving': ['2010-11-26', '2011-11-25', '2012-11-23', '2013-11-29'],
                              'Christmas': ['2010-12-31', '2011-12-30', '2012-12-28', '2013-12-27']}))

# change data type for each column to datetime
for col_name in holidays.columns:
    holidays[col_name] = pd.to_datetime(holidays[col_name])

def get_holiday_name(date):
    '''
    Return holiday's name if holiday falls within a given week; otherwise, return empty string.
    '''
    
    global holidays
    
    # iterate through a list of holiday
    for col_name in holidays.columns:
        # return holiday name if holiday is within the given week
        if len(holidays[holidays[col_name] == date]) > 0:
            return col_name
    
    # return empty string if holiday is not in the given week
    return ''

# End functions for HOLIDAY NAME
##################################################################################################


##################################################################################################
# Start functions for DATA RETRIEVAL

def load_ts_data(file_name) -> pd.DataFrame:
    '''
    Load time series data and change data type of "Date" from string to date.
    
    '''
    
    file_path = os.path.join('data', file_name)         # get file location
    ts_data = pd.read_csv(file_path)                    # load time series data
    ts_data['Date'] = pd.to_datetime(ts_data['Date'])   # change data type of Date from string to date.
    
    return ts_data


def get_store_sales(store_sales, store_number) -> pd.DataFrame:
    '''
    Retrieve and return sales data for a given store. Sort data by time period.
    '''
    
    data = store_sales[store_sales.Store == store_number].copy()     # retrieve store's weekly sales
    data = data.sort_values(const.T_LABEL)                           # sort data by time period in ascending order
    data.reset_index(inplace=True, drop=True)
    
    return data
    
# End functions for DATA RETRIEVAL
##################################################################################################


##################################################################################################
# Start functions for DATA TRANSFORMATION

def transform_data(ts_data, transform_type=''):
    '''
    Transform time series data.
    
    Parms:
      - ts_data: time series data
      - transform_type: type of transformation, e.g. log (Logarithm) or sqrt (Square Root)
    '''
    
    return ts_data if transform_type == '' else const.TRANSFORM.get(transform_type)(ts_data)


def inverse_transform(ts_data, transform_type=''):
    '''
    Convert transformed values back to original unit.
    '''
    
    return ts_data if transform_type == '' else const.REVERSE_TRANSFORM.get(transform_type)(ts_data)

# End functions for DATA TRANSFORMATION
##################################################################################################


##################################################################################################
# Start AUGMENTED DICKEY-FULLTER (ADF) test for stationary time series
#
# Hypothesis:
#   H_0: Unit root is present. Time series is not stationary.
#   H_a: Unit root is not present. Time series is stationary.
#
# ADF statistic should be a large negative value.
# Reject H_0 if p-value < 0.05

def adf_test(df_series):
    '''
    Perform Augmented Dickey–Fuller test to check whether time series data is stationary or not.
    '''
    
    # perform ADF test using statsmodels
    ADF_result = adfuller(df_series, maxlag=52)
    
    # return ADF Statistic and p-value
    return np.round(ADF_result[0], 2), np.round(ADF_result[1], 8)


def get_diff_order(df_series) -> int:
    '''
    Search and return a difference order that obtains a stationary time series.
    '''
    
    difference_order = 0       # set initial difference order to 0
    p_value = 0.6              # set initial p-value > 0.5
    
    # search for a difference order by running ADF test on the differenced series until p_value < 0.5
    while p_value >= 0.50:
        df_diff = np.diff(df_series, n=difference_order)    # transform data by differencing the series
        _, p_value = adf_test(df_diff)                      # perform ADF test on the differenced series
        difference_order += 1                               # increment difference order
        
    return difference_order

# End ADF test
##################################################################################################


##################################################################################################
# Start LJUNG-BOX test for independent and un-correlated residual
#
# Hypothesis:
#   H_0: There is no autocorrelation in the data.
#   H_a: There exists a significant autocorrelation.

def check_residuals(data):
    '''
    Check if residuals are independent and uncorrelated.
    '''

    # perform LJung-Box test for uncorrelated residuals, p_value should be > 0.05
    lb_test = acorr_ljungbox(data[const.ERROR_LABEL].values, lags=np.arange(1, 53, 1), return_df=True)
        
    # check if resdisuals are correlated
    for p_value in lb_test['lb_pvalue'].values:
        if p_value <= 0.05:
            return 'Residuals are correlated.'
            
    return 'Residuals are independent and uncorrelated.'

# End Ljung-Box test
##################################################################################################



##################################################################################################
# Start functions for making SALES FORECAST

def get_future_dates(initial_date, periods=52) -> list:
    '''
    Build and return a list of future dates based on initial_date.
    '''
   
    current_date = initial_date  # set the start date for computation
    future_dates = []            # a list of future dates
    
    for i in range(0, periods):
        current_date = current_date + datetime.timedelta(days=7)   # get the date of next 7 days
        future_dates.append(current_date)
    
    return future_dates


def make_forecast(model, data):
    '''
    Make rolling forecast on Test data.
    '''
    
    forecast_dates = []      # a list of sales date
    forecasts = []           # a list of predicted sales values
    lower_list = []          # a list of lower limits
    upper_list = []          # a list of upper limits

    # increase the size of train data by one
    # then re-fit the model and make sales forecast one period ahead
    for window in range(const.TEST_SIZE, 0, -1):
        train = data.iloc[:-window]         # get train data
        model.fit(train)                    # fit a model using train data
        
        # predict next sales value and save the result
        date, y_hat, lower, upper = model.predict(n_periods=1)
        forecast_dates.append(date)
        forecasts.append(y_hat)
        lower_list.append(lower)
        upper_list.append(upper)
    
    return forecast_dates, forecasts, lower_list, upper_list


def make_rolling_forecast(model, data, forecast_label) -> pd.DataFrame:
    '''
    Make rolling forecast on Test data and return the forecast dataframe.
    '''
    
    forecast_dates, y_hats, lower, upper = make_forecast(model, data)
    
    return pd.DataFrame({'Date': forecast_dates, forecast_label: y_hats, 
                         'Lower Bound': lower, 'Upper Bound': upper})

# End SALES FORECAST
##################################################################################################



##################################################################################################
# Start functions for MODEL SELECTION

def compute_metrics(actual, predict, unit=1):
    '''
    Compute MAPE, MAD, RMSE, R-Squared and return the scores.
    
    Parms:
      - actual: observe values
      - predict: predicted values
      - unit: current unit
    '''
    
    n = len(actual)
    deviations = np.abs(actual - predict)                             # compute absolute error: actual - predict
    mape = np.round(np.mean((deviations / actual)) * 100, 2)          # compute mean absolute percentage error
    mad = np.round(sum(deviations) / n, 6) * unit                     # compute mean absolute deviation
    rmse = np.round(np.sqrt(sum(deviations**2) / n), 6) * unit        # compute root mean square error
    
    return mape, mad, rmse, np.round(r2_score(actual, predict) * 100, 2)


def rank_model_results(metrics, metric_name='R-Squared', model_name='') -> pd.DataFrame:
    '''
    Rank all models store-wide based on metric_name, ONLY for ONE model type (e.g. ARIMA or SARIMA, but NOT both).
    '''
    
    results = None     # a dataframe containing result of optimal models
    
    # sort data in descending order if metric is R-Squared; otherwise, sort in ascending order
    sort_asc = False if metric_name == 'R-Squared' else True

    # select an optimal model for each store based on metric_name
    for store_num in metrics['Store'].unique():
        if model_name == 'ARIMA' or model_name == 'SARIMA':
            # for ARIMA/SARIMA models, make sure residuals are uncorrelated
            best_result = metrics.loc[((metrics['Store'] == store_num) & 
                                       (metrics['Residual Info'] != 'Residuals are correlated.'))].sort_values(metric_name,
                                                                                                               ascending=sort_asc).head(1)
        else: # all other models
            best_result = metrics[metrics['Store'] == store_num].sort_values(metric_name, ascending=sort_asc).head(1)
            
        # concatenate results dataframes
        results = best_result if results is None else pd.concat([results, best_result], axis=0)
  
    # rank models store-wide based on selected metric
    results = results.sort_values(metric_name, ascending=sort_asc).reset_index(drop=True)
    results['Rank'] = results.index + 1
    
    return results


def concat_metrics():
    '''
    Load and combine all metric files. Return a dataframe containing evaluation results for all models.
    '''
    
    file_list=['es_metrics.csv','holt_metrics.csv','hw_metrics.csv', 'arima_metrics.csv', 'sarima_metrics.csv']
    metrics = pd.DataFrame(pd.read_csv(os.path.join('results', file_list[0])))

    # load and combine all metrics files
    for i in range(1,len(file_list)):
        data = pd.read_csv(os.path.join('results', file_list[i]))
        df = pd.DataFrame(data)
        metrics = pd.concat([metrics, df],axis=0)
    
    metrics.reset_index(drop=True, inplace=True)
    
    return metrics

def select_models(metrics, metric_name='RMSE', rank_by='R-Squared'):
    '''
    Select an optimal model for each store based on metric_name, than rank the models based on rank_by.
    '''
    
    best_models = None
    sort_asc =  True if metric_name != 'R-Squared' else False

    # set parameter list
    if metric_name != 'R-Squared':
        parms = [metric_name, 'Alpha', 'Gamma', 'Delta', 'AR Order', 'MA Order', 'Diff Order', 'SVD', 
                 'Seasonal AR Order', 'Seasonal MA Order']
    else:
        parms = [metric_name]
    
    # select an optimal model for each store based on metric_name
    for store_num in metrics.Store.unique():
        df = metrics.loc[(metrics.Store == store_num) & 
                         (metrics['Residual Info'] != 'Residuals are correlated.')].sort_values(parms, 
                                                                                                ascending=sort_asc).head(1)

        # concatenate the results
        if best_models is None:
            best_models = df
        else:
            best_models = pd.concat([best_models, df], axis=0)

            
    best_models.reset_index(drop=True, inplace=True)
    
    # rank models based on rank_by
    rank_asc =  True if rank_by != 'R-Squared' else False     # set sort order: ascending/descending
    best_models = best_models.sort_values(rank_by, ascending=rank_asc).reset_index(drop=True)
    best_models['Rank'] = best_models.index + 1
    
    return best_models

# End MODEL SELECTION functions
##################################################################################################


##################################################################################################
# Start LINEAR REGRESSION

def regression_estimate(data, target):
    '''
    Fit a least squares regression line and return the intercept and coefficient values.
    '''
    
    X = data[const.T_LABEL].to_numpy().reshape(-1, 1)     # get time periods
    Y = data[target]                                      # get target values
        
    # fit a linear regression model
    lr_model = LinearRegression()
    lr_model.fit(X, Y)
    
    return lr_model.intercept_, lr_model.coef_[0]


# End LINEAR REGRESSION
##################################################################################################