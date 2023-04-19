from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import module.constants as const
import module.util_functions as utf
import datetime
import time
import numpy as np
import pandas as pd

class SARIMAX_Model:
    
    def __init__(self, p=0, q=0, P=0, Q=0, diff_order=0, transform='', model_type='SARIMA'):
        self.ar_order = p                              # set autoregressive order
        self.ma_order = q                              # set moving average order
        self.diff_order = diff_order                   # set difference order
        self.P = P                                     # set AR order for seasonal component
        self.Q = Q                                     # set MA order for seasonal component
        self.m = 52                                    # set number of time periods in a year
        self.model_type = model_type                   # model type: ARIMA or SARIMA
        self.transform_method = transform              # set transformation method
        
        # set labels
        self.observe_label = const.STORE_OBSERVE       # set observe label
        self.target = const.TARGET.get(transform)      # set label for target variable based on transform method
        self.forecast_label = self.model_type + ' Forecast'
        self.forecast_train_label = self.model_type + ' Train'
        self.forecast_test_label = self.model_type + ' Test'
        
        # set model's name
        if model_type == 'SARIMA':
            self.model_name = model_type + '(' + str(p) + ', ' + str(diff_order) + ', ' + str(q) + ')(' + \
                              str(P) + ', 0,' + str(Q) + ')[' + str(self.m) + ']' 
        else:   # ARIMA model
            self.model_name = model_type + '(' + str(p) + ', ' + str(diff_order) + ', ' + str(q) + ')'
        
    def fit(self, X_train, Y_train):
        '''
        Train SARIMAX model and make in-sample forecasts.
        
        Parms:
          - X_train: SVD components
          - Y_train: train dataframe
        '''
        
        # record start time
        start = time.process_time()
        
        # duplicate original data
        data = Y_train.copy()

        if self.model_type == 'SARIMA':  # SARIMA + SVD components
            model = SARIMAX(data[self.target], exog=X_train,
                            order=(self.ar_order, self.diff_order, self.ma_order),
                            seasonal_order=(self.P, 0, self.Q, self.m),
                            simple_differencing=False).fit(disp=False)
        else: # ARIMA + SVD components
            model = ARIMA(data[self.target], exog=X_train, 
                          order=(self.ar_order, self.diff_order, self.ma_order)).fit()
        
        # make in-sample forecast and compute forecast error
        data[self.forecast_train_label] = utf.inverse_transform(model.fittedvalues, self.transform_method)
        data[const.ERROR_LABEL] = data[self.observe_label] - data[self.forecast_train_label]
        
        # check if residuals are correlated
        self.residual_info = utf.check_residuals(data)
        
        # save data and model
        self.data = data
        self.model = model
        
        # record end time
        end = time.process_time()
        self.train_time = (end - start) * 10**3    # compute model's training time in milli-second
       
    def predict(self, X, n_periods=52, forecast_label=None):
        '''
        Forecast sales for next n periods.
        
        Parms:
          - X: SVD components
          - n_periods: number of future periods for making sales forecast
          - forecast_label: label for a variable that contains forecast values
        '''
        
        forecast_label = self.forecast_test_label if forecast_label == None else forecast_label     # set forecast label
        future_dates = utf.get_future_dates(self.data.Date.iloc[-1], n_periods)                     # get dates for next n periods
        
        # make out-of-sample forecast
        forecast = self.model.get_prediction(start=self.data.shape[0], end=self.data.shape[0]+n_periods-1, exog=X) 
        yhat = utf.inverse_transform(forecast.predicted_mean, self.transform_method)
        
        # get confidence intervals
        yhat_conf_int = forecast.conf_int(alpha=0.05)
        lower = utf.inverse_transform(yhat_conf_int['lower ' + self.target].values, self.transform_method)
        upper = utf.inverse_transform(yhat_conf_int['upper ' + self.target].values, self.transform_method)

        # create and return forecast data frame
        return pd.DataFrame({'Date': future_dates, forecast_label: yhat, 'Lower Bound': lower, 'Upper Bound': upper})