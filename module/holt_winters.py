import module.constants as const
import module.util_functions as utf
from sklearn.linear_model import LinearRegression
import datetime
import time
import numpy as np
import pandas as pd

class HoltWinters:
    
    def __init__(self, alpha=0.5, gamma=0.5, delta=0.5, transform='', preload=None):
        self.alpha = alpha                             # set smoothing constant for level
        self.gamma = gamma                             # set smoothing constant for growth rate
        self.delta = delta                             # set smoothing constant for seasonal component
        self.transform_method = transform              # set transformation method
        self.P = 52                                    # set number of time periods per year
        self.preload = preload                         # use some results from preload model if not None to reduce training time
        
        # set labels
        self.observe_label = const.STORE_OBSERVE            # set observe label
        self.target = const.TARGET.get(transform)           # set label for target variable based on transform method
        self.forecast_label = 'Holt-Winters Forecast' 
        self.forecast_train_label = 'Holt-Winters Train'
        self.forecast_test_label = 'Holt-Winters Test'
        self.model_name = 'Holt-Winters(' + str(alpha) + ', ' + str(gamma) + ', ' + str(delta) + ')'
        
    def fit(self, ts_data: pd.DataFrame):
        '''
        Compute model's components and make in-sample forecasts.
        '''
        
        # record start time
        start = time.process_time()
        
        self.data = ts_data.copy()        # duplicate original data
        self.compute_seasonal_factor()    # compute seasonal factor  
        self.compute_components()         # compute model's components and make in-sample forecasts
        
        # compute train's error
        self.data[const.ERROR_LABEL] = self.data[self.observe_label] - self.data[self.forecast_train_label]
        
        # record end time
        end = time.process_time()
        self.train_time = (end - start) * 10**3    # compute model's training time in milli-second

    
    def compute_components(self):
        '''
        Compute model's components and make in-sample forecasts.
        '''
        
        level = [self.level_0]               # a list of levels
        growth_rate = [self.growth_rate_0]   # a list of growth rates
        predictions = []                     # a list of in-sample forecasts
        sre = []                             # a list of squared relative error
        
        for i, y_t in enumerate(self.data[self.target].tolist()):
            t = i + 1    # set current time period
            
            # get seasonal index at time period t - 52
            seasonal_index = self.seasonal_factor[self.seasonal_factor['t'] == t - self.P]['Detrended'].tolist()[0]
            
            # compute level at time period t
            level.append(self.alpha * (y_t / seasonal_index) + (1 - self.alpha) * (level[i] + growth_rate[i]))
            
            # compute growth rate at time period t
            growth_rate.append(self.gamma * (level[t] - level[t-1]) + (1 - self.gamma) * growth_rate[i])
            
            # compute seasonal factor at time period t
            sn_t = self.delta * (y_t / level[t]) + (1 - self.delta) * seasonal_index
            self.seasonal_factor = pd.concat([self.seasonal_factor, pd.DataFrame({'t': [t], 'Detrended': [sn_t]})], axis=0)
            
            # make in-sample prediction
            forecast = utf.inverse_transform(((level[i] + growth_rate[i]) * seasonal_index), self.transform_method)
            predictions.append(forecast)
            
            # compute squared relative error
            error = ((y_t - (level[i] + growth_rate[i]) * seasonal_index) / ((level[i] + growth_rate[i]) * seasonal_index))**2
            sre.append(error)
        
        # save level, growth rate, predictions, and squared relative error
        self.level = level
        self.growth_rate = growth_rate
        self.data[self.forecast_train_label] = predictions
        self.data['Squared Relative Error'] = sre
        
        # compute standard error
        self.standard_error = np.sqrt(sum(sre) / (self.data[const.T_LABEL].values[-1] - 3))
        
    def compute_seasonal_factor(self):
        '''
        Fit a regression line to obtain initial value for level and growth rate and compute initial seasonal factors.
        '''
        
        if self.preload is None:
            # fit a least squares regression line to obtain intial level and initial growth rate
            self.level_0, self.growth_rate_0 = utf.regression_estimate(self.data, self.target)
        else:
            # get initial level and initial growth rate from preload model to reduce training time
            self.level_0 = self.preload.level_0
            self.growth_rate_0 = self.preload.growth_rate_0
        
        # compute initial seasonal factors
        self.data['Regression Estimates'] = self.data[const.T_LABEL].apply(lambda x: self.level_0 + x*self.growth_rate_0)
        self.data['Detrended'] = self.data[self.target] / self.data['Regression Estimates']    # detrend the data
        seasonal_factor = self.data.groupby(['Week'])['Detrended'].mean().reset_index()        # compute average Detrend per week
        
        # compute seasonal correction factor and update Detrended or Seasonal Factors
        self.correction_factor = self.P / sum(seasonal_factor['Detrended'])
        seasonal_factor['Detrended'] = self.correction_factor * seasonal_factor['Detrended']
        
        # set time periods of initial factors to the past
        seasonal_factor['t'] = seasonal_factor['Week'] - 52                                    
        del seasonal_factor['Week']
        
        # save initial seasonal factors
        self.seasonal_factor = seasonal_factor
        
    def predict_intervals(self, forecast, n, seasonal_factor):
        '''
        Compute 95% prediction intervals.
        '''
        
        # get the last value of level and growth rate
        level = self.level[-1]
        growth_rate = self.growth_rate[-1]
        
        # compute critical value c
        if n == 1:
            c = (level + growth_rate)**2
        elif n >= 2 and n <= self.P:
            c = sum([((self.alpha**2) * (1 + (n-j) * self.gamma )**2) * ((level + j * growth_rate)**2) + \
                 (level + n * growth_rate)**2 for j in range(1, n)])
        else:
            c = 1
        
        # compute confidence interval
        ci = const.Z.get('.025') * self.standard_error * np.sqrt(c) * seasonal_factor
        
        # compute lower/upper limit
        lower = utf.inverse_transform(forecast - ci, self.transform_method)
        upper = utf.inverse_transform(forecast + ci, self.transform_method)
        
        return lower, upper
    
    def predict(self, n_periods=52):
        '''
        Make out-of-sample forecast for next n periods.
        '''
        
        predictions = []    # a list of out-of-sample forecasts
        lower = []          # a list of lower limits 
        upper = []          # a list of upper limits
        
        # get a list of future dates based on the last Date in the data
        future_dates = utf.get_future_dates(self.data.Date.iloc[-1])

        # get the last time period
        t = self.data[const.T_LABEL].iloc[-1]
        
        for i in range(0, n_periods):
            # get seasonal factor at time period t - 52
            t = t + 1
            seasonal_index = self.seasonal_factor[self.seasonal_factor['t'] == t - self.P]['Detrended'].tolist()[0]
            
            # compute point forecast
            point_forecast = (self.level[-1] + (i + 1) * self.growth_rate[-1]) * seasonal_index
            predictions.append(utf.inverse_transform(point_forecast, self.transform_method))
            
            # get prediction interval
            low, high = self.predict_intervals(point_forecast, i + 1, seasonal_index)
            lower.append(low)
            upper.append(high)
        
        # create and return forecast data frame
        return pd.DataFrame({'Date': future_dates, self.forecast_test_label: predictions, 'Lower Bound': lower, 'Upper Bound': upper})