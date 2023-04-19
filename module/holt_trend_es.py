import module.constants as const
import module.util_functions as utf
import time
import numpy as np
import pandas as pd

class Holt:
    
    def __init__(self, alpha=0.5, gamma=0.5, transform='', preload=None):
        self.alpha = alpha                          # set smoothing constant for level
        self.gamma = gamma                          # set smoothing constant for growth rate
        self.transform_method = transform           # set transformation method
        self.preload = preload                      # set preload model
        self.level_0 = None                         # set initial level
        self.growth_rate_0 = None                   # set initial growth rate
        
        # set labels
        self.observe_label = const.STORE_OBSERVE          # set observe label
        self.target = const.TARGET.get(transform)         # set label for target variable based on transform method
        self.forecast_label = "Holt Forecast"
        self.forecast_train_label = 'Holt Train'
        self.forecast_test_label = 'Holt Test'
        self.model_name = "Holt's Linear Trend"
        
    def fit(self, ts_data: pd.DataFrame):
        '''
        Compute model's components and make in-sample forecasts.
        '''
        
        # record start time
        start = time.process_time()
        
        # duplicate original data
        self.data = ts_data.copy()
        
        # compute inital level and growth rate   
        if self.preload is not None:
            # get initial level and initial growth rate from preload model
            self.level_0 = self.preload.level_0
            self.growth_rate_0 = self.preload.growth_rate_0
        elif self.level_0 is None or self.growth_rate_0 is None:
            # fit a least squares regression line to obtain intial level and initial growth rate
            self.level_0, self.growth_rate_0 = utf.regression_estimate(self.data, self.target)  
        
        # compute Levels, Growth Rates and in-sample forecasts
        self.compute_components()
        
        # record end time
        end = time.process_time()
        self.train_time = (end - start) * 10**3    # compute model's training time in milli-second

    
    def compute_components(self):
        '''
        Compute model's components and make in-sample forecast.
        '''
        
        level = [self.level_0]                # a list of levels (or means)
        growth_rate = [self.growth_rate_0]    # a list of growth rates
        predictions = []                      # a list of in-sample forecasts
        
        # compute level, growth rate and in-sample forecast
        # i = previous time period, and t = current time period
        for i, y_t in enumerate(self.data[self.target].tolist()):
            # set current time period
            t = i + 1
            
            # compute level at time period t
            level.append(self.alpha * y_t + (1 - self.alpha) * (level[i] + growth_rate[i]))
            
            # compute growth rate at time period t
            growth_rate.append(self.gamma * (level[t] - level[i]) + (1 - self.gamma) * growth_rate[i])
            
            # make in-sample forecast
            predictions.append(level[i] + growth_rate[i])
        
        # save level and growth rate
        self.level = level
        self.growth_rate = growth_rate
        
        # inverse transformation and save sales predictions
        self.data[self.forecast_train_label] = utf.inverse_transform(predictions, self.transform_method)
        
        # compute forecast's error
        self.data[const.ERROR_LABEL] = self.data[self.observe_label] - self.data[self.forecast_train_label]
        
        # compute standard error
        self.standard_error = np.sqrt(sum((self.data[self.target] - predictions)**2) / (self.data[const.T_LABEL].values[-1] - 2))
        
    def predict_interval(self, forecast, n):
        '''
        Compute 95% prediction interval of forecast value.
        '''
        
        # compute confidence interval
        ci = const.Z.get('.025') * self.standard_error
        if n >= 2:
            ci = ci * np.sqrt(1 + sum([(self.alpha**2) * (1 + j * self.gamma)**2 for j in range(1, n)]))
        
        # compute upper bound value
        upper = utf.inverse_transform(forecast + ci, self.transform_method) 
        
        # compute lower bound value
        lower = utf.inverse_transform(forecast - ci, self.transform_method)
        
        return lower, upper
    
    def predict(self, n_periods=52):
        '''
        Make out-of-sample forecast for next n periods.
        '''

        predictions = []    # a list of out-of-sample forecasts
        lower = []          # a list of lower limits 
        upper = []          # a list of upper limits
        forecast_dates = utf.get_future_dates(self.data.Date.iloc[-1], n_periods)
        
        # forecast sale values for the next n weeks
        for i in range(0, n_periods):
            # compute out-of-sample point forecast
            point_forecast = self.level[-1] + (i + 1) * self.growth_rate[-1]
            predictions.append(utf.inverse_transform(point_forecast, self.transform_method))
            
            # compute lower/upper bound of point forecast
            low, high = self.predict_interval(point_forecast, i + 1)
            lower.append(low)
            upper.append(high)
        
        
        if n_periods > 1:
            # n periods ahead forecasts
            return pd.DataFrame({'Date': forecast_dates, self.forecast_label: predictions, 'Lower Bound': lower, 'Upper Bound': upper})
        else:
            # one period ahead forecast
            return forecast_dates[0], predictions[0], lower[0], upper[0]