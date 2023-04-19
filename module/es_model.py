import module.constants as const
import module.util_functions as utf
import time
import numpy as np
import pandas as pd

class ExponentialSmoothing:
    
    def __init__(self, smoothing_constant=0.5, transform='', initial_level=None):
        self.alpha = smoothing_constant                 # set smoothing constant
        self.transform_method = transform               # set transformation method
        self.level_0 = initial_level                    # set initial level
        
        # set labels
        self.observe_label = const.STORE_OBSERVE        # set observe label
        self.target = const.TARGET.get(transform)       # set label for target variable based on transform method
        self.forecast_train_label = 'ES Train'
        self.forecast_test_label = 'ES Test'
        self.forecast_label = 'ES Forecast'
    
    def compute_smoothed_series(self, Y):
        '''
        Compute exponentially smoothed series.
        '''
        
        # create a list of exponetial smoothed series
        E = [self.level_0]
        
        # compute exponential smoothed series
        for y_t in Y.values:
            # compute the level at time period t using current observe value and previous level
            # l_t = alpha * y_t + (1 - alpha) * l_(t-1)
            E.append(self.alpha * y_t + (1 - self.alpha) * E[-1])
        
        self.exponential_series = E  # save smoothed series
    
    def fit(self, ts_data: pd.DataFrame):
        '''
        Train the model.
        
        Parm:
          - ts_data: a time series data frame.
        '''
        
        # record start time
        start = time.process_time()
        
        # duplicate original data
        data = ts_data.copy()
        
        # set initial level and compute exponential smoothed series
        self.compute_initial_level(data)                   
        self.compute_smoothed_series(data[self.target]) 
        
        # compute in-sample forecasts
        # y_hat_t = E[t-1]
        data[self.forecast_train_label] = utf.inverse_transform(pd.Series(self.exponential_series[:-1]), self.transform_method)
        data[const.ERROR_LABEL] = data[self.observe_label] - data[self.forecast_train_label]
        
        # compute standard error
        self.standard_error = np.sqrt(sum(data[const.ERROR_LABEL]**2) / (data.shape[0] - 1))
        
        # record end time
        end = time.process_time()
        
        # save data
        self.data = data
        self.train_time = (end - start) * 10**3    # compute model's training time in milli-second
        
    
    def compute_initial_level(self, data):
        '''
        Compute an initial level at time period 0 if an initial level is not provided.
        '''
        
        if self.level_0 is None:
            week_1 = data['Week'].values[0]                                  # get the week of first observation
            week_0 = week_1 - 1 if week_1 > 1 else 52                        # set week_0 to the previous week of week_1
            self.level_0 = data[data.Week == week_0][self.target].mean()     # average observed values at week_0
    
    def predict(self, n_periods: int):
        '''
        Compute 95% prediction intervals.
        '''
        
        forecast_dates = utf.get_future_dates(self.data.Date.iloc[-1], n_periods)
        lower = []   # forecast's lower bound
        upper = []   # forecast's upper bound
        
        # get the last value of smoothed series
        level = utf.inverse_transform(self.exponential_series[-1], self.transform_method)    
        conf_interval = const.Z.get('.025') * self.standard_error
        
        # compute confident intervals
        for n in range(1, n_periods + 1):
            if n == 1:
                lower.append(level - conf_interval)
                upper.append(level + conf_interval)
            elif n == 2:
                lower.append(level - conf_interval * np.sqrt(1 + self.alpha**2))
                upper.append(level + conf_interval * np.sqrt(1 + self.alpha**2))
            else:
                lower.append(level - conf_interval * np.sqrt(1 + (n - 1) * self.alpha**2))
                upper.append(level + conf_interval * np.sqrt(1 + (n - 1) * self.alpha**2))
        
        if n_periods > 1:
            # n periods ahead forecasts
            return pd.DataFrame({'Date': forecast_dates, self.forecast_label: [level]*n_periods, 
                                 'Lower Bound': lower, 'Upper Bound': upper})
        else:
            # one period ahead forecast
            return forecast_dates[0], level, lower[0], upper[0]