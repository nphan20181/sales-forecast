import numpy as np

# set directories
DATA_DIR = 'data'
RESULTS_DIR = 'results'

TEST_SIZE = 52
CURRENCY_UNIT = 1000000

# set labels
STORE_OBSERVE = 'Weekly Sales (Million)'
DEPT_OBSERVE = 'Weekly_Sales'
X_LABEL ='Date'
T_LABEL = 'Time Period'
ERROR_LABEL = 'Residual'

# name of target variable according to transformation method
TARGET = {'':'Weekly Sales (Million)', 
          'Log': 'Log of Weekly Sales (Million)',}


# set selected features
FEATURES = ['Scaled_Week', 'Super Bowl', 'Labor Day', 'Thanksgiving', 'Before Christmas', 'Christmas']

# name of evaluation metrics
METRIC_NAMES = {'AIC': 'Akaike Information Criterion',
                'MAPE': 'Mean Absolute Percentage Error',
                'MAD': 'Mean Absolute Deviation',
                'RMSE': 'Root Mean Square Error',
                'R-Squared': 'R-Squared'}

# transformation function
TRANSFORM = {'Log': np.log, 'Sqrt': np.sqrt}
REVERSE_TRANSFORM = {'Log': np.exp, 'Sqrt': np.square}
TRANSFORM_LABEL = {'': 'None', 'Log': 'Logarithm', 'Sqrt': 'Square Root'}

NUMBER_ORDER = {0: 'Zeroth', 1: 'First', 2: 'Second', 3: 'Third', 4: 'Fourth', 5: 'Fifth', 
                6: 'Sixth', 7: 'Seventh', 8: 'Eighth', 9: 'Ninth', 10: 'Tenth'}

# set figure's font color
FIG_FONT_COLOR = '#000066';

# z-value for computing 95% confidence interval
Z = {'.025': 1.96}