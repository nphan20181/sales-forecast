from module.sarima_model import SARIMAX_Model
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
import module.util_functions as utf
import module.constants as const
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


store_sales = utf.load_ts_data('store_sales.csv')         # get sales data for all stores
svd_results = pd.read_csv('results/svd_results.csv')      # get optimal number of SVD components per store

transform_method = 'Log'       # use "Log of Weekly Sales (Million)" as target variable
ar_list = []                   # a list of AR order
ma_list = []                   # a list of MA order
diff_list = []                 # a list of difference order
sar_list = []                  # a list of seasonal AR order
sma_list = []                  # a list of seasonal MA order
store_list = []                # a list of stores
mad_list = []                  # a list of MAD
mape_list = []                 # a list of MAPE
rmse_list = []                 # a list of RMSE
r_squared_list = []            # a list of R-Squared
aic_list = []                  # a list of AIC
train_times = []               # a list of model's training time
svd_list = []                  # a list of number of SVD components
residual_info = []             # a list of residual information

# iterate through every store
for store_num in store_sales.Store.unique():
    print('Store: %d' % store_num)
    
    # get sales data for a given store, then set train/test data
    data = utf.get_store_sales(store_sales, store_num)
    train = data[:-const.TEST_SIZE]
    test = data.tail(const.TEST_SIZE)
    
    diff_order = utf.get_diff_order(data[const.TARGET.get(transform_method)])        # get difference order
    n_components = svd_results[svd_results.Store == store_num]['SVD'].values[0]      # get optimal number of components
    
    # perform dimensionality reduction
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_train = svd.fit_transform(train[const.FEATURES])
    X_test = svd.transform(test[const.FEATURES])
    
    # evaluate model for different combinations of AR and MA orders
    for ar_order in tqdm(range(0, 13)):
        for ma_order in range(0, 13):
            # train ARIMA+SVD model and make sales forecast for the next 52 weeks
            model = SARIMAX_Model(p=ar_order, q=ma_order, diff_order=diff_order, 
                                  transform=transform_method, model_type='ARIMA')
            model.fit(X_train, train)
            forecasts = model.predict(X_test)[model.forecast_test_label].values 

            # evaluate model and save result
            mape, mad, rmse, r_squared = utf.compute_metrics(test[const.STORE_OBSERVE].values, forecasts)
            mape_list.append(mape)
            mad_list.append(mad)
            rmse_list.append(rmse)
            r_squared_list.append(r_squared)
            aic_list.append(model.model.aic)
            train_times.append(model.train_time)
            ar_list.append(ar_order)
            ma_list.append(ma_order)
            diff_list.append(diff_order)
            svd_list.append(n_components)
            residual_info.append(model.residual_info)
            store_list.append(store_num)

            
# create metric dataframe and save as a csv file
metrics = pd.DataFrame({'Store': store_list, 
                        'AR Order': ar_list, 'MA Order': ma_list, 'Diff Order': diff_list,
                        'MAPE': mape_list, 'MAD': mad_list, 'RMSE': rmse_list, 'R-Squared': r_squared_list,
                        'AIC': aic_list, 'SVD': svd_list, 'Residual Info': residual_info,
                        'Train Time': train_times})
metrics['Model'] = ["ARIMA"] * metrics.shape[0]
metrics.to_csv(os.path.join('results', 'arima_metrics.csv'), index=False)