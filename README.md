# Web Application to Evaluate Statistical Time Series Forecast Models: Application to Walmart Sales

Statistical time series forecast is a technique that predicts future values using past values and/or errors of sequenced time series data. Whereas __Exponential Smoothing__ estimates the observed value by weighting recent observations heavier than distant observations, __Holt’s Linear Trend__ expands Exponential Smoothing by adding a growth rate to the estimated value. Moreover, __multiplicative Holt-Winters__ accounts for seasonality by multiplying seasonal factor and sum of estimated mean and growth rate. In contrast, __Autoregressive Integrated Moving Average (ARIMA)__ predicts actual values using a linear combination of the mean of time series, a constant value, weighted past values, weighted past errors and current error. __Seasonal Autoregressive Integrated Moving Average__ further extends ARIMA by incorporating seasonality terms into the model. To enhance understanding of aforementioned statistical time series forecast methods, this study builds a web application that forecasts weekly sales of 45 different Walmart stores in the US using sales data collected between 2010 and 2012. The application not only enables users to evaluate aforesaid statistical time series forecast models but also enables users to gain practical experience with each model.

## Preprocessing

- [Data Preparation](https://github.com/nphan20181/sales-forecast/blob/main/prepare_ts_data.ipynb)
- [SVD Components](https://github.com/nphan20181/sales-forecast/blob/main/create_svd_results.ipynb)

## Statistical Time Series Forecast Models

1. Simple Exponential Smoothing
   - [Model Implementation](https://github.com/nphan20181/sales-forecast/blob/main/module/es_model.py)
   - [Model Metrics](https://github.com/nphan20181/sales-forecast/blob/main/create_es_metrics.ipynb)
1. Holt's Linear Trend
   - [Model Implementation](https://github.com/nphan20181/sales-forecast/blob/main/module/holt_trend_es.py)
   - [Model Metrics](https://github.com/nphan20181/sales-forecast/blob/main/create_holt_metrics.ipynb)
1. Multiplicative Holt-Winters
   - [Model Implementation](https://github.com/nphan20181/sales-forecast/blob/main/module/holt_winters.py)
   - [Model Metrics](https://github.com/nphan20181/sales-forecast/blob/main/create_holt_winters_metrics.ipynb)
1. (Seasonal) Autoregressive Integrated Moving Average Plus SVD Component(s)
   - [Model Implementation](https://github.com/nphan20181/sales-forecast/blob/main/module/sarima_model.py)
   - [ARIMA Metrics](https://github.com/nphan20181/sales-forecast/blob/main/create_arima_metrics.py)
   - [SARIMA Metrics](https://github.com/nphan20181/sales-forecast/blob/main/create_sarima_metrics.py)

## References

1. Mendenhall, W. (2019). SECOND COURSE IN STATISTICS: regression analysis. Prentice Hall.
1. Bowerman, B. L., O’connell, R. T., & Koehler, A. B. (2005). Forecasting, time series, and regression : an applied approach. Thomson Brooks/Cole.
1. Walmart Recruiting - Store Sales Forecasting. (n.d.). Kaggle.com. https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data
1. Ostertagova, E., & Ostertag, O. (2011). The Simple Exponential Smoothing Model. Modelling of Mechanical and Mechatronic Systems. Herľany, Slovak Republic: ResearchGate.
1. Peixeiro, M. (2022). Time Series Forecasting in Python. Simon and Schuster.
1. Hyndman, R., Koehler, A. B., J Keith Ord, Snyder, R. D., & Springerlink (Online Service. (2008). Forecasting with Exponential Smoothing : The State Space Approach. Springer Berlin Heidelberg.
1. Verma, Y. (2021, August 18). Complete Guide To Dickey-Fuller Test In Time-Series Analysis. Analytics India Magazine. https://analyticsindiamag.com/complete-guide-to-dickey-fuller-test-in-time-series-analysis/
