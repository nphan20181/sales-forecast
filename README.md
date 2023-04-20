# Web Application to Evaluate Statistical Time Series Forecast Models: Application to Walmart Sales

## Documents

 - [Project Poster Page 1](https://github.com/nphan20181/sales-forecast/blob/main/Poster%20Page%201.pdf)
 - [Project Poster Page 2](https://github.com/nphan20181/sales-forecast/blob/main/Poster%20Page%202.pdf)

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
