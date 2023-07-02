import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing



# Assuming the M5 dataset is stored in a CSV file
data = pd.read_csv('sales_train_evaluation.csv')

product_id = 'HOBBIES_1_001'
product_data = data[data['item_id'].str.contains(product_id)]

product_data_subset = product_data.iloc[:, 4:]

product_data_subset = product_data_subset.reset_index(drop=True)

product_data_subset = product_data_subset.T

dataraw = product_data_subset.iloc[2:,0:]



# heirarchy = np.zeros((19410,4))

# for i in range(10):
#     heirarchy[i*1941:(i+1)*1941-1,1] = np.sum(dataraw[:,i], axis=1)

print(dataraw)

data = product_data_subset

# # Perform bottom-up forecasting
# forecasts = []
# for group in hierarchy_groups:
#     group_forecasts = []
#     for level in hierarchy_levels:
#         # Select the time series data for the current group and level
#         group_level_data = data[(data['group'] == group) & (data['level'] == level)]
        
#         # Extract the time series values
#         time_series = group_level_data['value']
        
#         # Perform ETS forecasting on the time series
#         model = ExponentialSmoothing(time_series)
#         model_fit = model.fit()
#         forecast = model_fit.forecast(1)
        
#         # Append the forecasted value to the group's forecast
#         group_forecasts.append(forecast[0])
    
#     # Sum the forecasts across levels to get the group forecast
#     group_forecast = sum(group_forecasts)
    
#     # Append the group forecast to the overall forecasts
#     forecasts.append(group_forecast)

# # Assign the forecasts to the appropriate hierarchy levels and groups
# # ...

# # Print or use the forecasts as desired
# print(forecasts)




forecasts = []
for i in range(1):
    # Extract the time series values
    time_series = dataraw.iloc[:,i].to_numpy()
        
    # Perform ETS forecasting on the time series
    model = ExponentialSmoothing(time_series)
    model_fit = model.fit()
    forecast = model_fit.forecast(28)

    forecasts.append(forecast)

# Print or use the forecasts as desired
print(forecasts)

















