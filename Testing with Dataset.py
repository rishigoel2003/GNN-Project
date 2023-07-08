import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt


# Assuming the M5 dataset is stored in a CSV file
data = pd.read_csv('sales_train_evaluation.csv')

product_id = 'HOBBIES_1_001'
product_data = data[data['item_id'].str.contains(product_id)]

product_data_subset = product_data.iloc[:, 4:]

product_data_subset = product_data_subset.reset_index(drop=True)

product_data_subset = product_data_subset.T

dataraw = product_data_subset.iloc[2:,0:]




#ETS bottom up

n=10
forecasts = np.zeros((n,28))
for i in range(n):
    # Extract the time series values
    time_series = dataraw.iloc[:,i].to_numpy()
        
    # Perform ETS forecasting on the time series
    model = ExponentialSmoothing(time_series)
    model_fit = model.fit()
    forecast = model_fit.forecast(28)

    forecasts[i,:] = forecast[:]

# Print or use the forecasts as desired

print(forecasts)


initial = list(range(1901, 1942))
after = range(1942, 1970)

for j in range(n):
    plt.plot(initial,dataraw.iloc[1900:,j].to_numpy())
    plt.plot(after,forecasts[j,:])

plt.show()


#ETS 








#light gbm bottom up














