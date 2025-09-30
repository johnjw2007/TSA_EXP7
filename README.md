# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 30.09.2025
### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error


file_path = 'XAUUSD_2010-2023.csv'
data = pd.read_csv(file_path)

data['time'] = pd.to_datetime(data['time'], format='%d-%m-%Y %H:%M')
data.set_index('time', inplace=True)

close_prices = data['close']

weekly_close_prices = close_prices.resample('W').mean()

result = adfuller(weekly_close_prices.dropna())
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
if result[1] < 0.05:
    print("The data is stationary.")
else:
    print("The data is non-stationary.")

train_size = int(len(weekly_close_prices) * 0.8)
train, test = weekly_close_prices[:train_size], weekly_close_prices[train_size:]

fig, ax = plt.subplots(2, figsize=(8, 6))
plot_acf(train.dropna(), ax=ax[0], title='Autocorrelation Function (ACF)')
plot_pacf(train.dropna(), ax=ax[1], title='Partial Autocorrelation Function (PACF)')
plt.show()

ar_model = AutoReg(train.dropna(), lags=13).fit()

ar_pred = ar_model.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

plt.figure(figsize=(10, 4))
plt.plot(test, label='Test Data')
plt.plot(ar_pred, label='AR Model Prediction', color='red')
plt.title('AR Model Prediction vs Test Data')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

mse = mean_squared_error(test, ar_pred)
print(f'Mean Squared Error (MSE): {mse}')

plt.figure(figsize=(10, 4))
plt.plot(train, label='Train Data')
plt.plot(test, label='Test Data')
plt.plot(ar_pred, label='AR Model Prediction', color='red')
plt.title('Train, Test, and AR Model Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

```
### OUTPUT:

GIVEN DATA
<img width="1071" height="463" alt="image" src="https://github.com/user-attachments/assets/d243a47e-1875-46c1-a20b-a66d56e8d177" />

PACF - ACF
<img width="837" height="323" alt="image" src="https://github.com/user-attachments/assets/c5b836d6-d884-4fb1-a2b3-850124ef011e" />
<img width="845" height="331" alt="image" src="https://github.com/user-attachments/assets/c3c1a7c0-4cce-40ef-9649-bce8c9ac134e" />


PREDICTION
<img width="1060" height="482" alt="image" src="https://github.com/user-attachments/assets/e4b0df41-65c8-47cc-b80d-a2a5bf842637" />

FINIAL PREDICTION
<img width="1064" height="482" alt="image" src="https://github.com/user-attachments/assets/34945008-f87f-4e46-b540-c9c3909c0640" />

### RESULT:
Thus we have successfully implemented the auto regression function using python.
