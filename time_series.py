import pandas as pd
import matplotlib.pyplot as plt

# Load the data and edit the dataset
dataset=pd.read_csv("https://raw.githubusercontent.com/Dedeqq/Time-Series/main/station_sao_paulo.csv")
dataset.drop(labels=['D-J-F','M-A-M','J-J-A','S-O-N','metANN'],axis=1,inplace=True)
dataset.drop(dataset.index[[i for i in range(17)]],inplace=True)
values=dataset.drop(['YEAR'],axis=1)
df=values.stack().reset_index()
df=df.drop(['level_1'],axis=1)
months=['01','02','03','04','05','06','07','08','09','10','11','12']
dates=[f'{months[j]}-{i}' for i in range(1963,2020) for j in range(12)]
df.columns=['date','temperature']
df['date']=dates
for i in range(684):
    if df.iloc[i,1]==999.9: 
        df.iloc[i,1]=df.iloc[i-12,1]
df['date']=pd.to_datetime(df['date'])
df.set_index('date', inplace=True)


# Plot temperature and year average
df['year_average'] = df.rolling(window=12).mean()
plt.rcParams["figure.figsize"] = [10.50, 3.50]
plt.rcParams["figure.autolayout"] = True
fig = plt.figure()
df.plot()
spacing = 0.500
fig.subplots_adjust(bottom=spacing)
plt.savefig("dataset.jpg")
plt.show()
df=df.drop(['year_average'],axis=1)



from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df, model='multiplicative')
fig = result.plot()


# Split the data
test = df.loc['2010-01-01':]
train = df.loc[:'2009-12-01']

plt.figure()
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.legend(fontsize=10)
plt.savefig("dataset_split.jpg")
plt.show()

# Check stationarity
from statsmodels.tsa.stattools import adfuller
test_result=adfuller(train['temperature'])
print(test_result)
# p-value less than 0.05 incides that data is stationary, however later on I will check first difference

# Auto-corelation and partial auto-corelation
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(train['temperature'],ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(train['temperature'],ax=ax2)
plt.savefig("acf_and_pacf.jpg")
plt.show()


# Differencing
test_result=adfuller(train['temperature'].diff(12).dropna())
print(test_result)
train['temperature'].diff(12).dropna().plot()

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(train['temperature'].diff(12).dropna(),ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(train['temperature'].diff(12).dropna(),ax=ax2)
plt.show()


# Auto ARIMA check
from pmdarima import auto_arima
stepwise_model = auto_arima(train, start_p=1, start_q=1,
                           max_p=3, max_d=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)


# Seasonal ARIMA model
model = sm.tsa.statespace.SARIMAX(train, freq='MS', order=(1, 1, 2), seasonal_order=(1, 0, 2, 12))
model_fit = model.fit(disp=False)

fcast_len = len(test)
fcast = model_fit.forecast(fcast_len)
plt.figure(figsize=(20, 10))
plt.plot(fcast, label='Forecast')
plt.plot(test, label='Test')
plt.legend(fontsize=25)
plt.savefig("forecast.jpg")
plt.show()

# Evaluation
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse = mean_squared_error(test, fcast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test, fcast)

print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')