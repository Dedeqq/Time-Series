import pandas as pd
import matplotlib.pyplot as plt

# Load the data and edit the dataset
dataset=pd.read_csv("station_sao_paulo.csv")
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
        

plt.rcParams["figure.figsize"] = [10.50, 3.50]
plt.rcParams["figure.autolayout"] = True
fig = plt.figure()
df.plot()
spacing = 0.500
fig.subplots_adjust(bottom=spacing)
plt.show()