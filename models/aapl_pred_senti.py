
#pred made for aapl 
#fin_stocks = tsla aapl ea
#FEATURES = ['high', 'low', 'open', 'close', 'volume', 'Month', 'Year','atr','ma5','ma30','ma180', 'rsi', 'obv', 'macd', 'eps','senti','prev']
# Mean Absolute Percentage Error (MAPE): 16.71 %
# Median Absolute Percentage Error (MDAPE): 15.44 %
#see image aapl_pred_senti.png




import math
import numpy as np 
import pandas as pd 
import time
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
# from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import keras
import tensorflow
from datetime import date, timedelta, datetime
# This function adds plotting functions for calender dates
from pandas.plotting import register_matplotlib_converters
# Important package for visualization - we use this to plot the market data
import matplotlib.pyplot as plt 
# Formatting dates
import matplotlib.dates as mdates
# Packages for measuring model performance / errors
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Deep learning library, used for neural networks
from keras.models import Sequential 
# Deep learning classes for recurrent and regular densely-connected layers
from keras.layers import LSTM, Dense, Dropout
# EarlyStopping during model training
from keras.callbacks import EarlyStopping
# This Scaler removes the median and scales the data according to the quantile range to normalize the price data 
from sklearn.preprocessing import RobustScaler
#new
import requests
#EXTRACTING INFO FOR TARGET STOCK
#key='DUQG7U14LK4C3BT2'
key = "5Y3MCR5FPN0GYZGO"
apiKey = key
t = 'AAPL'
ticker = t
    
df_temp = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='+ticker+'&apikey='+apiKey+'&outputsize=full&datatype=csv') 
ma5 = pd.read_csv('https://www.alphavantage.co/query?function=EMA&symbol='+ticker+'&interval=daily&time_period=5&series_type=close&apikey='+apiKey+'&datatype=csv')
ma30 = pd.read_csv('https://www.alphavantage.co/query?function=EMA&symbol='+ticker+'&interval=daily&time_period=30&series_type=close&apikey='+apiKey+'&datatype=csv')
ma180 = pd.read_csv('https://www.alphavantage.co/query?function=EMA&symbol='+ticker+'&interval=daily&time_period=180&series_type=close&apikey='+apiKey+'&datatype=csv')
atr = pd.read_csv('https://www.alphavantage.co/query?function=ATR&symbol='+ticker+'&interval=daily&time_period=14&apikey='+apiKey+'&datatype=csv&outputsize=full')


ma5 = ma5.truncate( after = 2564)
ma30 = ma30.truncate( after = 2564)
ma180= ma180.truncate( after = 2564)
df_temp = df_temp.truncate( after = 2564)
atr = atr.truncate(after = 2564)


df = df_temp.copy()
df['atr'] = atr['ATR']
df['ma5'] = ma5['EMA']
df['ma30'] = ma30['EMA']
df['ma180'] = ma180['EMA']

#new
time.sleep(60)
rsi = pd.read_csv('https://www.alphavantage.co/query?function=RSI&symbol='+ticker+'&interval=daily&time_period=14&series_type=open&apikey='+apiKey+'&datatype=csv')
rsi['time'][0] = rsi['time'][0].split()[0]
obv = pd.read_csv('https://www.alphavantage.co/query?function=OBV&symbol='+ticker+'&interval=daily&apikey='+apiKey+'&datatype=csv')
obv['time'][0] = obv['time'][0].split()[0]
macd = pd.read_csv('https://www.alphavantage.co/query?function=MACD&symbol='+ticker+'&interval=daily&series_type=open&apikey='+apiKey+'&datatype=csv')
url = 'https://www.alphavantage.co/query?function=EARNINGS&symbol='+ticker+'&apikey='+apiKey
r = requests.get(url)
data = r.json()
eps_list = data['quarterlyEarnings']
eps = pd.DataFrame(columns=['date', 'eps'])
for i in range(len(eps_list)):
    eps.loc[i] = [eps_list[i]['reportedDate'],eps_list[i]['reportedEPS']]
rsi = rsi.truncate(after = 2564)
obv = obv.truncate(after = 2564)
macd = macd.truncate(after = 2564)
eps = eps.truncate(after = 2564)
df['rsi'] = rsi['RSI']
df['obv'] =  obv['OBV']
df['macd'] = macd['MACD_Hist']
df.set_index('timestamp',drop = True, inplace = True)
eps.set_index('date',drop = True, inplace = True)
df['eps'] = eps['eps']
df.reset_index(inplace =True)
eps.reset_index(inplace  = True)
row = 0

for i in range(len(df)):
    if(pd.isnull(df['eps'][i])):
        df['eps'][i] = eps['eps'][row]
    else:
        row = row+1 
    
#now df has 14 columns with diff daily feature data for the past 2 years
temp=pd.DataFrame()
temp['time']=df['timestamp']
temp['senti']=np.nan
temp['prev']=np.nan
index_eps=0
# print(eps['eps'][1])
for i in range(len(df)-1):
    temp['senti'][i]=0.9996781
    temp['prev'][i]=df['close'][i+1]
# df['eps']=temp['eps']
df['senti']=temp['senti']
df['prev']=temp['prev']
df = df[:-1]

train_dfs = df.copy()
train_dfs  = train_dfs.iloc[::-1]
train_dfs = train_dfs.set_index('timestamp')


#transform date indexes
date_index = train_dfs.index
# Adding Month and Year in separate columns
d = pd.to_datetime(train_dfs.index)
train_dfs['Month'] = d.strftime("%m") 
train_dfs['Year'] = d.strftime("%Y") 
train_dfs = train_dfs.reset_index(drop=True).copy()

#new
# List of considered Features
FEATURES = ['high', 'low', 'open', 'close', 'volume', 'Month', 'Year','atr','ma5','ma30','ma180', 'rsi', 'obv', 'macd', 'eps','senti','prev']
# print('FEATURE LIST')
# print([f for f in FEATURES])

# Create the dataset with features and filter the data to the list of FEATURES
data = pd.DataFrame(train_dfs)
data_filtered = data[FEATURES]

# We add a prediction column and set dummy values to prepare the data for scaling
data_filtered_ext = data_filtered.copy()
data_filtered_ext['Prediction'] = data_filtered_ext['close'] 

#SCALING
# Calculate the number of rows in the data
nrows = data_filtered.shape[0]
np_data_unscaled = np.array(data_filtered)
np_data_unscaled = np.reshape(np_data_unscaled, (nrows, -1))
print(np_data_unscaled.shape)

# Transform the data by scaling each feature to a range between 0 and 1
scaler = RobustScaler()
np_data = scaler.fit_transform(np_data_unscaled)

# Creating a separate scaler that works on a single column for scaling predictions
scaler_pred = RobustScaler()
df_Close = pd.DataFrame(data_filtered_ext['close'])
np_Close_scaled = scaler_pred.fit_transform(df_Close)

#SPLITTING
#Settings
sequence_length = 100

# Split the training data into x_train and y_train data sets
# Get the number of rows to train the model on 80% of the data 
train_data_len = math.ceil(np_data.shape[0] * 0.8) #2616

# Create the training data
train_data = np_data[0:train_data_len, :]
x_train, y_train = [], []

# The RNN needs data with the format of [samples, time steps, features].
# Here, we create N samples, 100 time steps per sample, and 2 features
for i in range(100, train_data_len):
    x_train.append(train_data[i-sequence_length:i,:]) #contains 100 values 0-100 * columsn
    y_train.append(train_data[i, 0]) #contains the prediction values for validation
    
# Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Create the test data
test_data = np_data[train_data_len - sequence_length:, :]

# Split the test data into x_test and y_test
x_test, y_test = [], []
test_data_len = test_data.shape[0]
for i in range(sequence_length, test_data_len):
    x_test.append(test_data[i-sequence_length:i,:]) #contains 100 values 0-100 * columns
    y_test.append(test_data[i, 0]) #contains the prediction values for validation
# Convert the x_train and y_train to numpy arrays
x_test, y_test = np.array(x_test), np.array(y_test)

# Convert the x_train and y_train to numpy arrays
x_test = np.array(x_test); y_test = np.array(y_test)
    
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#FITTING IN MODEL
# Configure the neural network model
model = Sequential()
# Model with 100 Neurons 
# inputshape = 100 Timestamps, each with x_train.shape[2] variables
n_neurons = x_train.shape[1] * x_train.shape[2]
print(n_neurons, x_train.shape[1], x_train.shape[2])
model.add(LSTM(n_neurons, return_sequences=False, 
               input_shape=(x_train.shape[1], x_train.shape[2]))) 
model.add(Dense(1, activation='relu'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
epochs = 8
early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
history = model.fit(x_train, y_train, batch_size=16, 
                    epochs=epochs, callbacks=[early_stop])

# Get the predicted values
predictions_init = model.predict(x_test)
#extract last layer of RNN
extractor = keras.Model(inputs=model.inputs,
                        outputs=model.layers[-1].output)
features = extractor(x_test)
features=features.numpy()
features_ticker = features
all_features = []
all_features.append(features.tolist())
sum =0
for i in features:
    sum = sum + i
sum_layers = []
all_stocks = []
all_stocks.append(ticker)
sum_layers.append(sum)

tickers = ['AAPL','BB','AMD','INTC','MSFT','AMZN','IBM','TXN','GOOG','WBA','NOC','BA','LMT','MCD','NAV','MA','GE','AXP','PEP','KO','JNJ','TM','HMC','XOM','CVX','VLO','F','BAC','NOK','EA','HPE','ABT','NVDA','CSCO','BBBY','AAL','T','OXY','FUBO','SWN','UAL','BSX','GM','EBAY','V','DIS','WMT']
#tickers = ['XOM','CVX','VLO','F','BAC','NOK','EA','HPE','ABT','NVDA','CSCO','BBBY','AAL','T','OXY','FUBO','SWN','UAL','BSX','GM','EBAY','V','DIS','WMT']


#incomplete run
#tickers = ['MSFT','AMZN','IBM','TXN','GOOG','WBA','NOC','BA','LMT','MCD','NAV','MA','GE','AXP','PEP','KO','JNJ','TM','HMC','XOM','CVX','VLO','F','BAC','NOK','EA','HPE','ABT','NVDA','CSCO','BBBY','AAL','T','OXY','FUBO','SWN','UAL','BSX','GM','EBAY','V','DIS','WMT']

for ticker in tickers:
    print("start:",ticker)
    time.sleep(50)
    df_temp = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='+ticker+'&apikey='+apiKey+'&outputsize=full&datatype=csv') 
    ma5 = pd.read_csv('https://www.alphavantage.co/query?function=EMA&symbol='+ticker+'&interval=daily&time_period=5&series_type=close&apikey='+apiKey+'&datatype=csv')
    ma30 = pd.read_csv('https://www.alphavantage.co/query?function=EMA&symbol='+ticker+'&interval=daily&time_period=30&series_type=close&apikey='+apiKey+'&datatype=csv')
    ma180 = pd.read_csv('https://www.alphavantage.co/query?function=EMA&symbol='+ticker+'&interval=daily&time_period=180&series_type=close&apikey='+apiKey+'&datatype=csv')
    atr = pd.read_csv('https://www.alphavantage.co/query?function=ATR&symbol='+ticker+'&interval=daily&time_period=14&apikey='+apiKey+'&datatype=csv&outputsize=full')
    #time.sleep(50)
    if len(ma180)>=2565 : 
        ma5 = ma5.truncate( after = 2564)
        ma30 = ma30.truncate( after = 2564)
        ma180= ma180.truncate( after = 2564)
        df_temp = df_temp.truncate( after = 2564)
        atr = atr.truncate(after = 2564)
    else:
        ma5 = ma5.truncate( after = len(ma180)+1)
        ma30 = ma30.truncate( after = len(ma180)+1)
        ma180= ma180.truncate( after = len(ma180)+1)
        df_temp = df_temp.truncate( after = len(ma180)+1)
        atr = atr.truncate(after = len(ma180)+1)   
    df = df_temp.copy()
    df['atr'] = atr['ATR']
    df['ma5'] = ma5['EMA']
    df['ma30'] = ma30['EMA']
    df['ma180'] = ma180['EMA']
    #now df has 10 columns with diff daily feature data for the past 2 years
    #new
    time.sleep(50)
    #apiKey = "5Y3MCR5FPN0GYZGO"
    rsi = pd.read_csv('https://www.alphavantage.co/query?function=RSI&symbol='+ticker+'&interval=daily&time_period=14&series_type=open&apikey='+apiKey+'&datatype=csv')
    rsi['time'][0] = rsi['time'][0].split()[0]
    obv = pd.read_csv('https://www.alphavantage.co/query?function=OBV&symbol='+ticker+'&interval=daily&apikey='+apiKey+'&datatype=csv')
    obv['time'][0] = obv['time'][0].split()[0]
    macd = pd.read_csv('https://www.alphavantage.co/query?function=MACD&symbol='+ticker+'&interval=daily&series_type=open&apikey='+apiKey+'&datatype=csv')
    url = 'https://www.alphavantage.co/query?function=EARNINGS&symbol='+ticker+'&apikey='+apiKey
    r = requests.get(url)
    data = r.json()
    eps_list = data['quarterlyEarnings']
    eps = pd.DataFrame(columns=['date', 'eps'])
    for i in range(len(eps_list)):
        eps.loc[i] = [eps_list[i]['reportedDate'],eps_list[i]['reportedEPS']]
    if len(ma180)>=2565 : 
        rsi = rsi.truncate(after = 2564)
        obv = obv.truncate(after = 2564)
        macd = macd.truncate(after = 2564)
        eps = eps.truncate(after = 2564)
    else:
        rsi = rsi.truncate(after = len(ma180)+1)
        obv = obv.truncate(after = len(ma180)+1)
        macd = macd.truncate(after = len(ma180)+1)
        eps = eps.truncate(after = len(ma180)+1)     
    
    df['rsi'] = rsi['RSI']
    df['obv'] =  obv['OBV']
    df['macd'] = macd['MACD_Hist']
    df.set_index('timestamp',drop = True, inplace = True)
    eps.set_index('date',drop = True, inplace = True)
    df['eps'] = eps['eps']
    df.reset_index(inplace =True)
    eps.reset_index(inplace  = True)
    row = 0
    
    for i in range(len(df)):
        if(pd.isnull(df['eps'][i])):
            df['eps'][i] = eps['eps'][row]
        else:
            row = row+1 
    temp=pd.DataFrame()
    temp['time']=df['timestamp']
    temp['senti']=np.nan
    temp['prev']=np.nan
    index_eps=0
    # print(eps['eps'][1])
    for i in range(len(df)-1):
        if ticker=='TSLA':
            temp['senti'][i]= 0.9997481
        if ticker=='AAPL':
            temp['senti'][i]= 0.9996781
        if ticker=='EA':
            temp['senti'][i]= 0.9999581   
        temp['prev'][i]=df['close'][i+1]
    # df['eps']=temp['eps']
    df['senti']=temp['senti']
    df['prev']=temp['prev']
    df = df[:-1]
    
    #now df has 14 columns with diff daily feature data for the past 2 years
    
        
            
    train_dfs_temp = df.copy()
    train_dfs_temp  = train_dfs_temp.iloc[::-1]
    train_dfs_temp = train_dfs_temp.set_index('timestamp')

    date_index = train_dfs_temp.index
    # Adding Month and Year in separate columns
    d = pd.to_datetime(train_dfs_temp.index)
    train_dfs_temp['Month'] = d.strftime("%m") 
    train_dfs_temp['Year'] = d.strftime("%Y") 
    # We reset the index, so we can convert the date-index to a number-index
    train_dfs_temp = train_dfs_temp.reset_index(drop=True).copy()
    
    #feature engineering
    # List of considered Features
    #FEATURES = ['high', 'low', 'open', 'close', 'volume', 'Month', 'Year','atr','ma5','ma30','ma180']
    FEATURES = ['high', 'low', 'open', 'close', 'volume', 'Month', 'Year','atr','ma5','ma30','ma180', 'rsi', 'obv', 'macd', 'eps','senti','prev']

    # print('FEATURE LIST')
    # print([f for f in FEATURES])
    
    # Create the dataset with features and filter the data to the list of FEATURES
    data = pd.DataFrame(train_dfs_temp)
    data_filtered = data[FEATURES]
    
    # We add a prediction column and set dummy values to prepare the data for scaling
    data_filtered_ext = data_filtered.copy()
    data_filtered_ext['Prediction'] = data_filtered_ext['close'] 
    
    #SCALING
    # Calculate the number of rows in the data
    nrows = data_filtered.shape[0]
    np_data_unscaled = np.array(data_filtered)
    np_data_unscaled = np.reshape(np_data_unscaled, (nrows, -1))
    print(np_data_unscaled.shape)
    
    # Transform the data by scaling each feature to a range between 0 and 1
    scaler = RobustScaler()
    np_data = scaler.fit_transform(np_data_unscaled)
    
    # Creating a separate scaler that works on a single column for scaling predictions
    scaler_pred = RobustScaler()
    df_Close = pd.DataFrame(data_filtered_ext['close'])
    np_Close_scaled = scaler_pred.fit_transform(df_Close)
    
    #SPLITTING
    #Settings
    sequence_length = 100
    
    # Split the training data into x_train and y_train data sets
    # Get the number of rows to train the model on 80% of the data 
    train_data_len = math.ceil(np_data.shape[0] * 0.8) #2616
    
    # Create the training data
    train_data = np_data[0:train_data_len, :]
    x_train, y_train = [], []
    
    # The RNN needs data with the format of [samples, time steps, features].
    # Here, we create N samples, 100 time steps per sample, and 2 features
    for i in range(100, train_data_len):
        x_train.append(train_data[i-sequence_length:i,:]) #contains 100 values 0-100 * columsn
        y_train.append(train_data[i, 0]) #contains the prediction values for validation
        
    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    # Create the test data
    test_data = np_data[train_data_len - sequence_length:, :]
    
    # Split the test data into x_test and y_test
    x_test, y_test = [], []
    test_data_len = test_data.shape[0]
    for i in range(sequence_length, test_data_len):
        x_test.append(test_data[i-sequence_length:i,:]) #contains 100 values 0-100 * columns
        y_test.append(test_data[i, 0]) #contains the prediction values for validation
    # Convert the x_train and y_train to numpy arrays
    x_test, y_test = np.array(x_test), np.array(y_test)
    
    # Convert the x_train and y_train to numpy arrays
    x_test = np.array(x_test); y_test = np.array(y_test)
        
    # print(x_train.shape, y_train.shape)
    # print(x_test.shape, y_test.shape)
    
    extractor = keras.Model(inputs=model.inputs,
                        outputs=model.layers[-1].output)
    features = extractor(x_test)
    features=features.numpy()
    #print(features)
    all_features.append(features.tolist())    
    all_stocks.append(ticker)
    sum_layers.append(sum)
    print("end:",ticker)
            
#redundant?   
new_features = [[]] 
for list in all_features:
    for x in list:
        for y in x:
            new_features.append(y)   

#creates a new datafram n_dim containing last layeer for each stock
minlen = len(all_features[0])
for list in all_features:
    if len(list) < minlen:
        minlen = len(list)
        
col_temp = np.arange(1, minlen+1)
col_temp = col_temp.tolist()
col = ['Name']  
col = col + col_temp
n_dim = pd.DataFrame(columns = col)
for i in range(len(all_stocks)):
    row = [all_stocks[i]]
    for j in range(minlen):
        row.append(all_features[i][j][0])
    n_dim.loc[len(n_dim)] = row    
n_dim = n_dim.set_index('Name')

#clustering with various k values and plotting the elbow curve
from sklearn.cluster import KMeans
from math import sqrt
import  pylab as pl
import numpy as np        
sse = []
for k in range(2,len(n_dim)):
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(n_dim)
    sse.append(kmeans.inertia_) #SSE for each n_clusters
pl.plot(range(2,len(n_dim)), sse)
pl.title("Elbow Curve")
pl.show()
 





#clustering with most optimanl value of k
k =10
kmeans = KMeans(n_clusters = k).fit(n_dim)
centroids = kmeans.cluster_centers_
clusters = kmeans.predict(n_dim)
labelled_stocks_ndim = pd.DataFrame({'stock name':all_stocks, 'Label':kmeans.labels_})
labelled_ndim_sorted = labelled_stocks_ndim.sort_values('Label')

ticker = t
labelled_stocks_ndim.set_index("stock name", inplace = True)
row  = labelled_stocks_ndim.loc[ticker] 
label_num  = row["Label"]

temp = labelled_ndim_sorted.where(labelled_ndim_sorted['Label']==label_num)
temp.dropna(inplace= True)
fin_stocks = temp['stock name'].tolist()
fin_stocks.remove(ticker)
#CLUSTERING OVER

from sklearn.decomposition import PCA #Principal Component Analysis
from sklearn.cluster import KMeans #K-Means Clustering
from sklearn.preprocessing import StandardScaler #used for 'Feature Scaling'

#plotly imports
import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot



#PCA
n_dim["Cluster"] = clusters
plotX = n_dim.reset_index()
n_dim.reset_index(inplace = True)
# plotX = plotX.drop('Name',axis = 1)
pca_2d = PCA(n_components=2)
PCs_2d = pd.DataFrame(pca_2d.fit_transform(plotX.drop(["Cluster","Name"], axis=1)))
PCs_2d.columns = ["PC1_2d", "PC2_2d"]
plotX = pd.concat([plotX,PCs_2d], axis=1, join='inner')
plotX["dummy"] = 0
init_notebook_mode(connected=True)
data = []

# from matplotlib.pyplot import cm

# color=iter(cm.rainbow(np.linspace(0,1,k)))
   
for i in range(k):
    #c=next(color)
    # c = 'cyan'
    #c= color[i]
    cluster = plotX[plotX["Cluster"] == i]
    trace = go.Scatter(
                    x = cluster["PC1_2d"],
                    y = cluster["PC2_2d"],
                    mode = "markers",
                    name = "Cluster"+str(i),
                    #marker = dict(color = c),
                    #marker = dict(color = 'rgba(255, 128, 255,'+0.1*i+ ')'),
                    text = cluster["Name"])
    data.append(trace)

title = "Visualizing Clusters in Two Dimensions Using PCA"

layout = dict(title = title,
              xaxis= dict(title= 'PC1',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'PC2',ticklen= 5,zeroline= False)
             )

fig = dict(data = data, layout = layout)

plot(fig)


#TRAIN ON TARGET STOCK
ticker = t
key= '5Y3MCR5FPN0GYZGO'

df_temp = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='+ticker+'&apikey='+apiKey+'&outputsize=full&datatype=csv') 
ma5 = pd.read_csv('https://www.alphavantage.co/query?function=EMA&symbol='+ticker+'&interval=daily&time_period=5&series_type=close&apikey='+apiKey+'&datatype=csv')
ma30 = pd.read_csv('https://www.alphavantage.co/query?function=EMA&symbol='+ticker+'&interval=daily&time_period=30&series_type=close&apikey='+apiKey+'&datatype=csv')
ma180 = pd.read_csv('https://www.alphavantage.co/query?function=EMA&symbol='+ticker+'&interval=daily&time_period=180&series_type=close&apikey='+apiKey+'&datatype=csv')
atr = pd.read_csv('https://www.alphavantage.co/query?function=ATR&symbol='+ticker+'&interval=daily&time_period=14&apikey='+apiKey+'&datatype=csv&outputsize=full')

ma5 = ma5.truncate( after = 2564)
ma30 = ma30.truncate( after = 2564)
ma180= ma180.truncate( after = 2564)
df_temp = df_temp.truncate( after = 2564)
atr = atr.truncate(after = 2564)
df = df_temp.copy()
df['atr'] = atr['ATR']
df['ma5'] = ma5['EMA']
df['ma30'] = ma30['EMA']
df['ma180'] = ma180['EMA']
#now df has 10 columns with diff daily feature data for the past 2 years
#new
time.sleep(50)
#apiKey = "5Y3MCR5FPN0GYZGO"
rsi = pd.read_csv('https://www.alphavantage.co/query?function=RSI&symbol='+ticker+'&interval=daily&time_period=14&series_type=open&apikey='+apiKey+'&datatype=csv')
rsi['time'][0] = rsi['time'][0].split()[0]
obv = pd.read_csv('https://www.alphavantage.co/query?function=OBV&symbol='+ticker+'&interval=daily&apikey='+apiKey+'&datatype=csv')
obv['time'][0] = obv['time'][0].split()[0]
macd = pd.read_csv('https://www.alphavantage.co/query?function=MACD&symbol='+ticker+'&interval=daily&series_type=open&apikey='+apiKey+'&datatype=csv')
url = 'https://www.alphavantage.co/query?function=EARNINGS&symbol='+ticker+'&apikey='+apiKey
r = requests.get(url)
data = r.json()
eps_list = data['quarterlyEarnings']
eps = pd.DataFrame(columns=['date', 'eps'])
for i in range(len(eps_list)):
    eps.loc[i] = [eps_list[i]['reportedDate'],eps_list[i]['reportedEPS']]
if len(ma180)>=2565 : 
    rsi = rsi.truncate(after = 2564)
    obv = obv.truncate(after = 2564)
    macd = macd.truncate(after = 2564)
    eps = eps.truncate(after = 2564)
else:
    rsi = rsi.truncate(after = len(ma180)+1)
    obv = obv.truncate(after = len(ma180)+1)
    macd = macd.truncate(after = len(ma180)+1)
    eps = eps.truncate(after = len(ma180)+1)     

df['rsi'] = rsi['RSI']
df['obv'] =  obv['OBV']
df['macd'] = macd['MACD_Hist']
df.set_index('timestamp',drop = True, inplace = True)
eps.set_index('date',drop = True, inplace = True)
df['eps'] = eps['eps']
df.reset_index(inplace =True)
eps.reset_index(inplace  = True)
row = 0

for i in range(len(df)):
    if(pd.isnull(df['eps'][i])):
        df['eps'][i] = eps['eps'][row]
    else:
        row = row+1 
    

train_dfs = df.copy()
train_dfs  = train_dfs.iloc[::-1]
train_dfs = train_dfs.set_index('timestamp')


#transform date indexes
date_index_fin = train_dfs.index
# Adding Month and Year in separate columns
d = pd.to_datetime(train_dfs.index)
train_dfs['Month'] = d.strftime("%m") 
train_dfs['Year'] = d.strftime("%Y") 
train_dfs = train_dfs.reset_index(drop=True).copy()


# List of considered Features
#FEATURES = ['high', 'low', 'open', 'close', 'volume', 'Month', 'Year','atr','ma5','ma30','ma180']

FEATURES = ['high', 'low', 'open', 'close', 'volume', 'Month', 'Year','atr','ma5','ma30','ma180', 'rsi', 'obv', 'macd', 'eps']
# print('FEATURE LIST')
# print([f for f in FEATURES])

# Create the dataset with features and filter the data to the list of FEATURES
data = pd.DataFrame(train_dfs)
data_filtered = data[FEATURES]
data_filtered_fin = data_filtered.copy()
# data_fin = data.copy()
# We add a prediction column and set dummy values to prepare the data for scaling
data_filtered_ext = data_filtered.copy()
data_filtered_ext['Prediction'] = data_filtered_ext['close'] 

#SCALING
# Calculate the number of rows in the data
nrows = data_filtered.shape[0]
np_data_unscaled = np.array(data_filtered)
np_data_unscaled = np.reshape(np_data_unscaled, (nrows, -1))
print(np_data_unscaled.shape)

# Transform the data by scaling each feature to a range between 0 and 1
scaler = RobustScaler()
np_data = scaler.fit_transform(np_data_unscaled)

# Creating a separate scaler that works on a single column for scaling predictions
scaler_pred = RobustScaler()
df_Close = pd.DataFrame(data_filtered_ext['close'])
np_Close_scaled = scaler_pred.fit_transform(df_Close)

#SPLITTING
#Settings
sequence_length = 100

# Split the training data into x_train and y_train data sets
# Get the number of rows to train the model on 80% of the data 
train_data_len_fin = math.ceil(np_data.shape[0] * 0.8) #2616

# Create the training data
train_data = np_data[0:train_data_len_fin, :]
x_train, y_train = [], []

# The RNN needs data with the format of [samples, time steps, features].
# Here, we create N samples, 100 time steps per sample, and 2 features
for i in range(100, train_data_len_fin):
    x_train.append(train_data[i-sequence_length:i,:]) #contains 100 values 0-100 * columsn
    y_train.append(train_data[i, 0]) #contains the prediction values for validation
    
# Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Create the test data
test_data = np_data[train_data_len_fin - sequence_length:, :]

# Split the test data into x_test and y_test
x_test, y_test = [], []
test_data_len = test_data.shape[0]
for i in range(sequence_length, test_data_len):
    x_test.append(test_data[i-sequence_length:i,:]) #contains 100 values 0-100 * columns
    y_test.append(test_data[i, 0]) #contains the prediction values for validation
# Convert the x_train and y_train to numpy arrays
x_test, y_test = np.array(x_test), np.array(y_test)

# Convert the x_train and y_train to numpy arrays
x_test_fin = np.array(x_test); y_test = np.array(y_test)
print(x_train.shape, y_train.shape)
print(x_test_fin.shape, y_test.shape)



model = Sequential()
# Model with 100 Neurons 
# inputshape = 100 Timestamps, each with x_train.shape[2] variables
n_neurons = x_train.shape[1] * x_train.shape[2]
print(n_neurons, x_train.shape[1], x_train.shape[2])
model.add(LSTM(n_neurons, return_sequences=False, 
               input_shape=(x_train.shape[1], x_train.shape[2]))) 
model.add(Dense(1, activation='relu'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
"""
# Training the model
epochs = 12 
early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
history = model.fit(x_train, y_train, batch_size=16, 
                epochs=epochs, callbacks=[early_stop])
 """
fin_stocks.append(ticker)
time.sleep(50)
#TRAINING ON OTHER STOCKS IN THE CLUSTER
# fin_stocks=['EA', 'SYMC', 'MA', 'TSLA', 'V', 'AAPL']
fin_stocks = ['AAPL']

for ticker in fin_stocks:
    df_temp = pd.read_csv('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='+ticker+'&apikey='+apiKey+'&outputsize=full&datatype=csv') 
    ma5 = pd.read_csv('https://www.alphavantage.co/query?function=EMA&symbol='+ticker+'&interval=daily&time_period=5&series_type=close&apikey='+apiKey+'&datatype=csv')
    ma30 = pd.read_csv('https://www.alphavantage.co/query?function=EMA&symbol='+ticker+'&interval=daily&time_period=30&series_type=close&apikey='+apiKey+'&datatype=csv')
    ma180 = pd.read_csv('https://www.alphavantage.co/query?function=EMA&symbol='+ticker+'&interval=daily&time_period=180&series_type=close&apikey='+apiKey+'&datatype=csv')
    atr = pd.read_csv('https://www.alphavantage.co/query?function=ATR&symbol='+ticker+'&interval=daily&time_period=14&apikey='+apiKey+'&datatype=csv&outputsize=full')
    
    ma5 = ma5.truncate( after = 2564)
    ma30 = ma30.truncate( after = 2564)
    ma180= ma180.truncate( after = 2564)
    df_temp = df_temp.truncate( after = 2564)
    atr = atr.truncate(after = 2564)
    df = df_temp.copy()
    df['atr'] = atr['ATR']
    df['ma5'] = ma5['EMA']
    df['ma30'] = ma30['EMA']
    df['ma180'] = ma180['EMA']
    #now df has 10 columns with diff daily feature data for the past 2 years
    
    #new
    time.sleep(60)
    #apiKey = "5Y3MCR5FPN0GYZGO"
    rsi = pd.read_csv('https://www.alphavantage.co/query?function=RSI&symbol='+ticker+'&interval=daily&time_period=14&series_type=open&apikey='+apiKey+'&datatype=csv')
    rsi['time'][0] = rsi['time'][0].split()[0]
    obv = pd.read_csv('https://www.alphavantage.co/query?function=OBV&symbol='+ticker+'&interval=daily&apikey='+apiKey+'&datatype=csv')
    obv['time'][0] = obv['time'][0].split()[0]
    macd = pd.read_csv('https://www.alphavantage.co/query?function=MACD&symbol='+ticker+'&interval=daily&series_type=open&apikey='+apiKey+'&datatype=csv')
    url = 'https://www.alphavantage.co/query?function=EARNINGS&symbol='+ticker+'&apikey='+apiKey
    r = requests.get(url)
    data = r.json()
    eps_list = data['quarterlyEarnings']
    eps = pd.DataFrame(columns=['date', 'eps'])
    for i in range(len(eps_list)):
        eps.loc[i] = [eps_list[i]['reportedDate'],eps_list[i]['reportedEPS']]
    if len(ma180)>=2565 : 
        rsi = rsi.truncate(after = 2564)
        obv = obv.truncate(after = 2564)
        macd = macd.truncate(after = 2564)
        eps = eps.truncate(after = 2564)
    else:
        rsi = rsi.truncate(after = len(ma180)+1)
        obv = obv.truncate(after = len(ma180)+1)
        macd = macd.truncate(after = len(ma180)+1)
        eps = eps.truncate(after = len(ma180)+1)     
    
    df['rsi'] = rsi['RSI']
    df['obv'] =  obv['OBV']
    df['macd'] = macd['MACD_Hist']
    df.set_index('timestamp',drop = True, inplace = True)
    eps.set_index('date',drop = True, inplace = True)
    df['eps'] = eps['eps']
    df.reset_index(inplace =True)
    eps.reset_index(inplace  = True)
    row = 0
    
    for i in range(len(df)):
        if(pd.isnull(df['eps'][i])):
            df['eps'][i] = eps['eps'][row]
        else:
            row = row+1 
     
    temp=pd.DataFrame()
    temp['time']=df['timestamp']
    temp['senti']=np.nan
    temp['prev']=np.nan
    index_eps=0
    # print(eps['eps'][1])
    for i in range(len(df)-1):
        if ticker=='TSLA':
            temp['senti'][i]= 0.9997481
        if ticker=='AAPL':
            temp['senti'][i]= 0.9996781
        if ticker=='EA':
            temp['senti'][i]= 0.9999581   
        temp['prev'][i]=df['close'][i+1]
    # df['eps']=temp['eps']
    df['senti']=temp['senti']
    df['prev']=temp['prev']
    df = df[:-1]
    
    
    train_dfs = df.copy()
    train_dfs  = train_dfs.iloc[::-1]
    train_dfs = train_dfs.set_index('timestamp')
    
    
    #transform date indexes
    date_index = train_dfs.index
    # Adding Month and Year in separate columns
    d = pd.to_datetime(train_dfs.index)
    train_dfs['Month'] = d.strftime("%m") 
    train_dfs['Year'] = d.strftime("%Y") 
    train_dfs = train_dfs.reset_index(drop=True).copy()
    
    
    # List of considered Features
    #FEATURES = ['high', 'low', 'open', 'close', 'volume', 'Month', 'Year','atr','ma5','ma30','ma180','']
    FEATURES = ['high', 'low', 'open', 'close', 'volume', 'Month', 'Year','atr','ma5','ma30','ma180', 'rsi', 'obv', 'macd', 'eps','senti','prev']

    # print('FEATURE LIST')
    # print([f for f in FEATURES])
    
    # Create the dataset with features and filter the data to the list of FEATURES
    data = pd.DataFrame(train_dfs)
    data_filtered = data[FEATURES]
    
    # We add a prediction column and set dummy values to prepare the data for scaling
    data_filtered_ext = data_filtered.copy()
    data_filtered_ext['Prediction'] = data_filtered_ext['close'] 
    
    #SCALING
    # Calculate the number of rows in the data
    nrows = data_filtered.shape[0]
    np_data_unscaled = np.array(data_filtered)
    np_data_unscaled = np.reshape(np_data_unscaled, (nrows, -1))
    print(np_data_unscaled.shape)
    
    # Transform the data by scaling each feature to a range between 0 and 1
    scaler = RobustScaler()
    np_data = scaler.fit_transform(np_data_unscaled)
    
    # Creating a separate scaler that works on a single column for scaling predictions
    scaler_pred = RobustScaler()
    df_Close = pd.DataFrame(data_filtered_ext['close'])
    np_Close_scaled = scaler_pred.fit_transform(df_Close)
    
    #SPLITTING
    #Settings
    sequence_length = 100
    
    # Split the training data into x_train and y_train data sets
    # Get the number of rows to train the model on 80% of the data 
    train_data_len = math.ceil(np_data.shape[0] * 0.8) #2616
    
    # Create the training data
    train_data = np_data[0:train_data_len, :]
    x_train, y_train = [], []
    
    # The RNN needs data with the format of [samples, time steps, features].
    # Here, we create N samples, 100 time steps per sample, and 2 features
    for i in range(100, train_data_len):
        x_train.append(train_data[i-sequence_length:i,:]) #contains 100 values 0-100 * columsn
        y_train.append(train_data[i, 0]) #contains the prediction values for validation
        
    # Convert the x_train and y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    # Create the test data
    test_data = np_data[train_data_len - sequence_length:, :]
    
    # Split the test data into x_test and y_test
    x_test, y_test = [], []
    test_data_len = test_data.shape[0]
    for i in range(sequence_length, test_data_len):
        x_test.append(test_data[i-sequence_length:i,:]) #contains 100 values 0-100 * columns
        y_test.append(test_data[i, 0]) #contains the prediction values for validation
    # Convert the x_train and y_train to numpy arrays
    x_test, y_test = np.array(x_test), np.array(y_test)
    
    # Convert the x_train and y_train to numpy arrays
    x_test = np.array(x_test); y_test = np.array(y_test)
        
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    
    
    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    history = model.fit(x_train, y_train, batch_size=16, 
                    epochs=epochs, callbacks=[early_stop])

# Get the predicted values-+*/
predictions = model.predict(x_test_fin)
 
# Get the predicted values
pred_unscaled = scaler_pred.inverse_transform(predictions)

# The date from which on the date is displayed
# display_start_date = pd.Timestamp('today') - timedelta(days=500)

# Add the date column
data_filtered_sub = data_filtered_fin.copy()
data_filtered_sub['Date'] = date_index_fin

# Add the difference between the valid and predicted prices
train = data_filtered_sub[:train_data_len_fin + 1]
valid = data_filtered_sub[train_data_len_fin:]
valid.insert(1, "Prediction", pred_unscaled.ravel(), True)
valid.insert(1, "Difference", valid["Prediction"] - valid["close"], True)
#------------------------------------------------------------
"""
# Zoom in to a closer timeframe
valid = valid[valid['Date'] > display_start_date]
train = train[train['Date'] > display_start_date]"""

# Visualize the data
fig, ax1 = plt.subplots(figsize=(22, 10), sharex=True)
xt = train['Date']; yt = train[["close"]]
xv = valid['Date']; yv = valid[["close", "Prediction"]]
plt.title("Predictions vs Actual Values", fontsize=20)
plt.ylabel(ticker, fontsize=18)
plt.plot(xt, yt, color="#039dfc", linewidth=2.0)
plt.plot(xv, yv["Prediction"], color="#E91D9E", linewidth=2.0)
plt.plot(xv, yv["close"], color="black", linewidth=2.0)
plt.xticks( rotation ='vertical')
size = len(yt) + len(yv)
plt.xticks(np.arange(0,size,50))
plt.legend(["Train", "Test Predictions", "Actual Values"], loc="upper left")


MAPE = np.mean((np.abs(valid['Difference']/ valid['close']))) * 100
print('Mean Absolute Percentage Error (MAPE): ' + str(np.round(MAPE, 2)) + ' %')

# Median Absolute Percentage Error (MDAPE)
MDAPE = np.median((np.abs(valid['Difference']/ valid['close'])) ) * 100
print('Median Absolute Percentage Error (MDAPE): ' + str(np.round(MDAPE, 2)) + ' %')
