import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
st.title("Stock Trend Prediction")

user_input = st.text_input("Enter Stock Ticker")
start=st.date_input("Enter starting date")
end=st.date_input("Enter end date ")
df=yf.Ticker(user_input)
df2=df.history(start=start,end=end)

#Describing Data
st.subheader("Data from" +  str(start)  +  "to"  +  str(end))
st.write(df2.describe())
#Visualization
st.subheader("Closing Price vs Time chart")
fig= plt.figure(figsize=(12,6))
plt.plot(df2.Close)
st.pyplot(fig)

st.subheader("Closing Price vs Time chart with 100MA")
ma100 = df2.Close.rolling(100).mean()
fig= plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df2.Close)
st.pyplot(fig)

st.subheader("Closing Price vs Time chart with 100MA & 200MA")
ma100 = df2.Close.rolling(100).mean()
ma200 = df2.Close.rolling(200).mean()
fig= plt.figure(figsize=(12,6))
plt.plot(ma100,"r")
plt.plot(ma200,"g")
plt.plot(df2.Close, "b")
st.pyplot(fig)

data_training = pd.DataFrame(df2['Close'][0:int(len(df2)*0.70)])
data_testing = pd.DataFrame(df2['Close'][int(len(df2)*0.70):int(len(df2))])
print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)


model=load_model('keras_model.h5')

#Testing part
past_100_days = data_training.tail(100)
final_df= past_100_days.append(data_testing,ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test=[]
y_test=[]
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
x_test, y_test =np.array(x_test), np.array(y_test)
y_predicted =model.predict(x_test)
scaler = scaler.scale_

scale_factor=1/ 0.01670676
y_predicted=y_predicted* scale_factor
y_test=y_test*scale_factor

#Final Graph
st.subheader("Predicted vs Original")
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test, "b", label = "Original Price")
plt.plot(y_predicted, "r", label = "Predicted Price")
plt.xlabel("Time")
plt.xlabel("Price")
plt.legend()
st.pyplot(fig2)