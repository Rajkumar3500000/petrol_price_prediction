import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 

pt=pd.read_csv("petrol.csv")
print(pt.head())
print(pt.columns)


pt["Date"]=pd.to_datetime(pt["Date"], format="%d.%m.%Y")

pt["year"]=pt["Date"].dt.year
pt["month"]=pt["Date"].dt.month
print(pt.head())

x=pt[["year","month"]]
y=pt["Avg Petrol Rate India (₹/L)"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


model=LinearRegression()
model.fit(x_train,y_train)

year=int(input("enter year:"))
if year<2000:
    print("invalid")
    exit()
month=int(input("enter month:"))
if month>12:
    print("invalid month")
    exit()

prediction=model.predict([[year,month]])
print("petrol rate is:",prediction[0],2)