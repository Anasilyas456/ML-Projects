###import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder,StandardScaler
df=pd.read_csv('C:\\Users\\QHS 006\\Desktop\\code alpha\\credit scoring\\bank.csv', delimiter=';')
print(df.head())
# print(df.columns)

print("-------------------------------------------------------------------------------")

####now i encode the data
encoder=LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = encoder.fit_transform(df[col])
print(df.head())
# print(df.isnull().sum())
plt.plot(df)
plt.show()
print("-------------------------------------------------------------------------------------")
#### now i standardize the encode data
stander=StandardScaler()
scaled_data=stander.fit_transform(df)
data=pd.DataFrame(scaled_data,columns=df.columns)
print(data.head())
plt.plot(data)
plt.show()
print("-------------------------------------------------------------------------------------")
########now remove the outliers from data
z_score=np.abs(stats.zscore(data))
threshold=3

outliers = np.where(z_score > threshold)[0]
new_data=(data.drop(index=outliers))
print(new_data)
plt.plot(new_data)
plt.show()
print("-------------------------------------------------------------------------------------")
####now divide the new_data into x and y
x=new_data.drop('y',axis=1)
y=new_data['y']
y = y.astype(int)
#####train the data
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(xtrain,ytrain)
#####predict the data
pred=model.predict(xtest)
accuracy=classification_report(ytest,pred)
print(accuracy)
print("-------------------------------------------------------------------------------------")
