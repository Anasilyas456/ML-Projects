 #########import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



###load the data
df=pd.read_csv('C:\\Users\\QHS 006\\Downloads\\archive (2)\\global_cancer_predictions.csv')
# print(df.isnull().sum().sort_values(ascending=True))
# print(df.head())
# print(df.dtypes)


##now encode the data
le=LabelEncoder()
for col in df.columns:
    if df[col].dtype=='object' or df[col].dtype=='category':
        df[col]=le.fit_transform(df[col])


##split the data in x and y
x=df.drop('Cancer_Type',axis=1)
y=df['Cancer_Type']

##now scale the data
scale=StandardScaler()
scaled_df=scale.fit_transform(x)

##now train and test the model
xtrain,xtest,ytrain,ytest=train_test_split(scaled_df,y,test_size=0.2,random_state=42)
##create a model
model=RandomForestClassifier(criterion='entropy',max_depth=17,max_features=5)
model.fit(xtrain,ytrain)
##now predict the value
pred=model.predict(xtest)
###now evaluate the model
accuracy=accuracy_score(ytest,pred)
print(accuracy)

