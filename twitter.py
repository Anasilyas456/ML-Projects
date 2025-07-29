########import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from sklearn.preprocessing import LabelEncoder,StandardScaler
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

####load the data
train_data=pd.read_csv("C:\\Users\\QHS 006\\Desktop\\ict research paper\\twitter_training.csv", encoding="utf-8", header=None, skiprows=1)
val_data=pd.read_csv("C:\\Users\\QHS 006\\Desktop\\ict research paper\\twitter_validation.csv", encoding="utf-8", header=None, skiprows=1)
train_data.columns = ["ID", "Topic", "Sentiment", "Tweet"]
val_data.columns = ["ID", "Topic", "Sentiment", "Tweet"]
####now concatenate both data
df=pd.concat([train_data,val_data],ignore_index=True)
print(df.head())
print(df.columns)
print(df.info)
print(df.isnull().sum())
######now replace the empty columns
df["Tweet"].fillna("", inplace=True) 
df["Tweet"] = df["Tweet"].astype(str)
print(df.isnull().sum())

#######Function to clean text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"@\w+|\#", "", text)  # Remove mentions and hashtags
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    return text

# Apply text cleaning
df['Tweet']=df["Tweet"].apply(clean_text)
print(df.head())
#####now encode the data
encoder=LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype == 'categorical':
        df[col]=encoder.fit_transform(df[col])
print(df.head())
plt.plot(df)
# plt.show()
########now standardize the data
# Select columns to scale (excluding 'Sentiment')
columns_to_scale = df.drop(columns=['Sentiment'])

# Apply scaling to the selected columns
scaler = StandardScaler()
scaled_data = scaler.fit_transform(columns_to_scale)

# Rebuild the DataFrame with the scaled data
data = pd.DataFrame(scaled_data, columns=columns_to_scale.columns)

# Add the 'Sentiment' column back to the DataFrame without scaling
data['Sentiment'] = df['Sentiment']

# Display the first few rows
print(data.head())

# Plot the scaled data
plt.plot(data)
# plt.show()

#####now remove the outliers
z_score=np.abs(stats.zscore(data))
threshold=3
outliers=np.where(z_score>threshold)
data_cleaned = data.drop(outliers[0])
data1=pd.DataFrame(data_cleaned)
print(data1.head())
plt.plot(data1)
# plt.show()
###now make x and y variable
x=data1.drop('Sentiment',axis=1)
y=data1["Sentiment"]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
model=RandomForestClassifier(criterion='log_loss',max_depth=100,max_features='log2')
model.fit(xtrain,ytrain)
pred=model.predict(xtest)
accuracy=accuracy_score(ytest,pred)
print(accuracy)
