##############resume screening app############
########import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

##now load the data
df=pd.read_csv(r'C:\Users\QHS 006\Downloads\archive (1)\UpdatedResumeDataSet.csv')
# print(df.head())
# print(df['Category'].value_counts())
# print(df['Category'].unique())
# print(df['Resume'][0])

###now clean the text by regex


def clean_resume(resume_text):
    # Remove URLs
    resume_text = re.sub(r'http[s]?://\S+', '', resume_text)  # Match URLs
    
    # Remove email addresses (simple pattern for email)
    resume_text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,4}\b', '', resume_text)
    
    # Remove phone numbers (basic pattern)
    resume_text = re.sub(r'\(?\+?[0-9]*\)?[0-9_\- \(\)]*', '', resume_text)

    # Remove all symbols (punctuation marks, etc.)
    resume_text = re.sub(r'[^\w\s]', '', resume_text)  # Remove everything except letters, numbers, and spaces

    # Remove multiple spaces and replace with a single space
    resume_text = re.sub(r'\s+', ' ', resume_text).strip()
    
    return resume_text
df['Resume'] = df['Resume'].apply(clean_resume)

######now i encode the output data 
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(df['Category'])


# unique_labels = np.unique(y)
# # print("Unique Encoded Labels:", unique_labels)
# original_categories = le.inverse_transform(np.unique(y))
# # print("Unique Labels with Categories:")
# for label, category in zip(np.unique(y), original_categories):
    # print(f"Encoded Label: {label}, Original Category: {category}")
    


#######now i vectorize an input text data

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
x=tfidf.fit_transform(df['Resume'])


######now i train and split it
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)

###mow create a model and evaluate it.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model=LogisticRegression()
model.fit(xtrain,ytrain)
pred=model.predict(xtest)
print(accuracy_score(ytest,pred))

######now save the models 
import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(model,open('model.pkl','wb'))



my_resume="""James Brown
Email: james.brown@example.com
Phone: +1432167890
LinkedIn: linkedin.com/in/jamesbrown
Website: jamesbrownportfolio.com

### Professional Summary
Dynamic Sales Manager with 5+ years of experience in leading sales teams, building customer relationships, and achieving revenue targets. Expertise in market analysis, team management, and product positioning. Adept at driving sales growth and expanding market share in competitive industries.

### Skills
- Sales Strategies: B2B, B2C, Lead Generation
- CRM Tools: Salesforce, HubSpot, Zoho
- Market Analysis & Forecasting
- Negotiation and Closing Deals
- Leadership and Team Management

### Work Experience
#### Sales Manager
XYZ Enterprises | July 2018 – Present
- Managed a team of 10 sales representatives and increased revenue by 20% YoY.
- Developed sales strategies to attract new clients and retain existing customers.
- Conducted market research to understand trends and adjust sales tactics accordingly.
- Achieved top sales manager status in Q2 2020 by exceeding targets by 30%.

#### Sales Executive
ABC Products Ltd. | January 2015 – June 2018
- Generated leads and secured contracts with small and medium-sized businesses.
- Worked closely with the marketing team to develop promotional strategies.
- Managed client relationships, ensuring high levels of satisfaction and repeat business.

### Education
Bachelor of Business Administration
XYZ University | Graduated: May 2014

### Certifications
- Certified Sales Leader – Sales Management Association
- Salesforce Certified

### Personal Projects
- **Sales Training Program**: Developed and conducted a sales training program for new hires.

### References
Available upon request.


"""

# ######load the model

model=pickle.load(open('model.pkl','rb'))
tfidf=pickle.load(open('vectorizer.pkl','rb'))

#####clean the input resume
cleaned_resume=clean_resume(my_resume)

######now vectorize the resume
input_feature=tfidf.transform([cleaned_resume])
####make a prediction
predicted=model.predict(input_feature)[0]
#####map category id into category name
mapping={
0: 'Advocate',
 1: 'Arts',
 2: 'Automation Testing',
 3: 'Blockchain',
 4: 'Business Analyst',
 5: 'Civil Engineer',
 6: 'Data Science',
 7: 'Database',
 8: 'DevOps Engineer',
 9: 'DotNet Developer',
 10: 'ETL Developer',
 11: 'Electrical Engineering',
 12: 'HR',
 13: 'Hadoop',
 14: 'Health and fitness',
 15: 'Java Developer',
 16: 'Mechanical Engineer',
 17: 'Network Security Engineer',
 18: 'Operations Manager',
 19: 'PMO',
 20: 'Python Developer',
 21: 'SAP Developer',
 22: 'Sales',
 23: 'Testing',
 24: 'Web Designing',
}


category_name=mapping.get(predicted,'unknown')
print("predicted category:",category_name)

