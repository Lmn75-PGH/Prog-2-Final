import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

url = 'https://github.com/Lmn75-PGH/Prog-2-Final/social_media_usage.csv?raw=true'
s = pd.read_csv(url,index_col=0)

#s = pd.read_csv('C:\\Users\\lmnat\\OneDrive\\Desktop\\Prog 2 Final\\social_media_usage.csv')

#s

def clean_sm(x):
    # Use np.where to check if x is equal to 1
    # If yes, make x = 1, otherwise make x = 0
    x = np.where(x == 1, 1, 0)
    # Return x
    return x

print (clean_sm(s.web1h))

ss = s[["income", "educ2", "par", "marital", "gender", "age", "web1h"]]
#print(ss)

ss['female'] = np.where(ss['gender'] == 1, 1, 0)
ss['sm_li'] = clean_sm(s.web1h)
ss['married'] = np.where(ss['marital'] == 1, 1, 0)
ss['parent'] = np.where(ss['par'] == 1, 1, 0)
ss['income'] = ss['income'].replace([98, 99], np.nan)
ss['education'] = ss['educ2'].replace([98, 99], np.nan)
ss['age'] = np.where(ss['age'] > 98, np.nan, ss['age'])

#print(ss)

ss = ss[["income", "education", "parent", "married", "female", "age", "sm_li"]]

ss = ss.dropna()

sns.heatmap(ss.isnull(), cbar=False)

sns.countplot(x='sm_li', data=ss)

sns.countplot(x='sm_li', hue='female', data=ss)

sns.countplot(x='sm_li', hue='education', data=ss)

sns.countplot(x='sm_li', hue='parent', data=ss)

sns.countplot(x='sm_li', hue='married', data=ss)

plt.hist(ss['age'])

#ss.dtypes

y_data = ss['sm_li']

x_data = ss.drop('sm_li', axis = 1)

from sklearn.model_selection import train_test_split

x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x_data, y_data, test_size = 0.2)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(x_training_data, y_training_data)

predictions = model.predict(x_test_data)

from sklearn.metrics import classification_report

classification_report(y_test_data, predictions)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test_data, predictions))

print(model.coef_)

print(model.intercept_)

pd.DataFrame(zip(x_data.columns, model.coef_[0]), columns=['feature', 'coef'])

#Start app to grab user inputs

st.title ("Hello User!!!")

st.header("Welcome to my Predictive App")
st.subheader("Please supply some information about yourself and we will try to predict if you are a LinkedIn User")

gender = st.radio("Select Gender: ", ('Male', 'Female'))

if (gender == 'Female'):
    gen_num=1
    st.success("Female")
else:
    gen_num=0
    st.success("Male")

education = st.selectbox("Highest Level of Education: ", ['Less than high school (Grades 1-8 or no formal schooling)', 'High school incomplete (Grades 9-11 or Grade 12 with NO diploma)', 'High school graduate (Grade 12 with diploma or GED certificate)', 'Some college, no degree (includes some community college)','Two-year associate degree from a college or university', 'Four-year college or university degree/Bachelors degree (e.g., BS, BA, AB)','Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)','Postgraduate or professional degree,including masters, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)']) 

st.write("Your education level is: ", education)

if (education == 'Less than high school (Grades 1-8 or no formal schooling)'):
    edu_num=1
if (education == 'High school incomplete (Grades 9-11 or Grade 12 with NO diploma)'):
    edu_num=2    
if (education == 'High school graduate (Grade 12 with diploma or GED certificate)'):
    edu_num=3    
if (education == 'Some college, no degree (includes some community college)'):
    edu_num=4     
if (education == 'Two-year associate degree from a college or university'):
    edu_num=5    
if (education == 'Four-year college or university degree/Bachelors degree (e.g., BS, BA, AB)'):
    edu_num=6 
if (education == 'Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)'):
    edu_num=7
if (education == 'Postgraduate or professional degree,including masters, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)'):
    edu_num=8


married= st.radio("Are you married?: ", ('Yes', 'No'))

if (married == 'Yes'):
    mar_num=1
    st.success("Yes- I am married")
else:
    mar_num=0
    st.success("No - I am not married")

kids = st.radio("Do you have children?: ", ('Yes', 'No'))

if (kids == 'Yes'):
    kids_num=1
    st.success("Yes- I have children")
else:
    kids_num=0
    st.success("No - I do not have children")    

age = st.slider("Select your Age", 1, 98)    
st.text('Selected: {}'.format(age))

income = st.selectbox("Household Income: ", ['Less than $10,000', "10 to under $20,000", "20 to under $30,000", "30 to under $40,000", "40 to under $50,000", "50 to under $75,000", "75 to under $100,000", "100 to under $150,000", "$150,000 or More"]) 

if (income == 'Less than $10,000'):
    income_num=1
if (income == '10 to under $20,000'):
    income_num=2    
if (income == '20 to under $30,000'):
    income_num=3    
if (income == '30 to under $40,000'):
    income_num=4     
if (income == '40 to under $50,000'):
    income_num=5    
if (income == '50 to under $75,000'):
    income_num=6 
if (income == '75 to under $100,000'):
    income_num=7
if (income == '100 to under $150,000'):
    income_num=8
if (income == '$150,000 or More'):
    income_num=9   

  


st.write("Your household income is: ", income)

#prediction formula

prob=-2.74009333+(income_num*0.176594)+(edu_num*0.400454)+(kids_num*0.153387)+(mar_num*0.058476)+(gen_num*0.283117)+(age*-0.030801)

import math as m
p=m.exp(prob)/(1+m.exp(prob))                                                     

st.write("Your Probability of being a linked in user is: ", p)

if p >= 0.5:
    st.subheader("You are likely a LinkedIn user!")
else:
    st.subheader("You are likely not a Linked In user")
